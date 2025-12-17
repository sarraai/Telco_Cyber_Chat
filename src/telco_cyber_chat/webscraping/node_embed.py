"""
node_embedder.py

Embeds Node-like objects (TextNode + optional RelationshipNode) using a remote
BGE-M3 hybrid service (dense + sparse) via embed_loader.get_hybrid_embeddings().

✅ Does NOT require an existing id on nodes.
- If node.id_ / node.id is missing or empty, we generate a stable one:
  - Prefer metadata["url"] when available
  - Else SHA256(node_type + text + metadata)

- Works for nodes coming from ANY scraper, as long as they have `.text`.
- Optimized for Colab: low concurrency, batching, retries, gentle throttling.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
from typing import Any, Dict, List, Optional, Tuple, Callable

import numpy as np

from ..embed_loader import get_hybrid_embeddings

import logging
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# NodeLike helpers (supports TextNode + your RelationshipNode dataclass)
# -----------------------------------------------------------------------------
def _node_text(node: Any) -> str:
    if hasattr(node, "text"):
        return str(getattr(node, "text") or "").strip()
    raise TypeError(f"Unsupported node type for embedding: {type(node)} (missing text)")


def _node_metadata(node: Any) -> Dict[str, Any]:
    md = getattr(node, "metadata", None)
    return md if isinstance(md, dict) else {}


def _get_existing_id(node: Any) -> Optional[str]:
    # TextNode uses id_
    if hasattr(node, "id_"):
        v = getattr(node, "id_", None)
        v = str(v).strip() if v is not None else ""
        return v or None

    # RelationshipNode (custom) uses id
    if hasattr(node, "id"):
        v = getattr(node, "id", None)
        v = str(v).strip() if v is not None else ""
        return v or None

    return None


def _set_node_id(node: Any, new_id: str) -> None:
    """
    Best-effort: assign generated id back to the node so downstream steps
    (like Qdrant upsert) can reuse it.
    """
    try:
        if hasattr(node, "id_"):
            setattr(node, "id_", new_id)
            return
        if hasattr(node, "id"):
            setattr(node, "id", new_id)
            return
    except Exception:
        # If it fails (immutable model), it's still okay: we'll return results keyed by new_id
        pass


def _generate_stable_id(node: Any) -> str:
    """
    Generate a stable ID if none exists.
    Priority:
      1) metadata["url"] if present (canonical URL is ideal)
      2) SHA256(node_type + text + normalized metadata)
    """
    md = _node_metadata(node)
    url = md.get("url")
    if isinstance(url, str) and url.strip():
        raw = f"{type(node).__name__}|url|{url.strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    text = _node_text(node)

    # Normalize metadata for stable hashing
    try:
        md_norm = json.dumps(md, sort_keys=True, ensure_ascii=False)
    except Exception:
        md_norm = str(md)

    raw = f"{type(node).__name__}|text|{text}|meta|{md_norm}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _node_id(node: Any) -> str:
    """
    Return an ID for this node.
    - Use existing id_ / id if present
    - Otherwise generate one (stable) and try to assign it back to the node
    """
    existing = _get_existing_id(node)
    if existing:
        return existing

    new_id = _generate_stable_id(node)
    _set_node_id(node, new_id)
    return new_id


async def _embed_single(
    node: Any,
    sem: asyncio.Semaphore,
    retry_count: int = 3,
    delay_between_retries: float = 2.0,
    per_call_sleep: float = 0.10,
) -> Tuple[str, Optional[np.ndarray], Optional[Dict[int, float]]]:
    """
    Returns: (node_id, dense, sparse)
    """
    nid = _node_id(node)
    text = _node_text(node)

    if not text:
        return nid, None, None

    async with sem:
        for attempt in range(retry_count):
            try:
                dense, sparse = await get_hybrid_embeddings(text)

                # tiny delay to reduce burst pressure on your tunnel/server
                if per_call_sleep:
                    await asyncio.sleep(per_call_sleep)

                return nid, dense, sparse

            except Exception as e:
                logger.warning("Attempt %d/%d failed for node %s: %s", attempt + 1, retry_count, nid, e)

                if attempt < retry_count - 1:
                    await asyncio.sleep(delay_between_retries * (attempt + 1))
                else:
                    logger.error("All retries failed for node %s", nid)
                    return nid, None, None

    return nid, None, None


async def embed_nodes_hybrid(
    nodes: List[Any],
    *,
    concurrency: int = 2,
    batch_size: Optional[int] = 20,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Dict[str, object]]:
    """
    Returns:
      Dict[node_id, {"dense": np.ndarray | None, "sparse": Dict[int,float] | None}]

    Note: node_id is auto-generated if the node has no id_ / id.
    """
    if not nodes:
        return {}

    total_nodes = len(nodes)
    logger.info("Embedding %d nodes (concurrency=%d, batch_size=%s)", total_nodes, concurrency, batch_size)

    # If batching enabled
    if batch_size and batch_size < total_nodes:
        return await _embed_nodes_in_batches(
            nodes,
            batch_size=batch_size,
            concurrency=concurrency,
            progress_callback=progress_callback,
        )

    sem = asyncio.Semaphore(concurrency)
    tasks = [_embed_single(node, sem) for node in nodes]

    completed = 0
    out: Dict[str, Dict[str, object]] = {}

    for coro in asyncio.as_completed(tasks):
        node_id, dense, sparse = await coro
        out[node_id] = {"dense": dense, "sparse": sparse}

        completed += 1
        if progress_callback:
            progress_callback(completed, total_nodes)

        if completed % 25 == 0:
            logger.info("Progress: %d/%d nodes embedded", completed, total_nodes)

    logger.info("✅ Completed embedding %d nodes", total_nodes)
    return out


async def _embed_nodes_in_batches(
    nodes: List[Any],
    *,
    batch_size: int,
    concurrency: int,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, Dict[str, object]]:
    total_nodes = len(nodes)
    all_results: Dict[str, Dict[str, object]] = {}

    total_batches = (total_nodes + batch_size - 1) // batch_size
    done = 0

    for b in range(total_batches):
        i = b * batch_size
        batch = nodes[i : i + batch_size]
        logger.info("Batch %d/%d (%d nodes)", b + 1, total_batches, len(batch))

        sem = asyncio.Semaphore(concurrency)
        tasks = [_embed_single(node, sem) for node in batch]
        results = await asyncio.gather(*tasks)

        for node_id, dense, sparse in results:
            all_results[node_id] = {"dense": dense, "sparse": sparse}

        done = min(i + batch_size, total_nodes)
        if progress_callback:
            progress_callback(done, total_nodes)

        # pause between batches to protect your remote service
        if b + 1 < total_batches:
            await asyncio.sleep(2.0)

    return all_results


# -----------------------------------------------------------------------------
# Colab/Jupyter-safe sync wrapper
# -----------------------------------------------------------------------------
def _run_coro_safely(coro):
    """
    - In normal Python scripts: uses asyncio.run
    - In notebooks (already-running loop): uses nest_asyncio + run_until_complete
      (if available). Otherwise, you must call the async function with `await`.
    """
    try:
        loop = asyncio.get_running_loop()
        running = loop.is_running()
    except RuntimeError:
        running = False

    if not running:
        return asyncio.run(coro)

    # Notebook case:
    try:
        import nest_asyncio  # pip install nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    except Exception as e:
        raise RuntimeError(
            "You're in a running event loop (notebook). "
            "Either `pip install nest_asyncio` or call: `await embed_nodes_hybrid(...)`."
        ) from e


def embed_nodes_hybrid_sync(
    nodes: List[Any],
    *,
    concurrency: int = 2,
    batch_size: int = 20,
    show_progress: bool = True,
) -> Dict[str, Dict[str, object]]:
    if not nodes:
        return {}

    if show_progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(nodes), desc="Embedding nodes")

            def progress_fn(current, total):
                pbar.n = current
                pbar.refresh()

            result = _run_coro_safely(
                embed_nodes_hybrid(
                    nodes,
                    concurrency=concurrency,
                    batch_size=batch_size,
                    progress_callback=progress_fn,
                )
            )
            pbar.close()
            return result
        except Exception as e:
            logger.warning("Progress UI fallback: %s", e)

    def simple_progress(current, total):
        if current % 25 == 0 or current == total:
            print(f"Progress: {current}/{total}")

    return _run_coro_safely(
        embed_nodes_hybrid(
            nodes,
            concurrency=concurrency,
            batch_size=batch_size,
            progress_callback=simple_progress,
        )
    )
