"""
node_embedder.py

Use the remote BGE-M3 embedding service (embed_loader.py)
to embed LlamaIndex TextNodes (dense + sparse) for Qdrant.

UPDATED:
- Supports embedding relationship nodes (RelationshipNode dataclass OR dicts),
  by converting them into (node_id, text) pairs.
- Still works exactly the same for TextNode.

OPTIMIZED FOR COLAB: Lower concurrency, batching, retries
"""

from __future__ import annotations

import asyncio
from typing import List, Dict, Tuple, Optional, Any, Iterable

import numpy as np
from llama_index.core.schema import TextNode

from ..embed_loader import get_hybrid_embeddings

import logging
logger = logging.getLogger(__name__)


# =============================================================================
# Helpers: accept TextNode / RelationshipNode / dict and normalize into (id, text)
# =============================================================================

def _coerce_str(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x.strip()
    return str(x).strip()


def _relationship_text_from_dict(d: Dict[str, Any]) -> str:
    """Fallback text builder if you pass relationship dicts."""
    rtype = _coerce_str(d.get("relationship_type"))
    src = _coerce_str(d.get("source_ref"))
    tgt = _coerce_str(d.get("target_ref"))
    desc = _coerce_str(d.get("description"))

    parts = []
    if rtype:
        parts.append(f"Relationship type: {rtype}")
    if src:
        parts.append(f"Source: {src}")
    if tgt:
        parts.append(f"Target: {tgt}")
    if desc:
        parts.append(desc)
    return "\n".join(parts).strip()


def _normalize_node_like(obj: Any) -> Tuple[str, str]:
    """
    Convert supported node-like objects into:
        (node_id: str, text: str)

    Supported:
      - TextNode: uses .id_ and .text
      - RelationshipNode dataclass: expects .id and .text
      - dict: expects "id"/"id_" + "text", otherwise builds relationship text
    """
    # 1) LlamaIndex TextNode
    if isinstance(obj, TextNode):
        return _coerce_str(obj.id_), _coerce_str(obj.text)

    # 2) RelationshipNode-like dataclass/object
    if hasattr(obj, "text") and (hasattr(obj, "id_") or hasattr(obj, "id")):
        raw_id = getattr(obj, "id_", None)
        if raw_id is None:
            raw_id = getattr(obj, "id", None)
        return _coerce_str(raw_id), _coerce_str(getattr(obj, "text", ""))

    # 3) dict node
    if isinstance(obj, dict):
        node_id = _coerce_str(obj.get("id_") or obj.get("id") or obj.get("node_id"))
        text = _coerce_str(obj.get("text"))
        if not text:
            # try to build relationship text if no explicit text
            text = _relationship_text_from_dict(obj)
        return node_id, text

    raise TypeError(
        f"Unsupported node type for embedding: {type(obj)}. "
        "Expected TextNode, RelationshipNode-like, or dict."
    )


def _normalize_nodes(nodes: Iterable[Any]) -> List[Tuple[str, str]]:
    """
    Return a clean list of (node_id, text), skipping empty ones.
    """
    out: List[Tuple[str, str]] = []
    for obj in nodes:
        try:
            node_id, text = _normalize_node_like(obj)
        except Exception as e:
            logger.warning("Skipping node (cannot normalize): %s", e)
            continue

        if not node_id:
            logger.warning("Skipping node with empty id.")
            continue
        if not text:
            logger.warning("Skipping node %s with empty text.", node_id)
            continue

        out.append((node_id, text))
    return out


# =============================================================================
# Core embed logic
# =============================================================================

async def _embed_single_text(
    node_id: str,
    text: str,
    sem: asyncio.Semaphore,
    retry_count: int = 3,
    delay_between_retries: float = 2.0,
) -> Tuple[str, Optional[np.ndarray], Optional[Dict[int, float]]]:
    """
    Embed a single text payload using the remote BGE-M3 service with retries.

    Returns:
        (node_id, dense, sparse)
    """
    async with sem:
        for attempt in range(retry_count):
            try:
                dense, sparse = await get_hybrid_embeddings(text)

                # Small delay to prevent overwhelming Colab / tunnel
                await asyncio.sleep(0.1)

                return node_id, dense, sparse

            except Exception as e:
                logger.warning(
                    "Attempt %d/%d failed for node %s: %s",
                    attempt + 1, retry_count, node_id, e
                )

                if attempt < retry_count - 1:
                    await asyncio.sleep(delay_between_retries * (attempt + 1))
                else:
                    logger.error("All retries failed for node %s", node_id)
                    return node_id, None, None

    return node_id, None, None


async def embed_nodes_hybrid(
    nodes: List[Any],  # ✅ now accepts relationship nodes too
    concurrency: int = 2,               # keep low for Colab
    batch_size: Optional[int] = None,   # process in smaller batches
    progress_callback: Optional[callable] = None,
) -> Dict[str, Dict[str, object]]:
    """
    Embed a list of node-like objects with dense + sparse BGE-M3 embeddings.

    Accepts:
      - TextNode
      - RelationshipNode-like objects (has .id or .id_ and .text)
      - dict nodes (id + text, or relationship fields)

    Returns:
        Dict[node_id, {"dense": np.ndarray | None, "sparse": Dict[int, float] | None}]
    """
    pairs = _normalize_nodes(nodes)
    if not pairs:
        return {}

    total_nodes = len(pairs)
    logger.info("Starting to embed %d nodes with concurrency=%d", total_nodes, concurrency)

    if batch_size and batch_size < total_nodes:
        return await _embed_pairs_in_batches(
            pairs,
            batch_size=batch_size,
            concurrency=concurrency,
            progress_callback=progress_callback,
        )

    sem = asyncio.Semaphore(concurrency)
    tasks = [_embed_single_text(node_id, text, sem) for node_id, text in pairs]

    completed = 0
    out: Dict[str, Dict[str, object]] = {}

    for coro in asyncio.as_completed(tasks):
        node_id, dense, sparse = await coro
        out[node_id] = {"dense": dense, "sparse": sparse}

        completed += 1
        if progress_callback:
            progress_callback(completed, total_nodes)

        if completed % 10 == 0:
            logger.info("Progress: %d/%d nodes embedded", completed, total_nodes)

    logger.info("✅ Completed embedding %d nodes", total_nodes)
    return out


async def _embed_pairs_in_batches(
    pairs: List[Tuple[str, str]],
    batch_size: int,
    concurrency: int,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Dict[str, object]]:
    """
    Process nodes in smaller batches to avoid overwhelming Colab.
    """
    total_nodes = len(pairs)
    all_results: Dict[str, Dict[str, object]] = {}

    for i in range(0, total_nodes, batch_size):
        batch = pairs[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_nodes + batch_size - 1) // batch_size

        logger.info("Processing batch %d/%d (%d nodes)", batch_num, total_batches, len(batch))

        sem = asyncio.Semaphore(concurrency)
        tasks = [_embed_single_text(node_id, text, sem) for node_id, text in batch]
        results = await asyncio.gather(*tasks, return_exceptions=False)

        for node_id, dense, sparse in results:
            all_results[node_id] = {"dense": dense, "sparse": sparse}

        if progress_callback:
            progress_callback(min(i + batch_size, total_nodes), total_nodes)

        if i + batch_size < total_nodes:
            logger.info("Waiting 2 seconds before next batch...")
            await asyncio.sleep(2)

    return all_results


def embed_nodes_hybrid_sync(
    nodes: List[Any],   # ✅ now accepts relationship nodes too
    concurrency: int = 2,
    batch_size: int = 20,
    show_progress: bool = True,
) -> Dict[str, Dict[str, object]]:
    """
    Synchronous wrapper for embed_nodes_hybrid with progress tracking.

    Recommended for Colab: batch_size=20, concurrency=2
    """
    if show_progress:
        try:
            from tqdm import tqdm
            pbar = tqdm(total=len(nodes), desc="Embedding nodes")

            def progress_fn(current, total):
                pbar.n = current
                pbar.refresh()

            result = asyncio.run(
                embed_nodes_hybrid(
                    nodes,
                    concurrency=concurrency,
                    batch_size=batch_size,
                    progress_callback=progress_fn,
                )
            )
            pbar.close()
            return result

        except ImportError:
            logger.warning("tqdm not available, falling back to basic progress")

    def simple_progress(current, total):
        if current % 10 == 0:
            print(f"Progress: {current}/{total}")

    return asyncio.run(
        embed_nodes_hybrid(
            nodes,
            concurrency=concurrency,
            batch_size=batch_size,
            progress_callback=simple_progress,
        )
    )
