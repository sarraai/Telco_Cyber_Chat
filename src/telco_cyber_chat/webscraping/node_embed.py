"""
node_embedder.py

Use the remote BGE-M3 embedding service (embed_loader.py)
to embed nodes (TextNodes + MITRE RelationshipNodes) with dense + sparse vectors.

OPTIMIZED FOR COLAB: lower concurrency, batching, retries.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from llama_index.core.schema import TextNode
from ..embed_loader import get_hybrid_embeddings

import logging
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# NodeLike helpers (supports TextNode + your RelationshipNode dataclass)
# -----------------------------------------------------------------------------
def _node_id(node: Any) -> str:
    if hasattr(node, "id_"):  # TextNode
        return str(getattr(node, "id_"))
    if hasattr(node, "id"):  # RelationshipNode
        return str(getattr(node, "id"))
    raise TypeError(f"Unsupported node type for embedding: {type(node)} (missing id/id_)")

def _node_text(node: Any) -> str:
    if hasattr(node, "text"):
        return str(getattr(node, "text") or "").strip()
    raise TypeError(f"Unsupported node type for embedding: {type(node)} (missing text)")

def _node_metadata(node: Any) -> Dict[str, Any]:
    m = getattr(node, "metadata", None)
    return dict(m or {})


async def _embed_single(
    node: Any,
    sem: asyncio.Semaphore,
    retry_count: int = 3,
    delay_between_retries: float = 2.0,
) -> Tuple[str, Optional[np.ndarray], Optional[Dict[int, float]]]:
    """
    Embed a single node using the remote BGE-M3 service with retries.

    Returns:
        (node_id, dense, sparse)
    """
    nid = _node_id(node)
    text = _node_text(node)

    if not text:
        # Nothing to embed
        return nid, None, None

    async with sem:
        for attempt in range(retry_count):
            try:
                dense, sparse = await get_hybrid_embeddings(text)

                # small delay to reduce burst pressure
                await asyncio.sleep(0.1)

                return nid, dense, sparse

            except Exception as e:
                logger.warning(f"Attempt {attempt+1}/{retry_count} failed for node {nid}: {e}")

                if attempt < retry_count - 1:
                    await asyncio.sleep(delay_between_retries * (attempt + 1))
                else:
                    logger.error(f"All retries failed for node {nid}")
                    return nid, None, None

    return nid, None, None


async def embed_nodes_hybrid(
    nodes: List[Any],                      # ✅ TextNode OR RelationshipNode
    concurrency: int = 2,
    batch_size: Optional[int] = 20,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Dict[str, object]]:
    """
    Embed nodes with dense + sparse BGE-M3 embeddings.

    Returns:
        Dict[node_id, {"dense": np.ndarray | None, "sparse": Dict[int,float] | None}]
    """
    if not nodes:
        return {}

    total_nodes = len(nodes)
    logger.info(f"Starting to embed {total_nodes} nodes (concurrency={concurrency}, batch_size={batch_size})")

    if batch_size and batch_size < len(nodes):
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

        if completed % 10 == 0:
            logger.info(f"Progress: {completed}/{total_nodes} nodes embedded")

    logger.info(f"✅ Completed embedding {total_nodes} nodes")
    return out


async def _embed_nodes_in_batches(
    nodes: List[Any],
    batch_size: int,
    concurrency: int,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Dict[str, object]]:
    total_nodes = len(nodes)
    all_results: Dict[str, Dict[str, object]] = {}

    for i in range(0, total_nodes, batch_size):
        batch = nodes[i : i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_nodes + batch_size - 1) // batch_size

        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} nodes)")

        sem = asyncio.Semaphore(concurrency)
        tasks = [_embed_single(node, sem) for node in batch]
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
    nodes: List[Any],
    concurrency: int = 2,
    batch_size: int = 20,
    show_progress: bool = True,
) -> Dict[str, Dict[str, object]]:
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
