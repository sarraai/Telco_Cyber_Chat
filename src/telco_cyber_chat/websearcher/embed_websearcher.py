"""
embed_websearcher.py

Embed LlamaIndex TextNodes using a REMOTE BGE-M3 (dense+sparse) service exposed via LangServe.

Depends on:
  - telco_cyber_chat.embed_loader.get_hybrid_embeddings (aiohttp -> /embed/invoke)
  - numpy

Env:
  - BGE_EMBEDDING_URL  (recommended: https://<ngrok-domain>/embed  OR /embed/invoke)
  - BGE_EMBEDDING_TIMEOUT (seconds, default 180 in embed_loader)

Goal:
  - keep heavy embedding model OUT of LangSmith deployment
  - embed nodes using your Colab/ngrok service safely (concurrency + retries)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from llama_index.core.schema import TextNode

from telco_cyber_chat.embed_loader import get_hybrid_embeddings

logger = logging.getLogger(__name__)

SparseDict = Dict[int, float]


@dataclass
class HybridEmbedding:
    """Container for dense and sparse embeddings."""
    dense: Optional[np.ndarray]
    sparse: Optional[SparseDict]


def _node_text(node: TextNode) -> str:
    """Extract text from a TextNode."""
    return str(getattr(node, "text", "") or "").strip()


def _node_metadata(node: TextNode) -> Dict[str, Any]:
    """Extract metadata from a TextNode."""
    md = getattr(node, "metadata", None)
    return md if isinstance(md, dict) else {}


def _get_existing_id(node: TextNode) -> Optional[str]:
    """Get the existing ID from a TextNode."""
    v = getattr(node, "id_", None)
    v = str(v).strip() if v is not None else ""
    return v or None


def _set_node_id(node: TextNode, new_id: str) -> None:
    """Set the ID on a TextNode."""
    try:
        node.id_ = new_id
    except Exception:
        pass


def _generate_stable_id(node: TextNode) -> str:
    """Generate a stable SHA256 hash ID for a TextNode."""
    md = _node_metadata(node)
    url = md.get("url")
    if isinstance(url, str) and url.strip():
        raw = f"TextNode|url|{url.strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    text = _node_text(node)
    try:
        md_norm = json.dumps(md, sort_keys=True, ensure_ascii=False)
    except Exception:
        md_norm = str(md)

    raw = f"TextNode|text|{text}|meta|{md_norm}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def ensure_textnode_ids(nodes: List[TextNode]) -> None:
    """Ensure all nodes have stable IDs."""
    for n in nodes:
        if not _get_existing_id(n):
            _set_node_id(n, _generate_stable_id(n))


async def _embed_one(
    node: TextNode,
    sem: asyncio.Semaphore,
    *,
    retry_count: int = 3,
    delay_between_retries: float = 5.0,  # ✅ Increased from 2.0 to 5.0
    per_call_sleep: float = 0.50,  # ✅ Increased from 0.10 to 0.50 for better rate limiting
) -> Tuple[str, HybridEmbedding]:
    """
    Embed a single TextNode with retry logic.
    
    Args:
        node: TextNode to embed
        sem: Semaphore for concurrency control
        retry_count: Number of retries on failure (default 3)
        delay_between_retries: Base delay between retries with exponential backoff (default 5.0s)
        per_call_sleep: Sleep after successful call for rate limiting (default 0.50s)
        
    Returns:
        Tuple of (node_id, HybridEmbedding)
    """
    nid = _get_existing_id(node) or _generate_stable_id(node)
    _set_node_id(node, nid)

    text = _node_text(node)
    if not text:
        logger.warning(f"Node {nid} has no text, returning empty embedding")
        return nid, HybridEmbedding(dense=None, sparse=None)

    async with sem:
        for attempt in range(retry_count):
            try:
                # Call remote embedding service
                dense, sparse = await get_hybrid_embeddings(text)

                # Rate limiting sleep after successful call
                if per_call_sleep:
                    await asyncio.sleep(per_call_sleep)

                logger.debug(f"✅ Successfully embedded node {nid} (attempt {attempt + 1})")
                return nid, HybridEmbedding(dense=dense, sparse=sparse)

            except Exception as e:
                logger.warning(
                    "Embedding failed (attempt %d/%d) for node=%s: %s",
                    attempt + 1,
                    retry_count,
                    nid,
                    str(e)[:100],  # Truncate long error messages
                )
                if attempt < retry_count - 1:
                    # Exponential backoff: 5s, 10s, 15s
                    await asyncio.sleep(delay_between_retries * (attempt + 1))
                else:
                    logger.error(f"❌ All {retry_count} retries failed for node={nid}")

        # All retries exhausted
        return nid, HybridEmbedding(dense=None, sparse=None)


async def embed_textnodes_hybrid(
    nodes: List[TextNode],
    *,
    concurrency: int = 1,  # ✅ Reduced from 2 to 1 to avoid overwhelming Colab
    batch_size: int = 10,  # ✅ Reduced from 20 to 10
    retry_count: int = 3,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, HybridEmbedding]:
    """
    Embed a list of TextNodes using remote BGE-M3 service.
    
    Process nodes in batches with controlled concurrency to avoid overwhelming
    the remote embedding service.
    
    Args:
        nodes: List of TextNodes to embed
        concurrency: Number of concurrent embedding requests (default 1 for Colab)
        batch_size: Process nodes in batches of this size (default 10)
        retry_count: Number of retries per node (default 3)
        progress_callback: Optional callback(done, total) for progress updates
        
    Returns:
        Dict mapping node_id to HybridEmbedding
    """
    if not nodes:
        return {}

    ensure_textnode_ids(nodes)

    total = len(nodes)
    out: Dict[str, HybridEmbedding] = {}
    
    logger.info(f"Starting to embed {total} nodes (concurrency={concurrency}, batch_size={batch_size})")

    for start in range(0, total, batch_size):
        batch = nodes[start : start + batch_size]
        batch_num = (start // batch_size) + 1
        total_batches = (total + batch_size - 1) // batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} nodes)")
        
        sem = asyncio.Semaphore(concurrency)

        tasks = [
            _embed_one(
                n,
                sem,
                retry_count=retry_count,
                delay_between_retries=5.0,  # ✅ Increased from 2.0
                per_call_sleep=0.50,  # ✅ Increased from 0.10
            )
            for n in batch
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Process results and handle exceptions
        success_count = 0
        fail_count = 0
        
        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Unexpected exception during embedding: {result}")
                fail_count += 1
                continue
            
            nid, emb = result
            out[nid] = emb
            
            if emb.dense is not None:
                success_count += 1
            else:
                fail_count += 1

        done = min(start + batch_size, total)
        logger.info(f"Batch {batch_num} complete: {success_count} success, {fail_count} failed")
        
        if progress_callback:
            progress_callback(done, total)

        # Longer sleep between batches to give Colab a break
        if done < total:
            logger.debug("Sleeping 3s before next batch...")
            await asyncio.sleep(3.0)  # ✅ Increased from 2.0

    # Final summary
    total_success = sum(1 for emb in out.values() if emb.dense is not None)
    total_failed = len(out) - total_success
    
    logger.info(f"Embedding complete: {total_success}/{len(nodes)} succeeded, {total_failed} failed")
    
    if total_success == 0:
        logger.error("❌ All embeddings failed! Check your BGE_EMBEDDING_URL and service status")
    
    return out


def _run_coro_safely(coro):
    """Run a coroutine in a safe way that handles existing event loops."""
    try:
        loop = asyncio.get_running_loop()
        running = loop.is_running()
    except RuntimeError:
        running = False

    if not running:
        return asyncio.run(coro)

    try:
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    except Exception as e:
        raise RuntimeError(
            "Running inside an active event loop. "
            "Either `pip install nest_asyncio` or call the async function with `await`."
        ) from e


def embed_textnodes_hybrid_sync(
    nodes: List[TextNode],
    *,
    concurrency: int = 1,  # ✅ Reduced from 2
    batch_size: int = 10,  # ✅ Reduced from 20
    retry_count: int = 3,
    show_progress: bool = True,
) -> Dict[str, HybridEmbedding]:
    """
    Synchronous wrapper for embed_textnodes_hybrid.
    
    Args:
        nodes: List of TextNodes to embed
        concurrency: Number of concurrent embedding requests (default 1)
        batch_size: Process nodes in batches of this size (default 10)
        retry_count: Number of retries per node (default 3)
        show_progress: Whether to show progress bar (default True)
        
    Returns:
        Dict mapping node_id to HybridEmbedding
    """
    if not nodes:
        return {}

    if show_progress:
        try:
            from tqdm import tqdm

            pbar = tqdm(total=len(nodes), desc="Embedding TextNodes")

            def progress_fn(done, total):
                pbar.n = done
                pbar.refresh()

            result = _run_coro_safely(
                embed_textnodes_hybrid(
                    nodes,
                    concurrency=concurrency,
                    batch_size=batch_size,
                    retry_count=retry_count,
                    progress_callback=progress_fn,
                )
            )
            pbar.close()
            return result
        except Exception as e:
            logger.warning("Progress UI fallback: %s", e)

    def simple_progress(done, total):
        if done % 10 == 0 or done == total:  # ✅ Changed from 25 to 10 for more frequent updates
            print(f"Progress: {done}/{total}")

    return _run_coro_safely(
        embed_textnodes_hybrid(
            nodes,
            concurrency=concurrency,
            batch_size=batch_size,
            retry_count=retry_count,
            progress_callback=simple_progress,
        )
    )


def attach_dense_to_nodes(nodes: List[TextNode], results: Dict[str, HybridEmbedding]) -> None:
    """
    Attach dense embeddings to nodes for compatibility with Qdrant ingest.
    (Sparse remains in results[nid].sparse.)
    
    Args:
        nodes: List of TextNodes to update
        results: Dict mapping node_id to HybridEmbedding
    """
    for n in nodes:
        nid = _get_existing_id(n)
        if not nid:
            continue
        emb = results.get(nid)
        if emb and emb.dense is not None:
            try:
                n.embedding = emb.dense.astype(np.float32).tolist()
            except Exception:
                pass
