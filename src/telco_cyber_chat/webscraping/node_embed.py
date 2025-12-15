"""
node_embedder.py

Use the remote BGE-M3 embedding service (embed_loader.py)
to embed LlamaIndex TextNodes (dense + sparse) for Qdrant.

OPTIMIZED FOR COLAB: Lower concurrency, batching, retries
"""

from __future__ import annotations

import asyncio
import time
from typing import List, Dict, Tuple, Optional

import numpy as np
from llama_index.core.schema import TextNode

from ..embed_loader import get_hybrid_embeddings

import logging
logger = logging.getLogger(__name__)


async def _embed_single_node(
    node: TextNode,
    sem: asyncio.Semaphore,
    retry_count: int = 3,
    delay_between_retries: float = 2.0,
) -> Tuple[str, Optional[np.ndarray], Optional[Dict[int, float]]]:
    """
    Embed a single TextNode using the remote BGE-M3 service with retries.

    Returns:
        (node_id, dense, sparse)
    """
    async with sem:
        for attempt in range(retry_count):
            try:
                dense, sparse = await get_hybrid_embeddings(node.text)
                
                # Small delay to prevent overwhelming Colab
                await asyncio.sleep(0.1)
                
                return node.id_, dense, sparse
                
            except Exception as e:
                logger.warning(f"Attempt {attempt + 1}/{retry_count} failed for node {node.id_}: {e}")
                
                if attempt < retry_count - 1:
                    await asyncio.sleep(delay_between_retries * (attempt + 1))
                else:
                    logger.error(f"All retries failed for node {node.id_}")
                    return node.id_, None, None
    
    return node.id_, None, None


async def embed_nodes_hybrid(
    nodes: List[TextNode],
    concurrency: int = 2,  # REDUCED from 5 to 2 for Colab
    batch_size: Optional[int] = None,  # Process in smaller batches
    progress_callback: Optional[callable] = None,
) -> Dict[str, Dict[str, object]]:
    """
    Embed a list of TextNodes with dense + sparse BGE-M3 embeddings.
    
    OPTIMIZED FOR COLAB:
    - Lower default concurrency (2 instead of 5)
    - Optional batching to prevent overwhelming the server
    - Progress tracking
    - Better error handling

    Args:
        nodes: list of LlamaIndex TextNode
        concurrency: max number of parallel HTTP calls (keep low for Colab)
        batch_size: if set, process nodes in batches of this size
        progress_callback: optional function(current, total) for progress updates

    Returns:
        Dict[node_id, {"dense": np.ndarray | None, "sparse": Dict[int, float] | None}]
    """
    if not nodes:
        return {}

    total_nodes = len(nodes)
    logger.info(f"Starting to embed {total_nodes} nodes with concurrency={concurrency}")

    # If batch_size is specified, process in batches
    if batch_size and batch_size < len(nodes):
        return await _embed_nodes_in_batches(
            nodes, 
            batch_size=batch_size,
            concurrency=concurrency,
            progress_callback=progress_callback
        )

    # Process all at once with limited concurrency
    sem = asyncio.Semaphore(concurrency)
    tasks = [_embed_single_node(node, sem) for node in nodes]

    # Track progress
    completed = 0
    out: Dict[str, Dict[str, object]] = {}
    
    for coro in asyncio.as_completed(tasks):
        node_id, dense, sparse = await coro
        out[node_id] = {
            "dense": dense,
            "sparse": sparse,
        }
        
        completed += 1
        if progress_callback:
            progress_callback(completed, total_nodes)
        
        if completed % 10 == 0:
            logger.info(f"Progress: {completed}/{total_nodes} nodes embedded")

    logger.info(f"✅ Completed embedding {total_nodes} nodes")
    return out


async def _embed_nodes_in_batches(
    nodes: List[TextNode],
    batch_size: int,
    concurrency: int,
    progress_callback: Optional[callable] = None,
) -> Dict[str, Dict[str, object]]:
    """
    Process nodes in smaller batches to avoid overwhelming Colab.
    """
    total_nodes = len(nodes)
    all_results: Dict[str, Dict[str, object]] = {}
    
    for i in range(0, total_nodes, batch_size):
        batch = nodes[i:i + batch_size]
        batch_num = i // batch_size + 1
        total_batches = (total_nodes + batch_size - 1) // batch_size
        
        logger.info(f"Processing batch {batch_num}/{total_batches} ({len(batch)} nodes)")
        
        sem = asyncio.Semaphore(concurrency)
        tasks = [_embed_single_node(node, sem) for node in batch]
        
        results = await asyncio.gather(*tasks, return_exceptions=False)
        
        for node_id, dense, sparse in results:
            all_results[node_id] = {
                "dense": dense,
                "sparse": sparse,
            }
        
        if progress_callback:
            progress_callback(min(i + batch_size, total_nodes), total_nodes)
        
        # Delay between batches to let Colab breathe
        if i + batch_size < total_nodes:
            logger.info("Waiting 2 seconds before next batch...")
            await asyncio.sleep(2)
    
    return all_results


# ============================================================================
# Convenience wrapper with progress bar (optional)
# ============================================================================
def embed_nodes_hybrid_sync(
    nodes: List[TextNode],
    concurrency: int = 2,
    batch_size: int = 20,  # Process 20 nodes at a time
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
                    progress_callback=progress_fn
                )
            )
            pbar.close()
            return result
            
        except ImportError:
            logger.warning("tqdm not available, falling back to basic progress")
    
    # No progress bar
    def simple_progress(current, total):
        if current % 10 == 0:
            print(f"Progress: {current}/{total}")
    
    return asyncio.run(
        embed_nodes_hybrid(
            nodes,
            concurrency=concurrency,
            batch_size=batch_size,
            progress_callback=simple_progress
        )
    )
