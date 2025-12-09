"""
node_embedder.py

Use the remote BGE-M3 embedding service (embed_loader.py)
to embed LlamaIndex TextNodes (dense + sparse) for Qdrant.
"""

from __future__ import annotations

import asyncio
from typing import List, Dict, Tuple, Optional

import numpy as np
from llama_index.core.schema import TextNode

from ..embed_loader import get_query_embeddings  # or the correct function name


async def _embed_single_node(
    node: TextNode,
    sem: asyncio.Semaphore,
) -> Tuple[str, Optional[np.ndarray], Optional[Dict[int, float]]]:
    """
    Embed a single TextNode using the remote BGE-M3 service.

    Returns:
        (node_id, dense, sparse)
    """
    async with sem:
        dense, sparse = await get_hybrid_embeddings(node.text)
    return node.id_, dense, sparse


async def embed_nodes_hybrid(
    nodes: List[TextNode],
    concurrency: int = 5,
) -> Dict[str, Dict[str, object]]:
    """
    Embed a list of TextNodes with dense + sparse BGE-M3 embeddings.

    Args:
        nodes: list of LlamaIndex TextNode
        concurrency: max number of parallel HTTP calls

    Returns:
        Dict[node_id, {"dense": np.ndarray | None, "sparse": Dict[int, float] | None}]
    """
    if not nodes:
        return {}

    sem = asyncio.Semaphore(concurrency)
    tasks = [_embed_single_node(node, sem) for node in nodes]

    results = await asyncio.gather(*tasks, return_exceptions=False)

    out: Dict[str, Dict[str, object]] = {}
    for node_id, dense, sparse in results:
        out[node_id] = {
            "dense": dense,
            "sparse": sparse,
        }
    return out
