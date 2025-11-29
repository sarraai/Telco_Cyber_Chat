import os
import aiohttp
import asyncio
import numpy as np
from typing import Tuple, Optional, Dict, Any
import logging

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BGE_EMBEDDING_URL = os.getenv("BGE_EMBEDDING_URL", "").strip()

BGE_EMBEDDING_TIMEOUT = int(os.getenv("BGE_EMBEDDING_TIMEOUT", "30"))

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Async Core Embedding Function
# -----------------------------------------------------------------------------
async def get_query_embeddings(
    query: str,
    return_dense: bool = True,
    return_sparse: bool = True,
    max_length: int = 8192
) -> Tuple[Optional[np.ndarray], Optional[Dict[int, float]]]:
    """
    Get BGE-M3 embeddings for a user query from the remote service (async).
    
    Args:
        query: User's input query text
        return_dense: Whether to return dense embeddings (1024-dim vector)
        return_sparse: Whether to return sparse embeddings (lexical weights)
        max_length: Maximum token length for BGE-M3
        
    Returns:
        Tuple of (dense_vector, sparse_weights):
            - dense_vector: np.ndarray of shape (1024,) or None
            - sparse_weights: Dict[int, float] or None
    """
    if not query or not query.strip():
        logger.warning("Empty query provided")
        return None, None
    
    if not BGE_EMBEDDING_URL:
        raise ValueError("BGE_EMBEDDING_URL must be set in environment variables")
    
    payload = {
        "input": {
            "text": query,
            "return_dense": return_dense,
            "return_sparse": return_sparse,
            "max_length": max_length
        }
    }
    
    timeout = aiohttp.ClientTimeout(total=BGE_EMBEDDING_TIMEOUT)
    
    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(
                BGE_EMBEDDING_URL,
                json=payload,
                headers={"Content-Type": "application/json"}
            ) as response:
                response.raise_for_status()
                data = await response.json()
        
        if not isinstance(data, dict) or "output" not in data:
            logger.error(f"Unexpected response format: {data}")
            return None, None
        
        output = data["output"]
        
        # Extract dense embeddings
        dense_vector = None
        if return_dense:
            dense_vecs = output.get("dense_vecs")
            if dense_vecs and isinstance(dense_vecs, list):
                dense_vector = np.array(dense_vecs, dtype=np.float32)
            else:
                logger.warning("No dense embeddings in response")
        
        # Extract sparse embeddings
        sparse_weights = None
        if return_sparse:
            lexical_weights = output.get("lexical_weights")
            if lexical_weights and isinstance(lexical_weights, dict):
                sparse_weights = {int(k): float(v) for k, v in lexical_weights.items()}
            else:
                logger.warning("No sparse embeddings in response")
        
        return dense_vector, sparse_weights
        
    except asyncio.TimeoutError:
        logger.error(f"Request timed out after {BGE_EMBEDDING_TIMEOUT}s")
        return None, None
    except aiohttp.ClientError as e:
        logger.error(f"HTTP request failed: {e}")
        return None, None
    except Exception as e:
        logger.error(f"Unexpected error during embedding: {e}")
        return None, None


# -----------------------------------------------------------------------------
# Async Qdrant-Compatible Helper Functions
# -----------------------------------------------------------------------------
async def get_dense_embedding(query: str) -> Optional[np.ndarray]:
    """Get only dense embedding (for Qdrant dense vector search)."""
    dense, _ = await get_query_embeddings(query, return_dense=True, return_sparse=False)
    return dense


async def get_sparse_embedding(query: str) -> Optional[Dict[int, float]]:
    """Get only sparse embedding (for Qdrant sparse vector search)."""
    _, sparse = await get_query_embeddings(query, return_dense=False, return_sparse=True)
    return sparse


async def get_hybrid_embeddings(query: str) -> Tuple[Optional[np.ndarray], Optional[Dict[int, float]]]:
    """Get both dense and sparse embeddings (for Qdrant hybrid search)."""
    return await get_query_embeddings(query, return_dense=True, return_sparse=True)
