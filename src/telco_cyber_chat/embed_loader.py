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
BGE_EMBEDDING_TIMEOUT = int(os.getenv("BGE_EMBEDDING_TIMEOUT", "60"))  # CHANGED: 30 → 60 for Colab

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# NEW: Helper to build correct endpoint URL
# -----------------------------------------------------------------------------
def _get_invoke_url() -> str:
    """Ensure URL ends with /invoke for LangServe compatibility."""
    if not BGE_EMBEDDING_URL:
        raise ValueError("BGE_EMBEDDING_URL must be set in environment variables")
    
    url = BGE_EMBEDDING_URL.rstrip('/')
    
    # If URL doesn't end with /invoke, append it
    if not url.endswith('/invoke'):
        url = f"{url}/invoke"
    
    return url


# -----------------------------------------------------------------------------
# Async Core Embedding Function
# -----------------------------------------------------------------------------
async def get_query_embeddings(
    query: str,
    return_dense: bool = True,
    return_sparse: bool = True,
    max_length: int = 4096,      # CHANGED: 8192 → 4096 for Colab stability
    retry_count: int = 3,        # NEW: Add retry logic
) -> Tuple[Optional[np.ndarray], Optional[Dict[int, float]]]:
    """
    Get BGE-M3 embeddings for a user query from the remote service (async).
    
    Args:
        query: User's input query text
        return_dense: Whether to return dense embeddings (1024-dim vector)
        return_sparse: Whether to return sparse embeddings (lexical weights)
        max_length: Maximum token length for BGE-M3
        retry_count: Number of retries on failure (NEW)
        
    Returns:
        Tuple of (dense_vector, sparse_weights):
            - dense_vector: np.ndarray of shape (1024,) or None
            - sparse_weights: Dict[int, float] or None
    """
    if not query or not query.strip():
        logger.warning("Empty query provided")
        return None, None
    
    # CHANGED: Use helper function to get correct URL
    try:
        invoke_url = _get_invoke_url()
    except ValueError as e:
        logger.error(str(e))
        return None, None
    
    payload = {
        "input": {
            "text": query,
            "return_dense": return_dense,
            "return_sparse": return_sparse,
            "max_length": max_length
        }
    }
    
    timeout = aiohttp.ClientTimeout(total=BGE_EMBEDDING_TIMEOUT)
    
    # NEW: Retry logic with exponential backoff
    for attempt in range(retry_count):
        try:
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.post(
                    invoke_url,  # CHANGED: Use invoke_url instead of BGE_EMBEDDING_URL
                    json=payload,
                    headers={"Content-Type": "application/json"}
                ) as response:
                    
                    # NEW: Handle specific HTTP errors
                    if response.status == 404:
                        logger.error(
                            f"❌ 404 Not Found at {invoke_url}\n"
                            f"   Check that:\n"
                            f"   1. Your Colab server is running\n"
                            f"   2. Ngrok tunnel is active\n"
                            f"   3. URL is correct: {BGE_EMBEDDING_URL}"
                        )
                        return None, None
                    
                    # NEW: Handle server overload
                    if response.status == 503:
                        logger.warning(f"Server overloaded (503), retrying... (attempt {attempt + 1}/{retry_count})")
                        await asyncio.sleep(2 ** attempt)  # Exponential backoff: 1s, 2s, 4s
                        continue
                    
                    response.raise_for_status()
                    data = await response.json()
            
            # Parse response (unchanged logic)
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
            logger.error(f"Request timed out after {BGE_EMBEDDING_TIMEOUT}s (attempt {attempt + 1}/{retry_count})")
            if attempt < retry_count - 1:
                await asyncio.sleep(2)  # Wait 2s before retry
            
        except aiohttp.ClientError as e:
            logger.error(f"HTTP request failed: {e} (attempt {attempt + 1}/{retry_count})")
            if attempt < retry_count - 1:
                await asyncio.sleep(2)
            
        except Exception as e:
            logger.error(f"Unexpected error during embedding: {e} (attempt {attempt + 1}/{retry_count})")
            if attempt < retry_count - 1:
                await asyncio.sleep(2)
    
    # NEW: All retries failed
    logger.error(f"❌ All {retry_count} attempts failed for query: {query[:50]}...")
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


# -----------------------------------------------------------------------------
# NEW: Health check utility
# -----------------------------------------------------------------------------
async def check_embedding_service() -> bool:
    """
    Check if the embedding service is responding.
    Useful for debugging connection issues.
    """
    try:
        dense, sparse = await get_query_embeddings(
            "test query",
            return_dense=True,
            return_sparse=True,
            retry_count=1
        )
        
        if dense is not None and sparse is not None:
            logger.info(f"✅ Embedding service is healthy (dense shape: {dense.shape}, sparse keys: {len(sparse)})")
            return True
        else:
            logger.error("❌ Embedding service returned None")
            return False
            
    except Exception as e:
        logger.error(f"❌ Embedding service health check failed: {e}")
        return False
