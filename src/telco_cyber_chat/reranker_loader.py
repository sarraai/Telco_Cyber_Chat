import os
import aiohttp
import asyncio
from typing import List, Optional
import logging

# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
BGE_RERANK_URL = os.getenv("BGE_RERANK_URL", "").strip()
BGE_RERANK_TIMEOUT = int(os.getenv("BGE_RERANK_TIMEOUT", "30"))

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Async core reranker function
# -----------------------------------------------------------------------------
async def get_rerank_scores(
    query: str,
    texts: List[str],
    timeout: Optional[int] = None,
) -> Optional[List[float]]:
    """
    Call the remote BGE reranker service (LangServe) asynchronously.

    Args:
        query: User's input query text
        texts: List of document chunks to score against the query
        timeout: Optional override for total request timeout (seconds)

    Returns:
        List of float scores (same length as `texts`) or:
          - [] if texts is empty
          - None if the request failed or response is invalid
    """
    if not query or not query.strip():
        logger.warning("Empty query provided to reranker")
        return None

    if not texts:
        logger.warning("Empty 'texts' list provided to reranker")
        return []

    if not BGE_RERANK_URL:
        raise ValueError("BGE_RERANK_URL must be set in environment variables")

    payload = {
        "input": {
            "query": query,
            "texts": texts,
        }
    }

    req_timeout = aiohttp.ClientTimeout(total=timeout or BGE_RERANK_TIMEOUT)

    try:
        async with aiohttp.ClientSession(timeout=req_timeout) as session:
            async with session.post(
                BGE_RERANK_URL,
                json=payload,
                headers={"Content-Type": "application/json"},
            ) as response:
                response.raise_for_status()
                data = await response.json()
    except asyncio.TimeoutError:
        logger.error(f"Reranker request timed out after {timeout or BGE_RERANK_TIMEOUT}s")
        return None
    except aiohttp.ClientError as e:
        logger.error(f"Reranker HTTP request failed: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error during rerank: {e}")
        return None

    # Expecting LangServe style: { "output": {"scores": [...]} , "metadata": {...} }
    if not isinstance(data, dict) or "output" not in data:
        logger.error(f"Unexpected reranker response format: {data}")
        return None

    output = data["output"]
    scores = output.get("scores")

    if not isinstance(scores, list):
        logger.error(f"'scores' missing or not a list in reranker response: {output}")
        return None

    try:
        return [float(s) for s in scores]
    except Exception as e:
        logger.error(f"Failed to convert reranker scores to float: {e}")
        return None
