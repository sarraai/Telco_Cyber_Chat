"""
Core helpers shared by all web scrapers.
- Qdrant client factory
- URL normalization
- url_already_ingested(): check if a URL is already in the vector DB
"""
import os
from functools import lru_cache
from typing import Optional, Dict, Any
from urllib.parse import urlparse, urlunparse
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

# ========= ENV CONFIG =========
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "Telco_CyberChat")

# ========= CLIENT =========
@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """
    Lazily create a single QdrantClient instance and cache it.
    """
    if not QDRANT_URL:
        raise RuntimeError("QDRANT_URL is not set in environment variables")
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY,
    )

# ========= URL NORMALIZATION =========
def normalize_url(url: str) -> str:
    """
    Normalize URLs so that small differences (trailing slash, casing, fragments)
    don't break deduplication.
    """
    url = (url or "").strip()
    if not url:
        return ""
    parsed = urlparse(url)
    # Lowercase scheme + host; keep path as-is
    scheme = parsed.scheme.lower()
    netloc = parsed.netloc.lower()
    # Drop fragment; keep query (usually part of canonical advisory URLs)
    path = parsed.path or ""
    if path.endswith("/"):
        path = path.rstrip("/")
    normalized = urlunparse((scheme, netloc, path, parsed.params, parsed.query, ""))
    return normalized

# ========= DEDUPE CHECK =========
def url_already_ingested(
    url: str,
    collection_name: Optional[str] = None,
    *,  # Force all following parameters to be keyword-only
    filter: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> bool:
    """
    Return True if a document with this URL is already present in Qdrant.
    - Assumes that your RAG documents store the URL in the payload under
      the key "url", and that the same normalization logic is used before
      upserting.
    - Accepts an optional `filter` dict (e.g. {"source": "cisco"}) which
      is AND-ed with the URL condition. Must be passed as keyword argument.
    - Accepts extra **kwargs for backwards compatibility; they are ignored.
    
    Usage:
        url_already_ingested(url)  # Simple check
        url_already_ingested(url, filter={"source": "cisco"})  # With filter
        url_already_ingested(url, collection_name="custom", filter={"source": "cisco"})
    """
    collection = collection_name or QDRANT_COLLECTION
    if not collection:
        raise RuntimeError("QDRANT_COLLECTION is not set in environment variables")
    
    norm_url = normalize_url(url)
    if not norm_url:
        return False
    
    # If some old code passes unexpected kwargs, just ignore them quietly.
    if kwargs:
        # You can switch this to a logger.debug if you add logging.
        pass
    
    client = get_qdrant_client()
    
    # Base condition: URL must match
    must_conditions = [
        qmodels.FieldCondition(
            key="url",
            match=qmodels.MatchValue(value=norm_url),
        )
    ]
    
    # Optionally AND extra field filters (e.g. source="cisco")
    if filter:
        for key, value in filter.items():
            must_conditions.append(
                qmodels.FieldCondition(
                    key=key,
                    match=qmodels.MatchValue(value=value),
                )
            )
    
    flt = qmodels.Filter(must=must_conditions)
    
    try:
        # exact=False is fine here; we only care if count > 0
        res = client.count(
            collection_name=collection,
            count_filter=flt,
            exact=False,
        )
        return (res.count or 0) > 0
    except Exception as e:
        # Be conservative: if Qdrant is down, don't skip scraping – return False
        print(f"[WARN] url_already_ingested() failed for {norm_url}: {e}")
        return False
