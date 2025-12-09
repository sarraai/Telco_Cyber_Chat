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

# ========= INDEX MANAGEMENT =========
_index_checked = {}

def ensure_url_index(collection_name: Optional[str] = None) -> bool:
    """
    Ensure that a keyword index exists on the 'url' field in Qdrant.
    This is required for filtering by URL in queries.
    Returns True if index exists or was created successfully.
    """
    collection = collection_name or QDRANT_COLLECTION
    if not collection:
        raise RuntimeError("QDRANT_COLLECTION is not set in environment variables")
    
    # Check if we've already verified this collection
    if _index_checked.get(collection):
        return True
    
    client = get_qdrant_client()
    
    try:
        print(f"[INFO] Checking if 'url' index exists in collection '{collection}'...")
        
        # Try to get collection info to see existing indexes
        collection_info = client.get_collection(collection_name=collection)
        
        # Check if 'url' field already has an index
        payload_schema = collection_info.config.params.payload_schema or {}
        if "url" in payload_schema:
            print(f"[INFO] Index on 'url' field already exists in collection '{collection}'")
            _index_checked[collection] = True
            return True
        
        # Create the index
        print(f"[INFO] Creating keyword index on 'url' field in collection '{collection}'...")
        client.create_payload_index(
            collection_name=collection,
            field_name="url",
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
        print(f"[INFO] Successfully created keyword index on 'url' field in collection '{collection}'")
        _index_checked[collection] = True
        return True
        
    except Exception as e:
        error_msg = str(e).lower()
        
        # If index already exists, that's fine
        if "already" in error_msg or "exist" in error_msg:
            print(f"[INFO] Index on 'url' field already exists in collection '{collection}' (caught exception)")
            _index_checked[collection] = True
            return True
        
        # If collection doesn't exist, that's a bigger problem
        if "not found" in error_msg or "does not exist" in error_msg:
            print(f"[ERROR] Collection '{collection}' does not exist in Qdrant!")
            return False
        
        # Other errors
        print(f"[ERROR] Failed to create/verify index on 'url' field: {e}")
        return False

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
    
    # Ensure the URL index exists before querying
    index_ok = ensure_url_index(collection)
    if not index_ok:
        print(f"[WARN] Could not verify/create index for 'url' field, proceeding without deduplication check for {norm_url}")
        return False
    
    # If some old code passes unexpected kwargs, just ignore them quietly.
    if kwargs:
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
        # Use count_filter parameter
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
