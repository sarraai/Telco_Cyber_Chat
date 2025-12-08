"""
Core helpers shared by all web scrapers.

- Qdrant client factory
- URL normalization
- url_already_ingested(): check if a URL is already in the vector DB
"""

import os
from functools import lru_cache
from typing import Optional
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
) -> bool:
    """
    Return True if a document with this URL is already present in Qdrant.

    Assumes that your RAG documents store the URL in the payload under the key "url",
    and that the same normalization logic is used before upserting.
    """
    collection = collection_name or QDRANT_COLLECTION
    if not collection:
        raise RuntimeError("QDRANT_COLLECTION is not set in environment variables")

    norm_url = normalize_url(url)
    if not norm_url:
        return False

    client = get_qdrant_client()

    flt = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="url",
                match=qmodels.MatchValue(value=norm_url),
            )
        ]
    )

    try:
        # exact=False is fine here; we only care if count > 0
        res = client.count(
            collection_name=collection,
            filter=flt,
            exact=False,
        )
        return (res.count or 0) > 0
    except Exception as e:
        # Be conservative: if Qdrant is down, don't skip scraping – return False
        print(f"[WARN] url_already_ingested() failed for {norm_url}: {e}")
        return False
