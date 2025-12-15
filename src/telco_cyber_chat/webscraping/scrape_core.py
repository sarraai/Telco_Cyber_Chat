"""
scrape_core.py

Core helpers shared by all web scrapers.

Responsibilities:
- Qdrant client factory (cached)
- URL normalization (stable canonical form)
- Ensure payload indexes exist for dedupe filters (vendor + url)
- url_already_ingested(): check if a document is already in Qdrant
"""

from __future__ import annotations

import os
import logging
from functools import lru_cache
from typing import Optional, Dict, Any, Iterable
from urllib.parse import urlparse, urlunparse

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

logger = logging.getLogger(__name__)

# ========= ENV CONFIG =========
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "Telco_CyberChat").strip()

# Fields used for dedupe checks
DEFAULT_INDEX_FIELDS = ("vendor", "url")


# ========= NORMALIZERS =========
def normalize_url(url: str) -> str:
    """
    Normalize URLs so that small differences (trailing slash, casing, fragments)
    don't break deduplication.
    """
    url = (url or "").strip()
    if not url:
        return ""

    parsed = urlparse(url)

    scheme = (parsed.scheme or "https").lower()
    netloc = (parsed.netloc or "").lower()

    path = parsed.path or ""
    if path.endswith("/"):
        path = path.rstrip("/")

    # Drop fragment; keep query (sometimes part of canonical URLs)
    return urlunparse((scheme, netloc, path, parsed.params, parsed.query, ""))


def normalize_vendor(vendor: str) -> str:
    return (vendor or "").strip().lower()


# Handy alias you can reuse in node_builder / qdrant_ingest too
def canonical_url(url: str) -> str:
    return normalize_url(url)


# ========= INDEX ENSURER =========
def ensure_keyword_index(
    client: QdrantClient,
    collection_name: str,
    field_name: str,
) -> None:
    """Ensure `field_name` has a KEYWORD payload index. Safe to call multiple times."""
    if not collection_name:
        return
    try:
        client.create_payload_index(
            collection_name=collection_name,
            field_name=field_name,
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
        logger.info("[QDRANT] Created payload index for '%s' (keyword).", field_name)
    except Exception as e:
        logger.debug("[QDRANT] Index '%s' may already exist or cannot be created: %s", field_name, e)


def ensure_payload_indexes(
    client: QdrantClient,
    collection_name: str,
    fields: Iterable[str] = DEFAULT_INDEX_FIELDS,
) -> None:
    # Only try to create indexes if collection exists
    try:
        client.get_collection(collection_name)
    except Exception:
        logger.debug("[QDRANT] Collection '%s' not found yet; skipping index creation.", collection_name)
        return

    for f in fields:
        ensure_keyword_index(client, collection_name, f)


# ========= CLIENT =========
@lru_cache(maxsize=1)
def get_qdrant_client() -> QdrantClient:
    """
    Lazily create a single QdrantClient instance and cache it.
    Also ensures payload indexes used by scrapers exist.
    """
    if not QDRANT_URL:
        raise RuntimeError("QDRANT_URL is not set in environment variables")

    client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)

    # Ensure the indexes your scrapers depend on (vendor + url)
    try:
        ensure_payload_indexes(client, QDRANT_COLLECTION, DEFAULT_INDEX_FIELDS)
    except Exception as e:
        logger.warning("[QDRANT] Warning: ensure_payload_indexes failed: %s", e)

    return client


# ========= DEDUPE CHECK =========
def url_already_ingested(
    url: str,
    collection_name: Optional[str] = None,
    *,
    vendor: Optional[str] = None,
    filter: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> bool:
    """
    Return True if a document with this (vendor + url) is already present in Qdrant.

    Recommended usage:
        url_already_ingested(url, vendor="huawei")
        url_already_ingested(url, vendor="variot")

    Notes:
    - Assumes Qdrant payload stores:
        payload["url"]    = normalize_url(url)
        payload["vendor"] = normalize_vendor(vendor)
    """
    _ = kwargs  # ignore legacy extras

    collection = collection_name or QDRANT_COLLECTION
    if not collection:
        raise RuntimeError("QDRANT_COLLECTION is not set in environment variables")

    norm_url = normalize_url(url)
    if not norm_url:
        return False

    client = get_qdrant_client()

    must_conditions = [
        qmodels.FieldCondition(
            key="url",
            match=qmodels.MatchValue(value=norm_url),
        )
    ]

    if vendor:
        must_conditions.append(
            qmodels.FieldCondition(
                key="vendor",
                match=qmodels.MatchValue(value=normalize_vendor(vendor)),
            )
        )

    if filter:
        for key, value in filter.items():
            if value is None:
                continue
            must_conditions.append(
                qmodels.FieldCondition(
                    key=key,
                    match=qmodels.MatchValue(value=value),
                )
            )

    flt = qmodels.Filter(must=must_conditions)

    try:
        res = client.count(
            collection_name=collection,
            count_filter=flt,
            exact=False,
        )
        return (res.count or 0) > 0
    except Exception as e:
        # If Qdrant is down / collection missing, do NOT skip scraping
        logger.warning("[WARN] url_already_ingested() failed for %s: %s", norm_url, e)
        return False
