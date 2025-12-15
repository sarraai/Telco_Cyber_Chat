from __future__ import annotations

import os
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

try:
    from qdrant_client.models import PayloadSelectorInclude
except Exception:
    PayloadSelectorInclude = None  # type: ignore

logger = logging.getLogger(__name__)

VENDOR_VALUE = "mitre"
MITRE_MOBILE_JSON_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/mobile-attack/mobile-attack.json"
)

QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "Telco_CyberChat")


# -------------------------------
# 1) FILTER FIRST: get existing stix_id for vendor=mitre
# -------------------------------
def fetch_existing_mitre_stix_ids(
    client: QdrantClient,
    collection_name: str,
    page_size: int = 256,
) -> Set[str]:
    existing: Set[str] = set()
    offset = None

    vendor_filter = Filter(
        must=[FieldCondition(key="vendor", match=MatchValue(value=VENDOR_VALUE))]
    )

    while True:
        with_payload = True
        if PayloadSelectorInclude is not None:
            with_payload = PayloadSelectorInclude(include=["stix_id"])

        points, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=vendor_filter,
            limit=page_size,
            offset=offset,
            with_payload=with_payload,
            with_vectors=False,
        )

        for p in points:
            payload = p.payload or {}
            sid = payload.get("stix_id")
            if isinstance(sid, str) and sid.strip():
                existing.add(sid.strip())

        if offset is None:
            break

    return existing


# -------------------------------
# 2) SCRAPE: download bundle JSON
# -------------------------------
def fetch_mobile_attack_bundle(timeout: int = 60) -> Dict[str, Any]:
    resp = requests.get(MITRE_MOBILE_JSON_URL, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError(f"[MITRE] Unexpected JSON root type: {type(data)}")
    return data


# -------------------------------
# URL extraction (stored, not a filter)
# -------------------------------
def extract_primary_url(obj: Dict[str, Any]) -> Optional[str]:
    refs = obj.get("external_references", []) or []
    if not isinstance(refs, list):
        return None

    for ref in refs:
        if not isinstance(ref, dict):
            continue
        src = (ref.get("source_name") or "").lower()
        url = ref.get("url")
        if ("mitre-attack" in src or "mitre-mobile-attack" in src) and isinstance(url, str):
            url = url.strip()
            if url:
                return url

    for ref in refs:
        if not isinstance(ref, dict):
            continue
        url = ref.get("url")
        if isinstance(url, str):
            url = url.strip()
            if url:
                return url

    return None


# -------------------------------
# Output docs: (relationship vs non-relationship)
# -------------------------------
def stix_obj_to_doc(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    stix_id = obj.get("id")
    stix_type = obj.get("type")

    if not isinstance(stix_id, str) or not stix_id.strip():
        return None
    if not isinstance(stix_type, str) or not stix_type.strip():
        return None

    scraped_date = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    url = extract_primary_url(obj)

    if stix_type == "relationship":
        return {
            "vendor": VENDOR_VALUE,
            "stix_id": stix_id.strip(),
            "type": "relationship",
            "scraped_date": scraped_date,
            "relationship_type": obj.get("relationship_type"),
            "source_ref": obj.get("source_ref"),
            "target_ref": obj.get("target_ref"),
            "description": (obj.get("description") or "").strip(),
            "url": url,
        }

    return {
        "vendor": VENDOR_VALUE,
        "stix_id": stix_id.strip(),
        "type": stix_type.strip(),
        "scraped_date": scraped_date,
        "name": (obj.get("name") or "").strip(),
        "description": (obj.get("description") or "").strip(),
        "url": url,
    }


# -------------------------------
# MAIN (your current function)
# -------------------------------
def scrape_mitre_mobile_filter_first_precheck(
    client: QdrantClient,
    collection_name: str = "Telco_CyberChat",
) -> List[Dict[str, Any]]:
    existing_stix_ids = fetch_existing_mitre_stix_ids(client, collection_name)
    logger.info("[MITRE] Existing stix_id loaded from Qdrant: %d", len(existing_stix_ids))

    bundle = fetch_mobile_attack_bundle()
    objects = bundle.get("objects") or []
    if not isinstance(objects, list):
        logger.error("[MITRE] Unexpected 'objects' type: %s", type(objects))
        return []

    docs: List[Dict[str, Any]] = []
    seen_stix_ids: Set[str] = set()

    n_existing = 0
    n_new = 0
    n_skipped_invalid = 0

    for obj in objects:
        if not isinstance(obj, dict):
            n_skipped_invalid += 1
            continue

        stix_id = obj.get("id")
        if not isinstance(stix_id, str) or not stix_id.strip():
            n_skipped_invalid += 1
            continue
        stix_id = stix_id.strip()

        if stix_id in seen_stix_ids:
            continue
        seen_stix_ids.add(stix_id)

        if stix_id in existing_stix_ids:
            n_existing += 1
            continue

        doc = stix_obj_to_doc(obj)
        if doc is None:
            n_skipped_invalid += 1
            continue

        docs.append(doc)
        n_new += 1

    logger.info(
        "[MITRE] Existing: %d | New: %d | Skipped invalid: %d | Unique in bundle: %d",
        n_existing, n_new, n_skipped_invalid, len(seen_stix_ids),
    )
    return docs


# -------------------------------
# ✅ WRAPPER EXPECTED BY YOUR PIPELINE/GRAPH
# -------------------------------
def scrape_mitre_mobile(check_qdrant: bool = True) -> List[Dict[str, Any]]:
    """
    Public scraper entrypoint expected by:
      - webscraping/__init__.py
      - ingest_pipeline.py
      - scraper_graph.py

    Returns: List[docs] where each doc includes:
      - vendor, stix_id, type, scraped_date, ... and url
    """
    # local import to avoid circular imports at package import time
    from telco_cyber_chat.webscraping.scrape_core import get_qdrant_client

    if check_qdrant:
        client = get_qdrant_client()
        return scrape_mitre_mobile_filter_first_precheck(client, collection_name=QDRANT_COLLECTION)

    # no precheck: return everything in the bundle
    bundle = fetch_mobile_attack_bundle()
    objects = bundle.get("objects") or []
    if not isinstance(objects, list):
        return []

    docs: List[Dict[str, Any]] = []
    seen: Set[str] = set()

    for obj in objects:
        if not isinstance(obj, dict):
            continue
        sid = obj.get("id")
        if not isinstance(sid, str) or not sid.strip():
            continue
        sid = sid.strip()
        if sid in seen:
            continue
        seen.add(sid)

        doc = stix_obj_to_doc(obj)
        if doc:
            docs.append(doc)

    return docs
