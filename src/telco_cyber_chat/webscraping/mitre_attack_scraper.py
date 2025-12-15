from __future__ import annotations

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
# NORMALIZE only when needed (new objects only)
# Output matches your two schemas
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
            "stix_id": stix_id,
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
        "stix_id": stix_id,
        "type": stix_type,
        "scraped_date": scraped_date,
        "name": (obj.get("name") or "").strip(),
        "description": (obj.get("description") or "").strip(),
        "url": url,
    }


# -------------------------------
# MAIN: filter first, then decide by stix_id BEFORE normalize
# -------------------------------
def scrape_mitre_mobile_filter_first_precheck(
    client: QdrantClient,
    collection_name: str = "Telco_CyberChat",
) -> List[Dict[str, Any]]:
    # 1) FILTER FIRST (vendor=mitre) -> existing stix_ids
    existing_stix_ids = fetch_existing_mitre_stix_ids(client, collection_name)
    logger.info("[MITRE] Existing stix_id loaded from Qdrant: %d", len(existing_stix_ids))

    # 2) SCRAPE (download JSON)
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

        # ✅ Decide new/existing FIRST using only stix_id (fast path)
        stix_id = obj.get("id")
        if not isinstance(stix_id, str) or not stix_id.strip():
            n_skipped_invalid += 1
            continue
        stix_id = stix_id.strip()

        # in-run dedupe
        if stix_id in seen_stix_ids:
            continue
        seen_stix_ids.add(stix_id)

        # Qdrant check is just set membership now (vendor+stix_id uniqueness)
        if stix_id in existing_stix_ids:
            n_existing += 1
            continue

        # ✅ Only now we normalize (extract url/name/description/relationship fields)
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
