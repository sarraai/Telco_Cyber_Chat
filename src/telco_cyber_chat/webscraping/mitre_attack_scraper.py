from __future__ import annotations

import json
import logging
import hashlib
from typing import List, Dict, Any, Optional
from datetime import datetime, timezone

import requests
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)

# -------------------------------
# Constants
# -------------------------------

VENDOR_VALUE = "mitre"
MITRE_MOBILE_JSON_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/mobile-attack/mobile-attack.json"
)

# Which STIX object types we keep
ALLOWED_TYPES = {
    "attack-pattern",
    "malware",
    "intrusion-set",
    "tool",
    "course-of-action",
    "relationship",  # ✅ Now included
}


# -------------------------------
# Hash Function (same as ingestion)
# -------------------------------

def mitre_id_to_int(mitre_id: str) -> int:
    """Convert STIX ID to consistent integer hash."""
    hash_value = hashlib.sha256(mitre_id.encode()).digest()
    return int.from_bytes(hash_value[:8], byteorder="big")


# -------------------------------
# Qdrant Check Functions
# -------------------------------

def stix_id_already_ingested(
    client: QdrantClient,
    collection_name: str,
    stix_id: str,
) -> bool:
    """
    Check if a STIX ID already exists in Qdrant for the MITRE vendor.
    Uses the new filter structure: vendor + stix_id
    """
    try:
        results = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="vendor",
                        match=MatchValue(value=VENDOR_VALUE),
                    ),
                    FieldCondition(
                        key="stix_id",
                        match=MatchValue(value=stix_id),
                    ),
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(results[0]) > 0
    except Exception as e:
        logger.error(f"[MITRE] Qdrant check failed for stix_id={stix_id}: {e}")
        return False


# -------------------------------
# Fetch bundle from GitHub
# -------------------------------

def fetch_mobile_attack_bundle(timeout: int = 30) -> Dict[str, Any]:
    """
    Download the mobile-attack STIX bundle JSON directly from GitHub.
    """
    resp = requests.get(MITRE_MOBILE_JSON_URL, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError(f"[MITRE] Unexpected JSON root type: {type(data)}")
    return data


# -------------------------------
# Helper to extract URL from external_references
# -------------------------------

def extract_primary_url(obj: Dict[str, Any]) -> Optional[str]:
    """
    Extract the primary URL from external_references.
    Returns None if no URL found.
    """
    refs = obj.get("external_references", []) or []
    for ref in refs:
        src = (ref.get("source_name") or "").lower()
        if "mitre-attack" in src or "mitre-mobile-attack" in src:
            if ref.get("url"):
                return ref["url"]
    for ref in refs:
        if ref.get("url"):
            return ref["url"]
    return None


# -------------------------------
# Convert STIX object to document structure
# -------------------------------

def stix_obj_to_doc(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """
    Convert a single STIX object to a document structure.
    Returns None if the object is not relevant.
    
    Structure matches your ingestion format:
    - For non-relationships: type, name, description, stix_id, url (optional)
    - For relationships: type, relationship_type, source_ref, target_ref, description, stix_id, url (optional)
    """
    stix_type = obj.get("type")
    stix_id = obj.get("id")
    
    if not stix_id or stix_type not in ALLOWED_TYPES:
        return None

    url = extract_primary_url(obj)
    
    if stix_type == "relationship":
        return {
            "stix_id": stix_id,
            "type": stix_type,
            "relationship_type": obj.get("relationship_type"),
            "source_ref": obj.get("source_ref", ""),
            "target_ref": obj.get("target_ref", ""),
            "description": obj.get("description", ""),
            "url": url,
        }
    else:
        # Check if has meaningful content
        description = (obj.get("description") or "").strip()
        if not description:
            return None
            
        return {
            "stix_id": stix_id,
            "type": stix_type,
            "name": obj.get("name", ""),
            "description": description,
            "url": url,
        }


# -------------------------------
# Main scraper with Qdrant dedupe
# -------------------------------

def scrape_mitre_mobile(
    client: QdrantClient,
    collection_name: str = "Telco_CyberChat",
    check_qdrant: bool = True,
) -> List[Dict[str, Any]]:
    """
    High-level function:

      - Downloads mobile-attack JSON from GitHub
      - Converts relevant STIX objects into document structures
      - Optional: skips STIX IDs already ingested in Qdrant (by vendor + stix_id filter)
      - Returns list of new documents to be ingested
      - Logs: Existing count and New count only
    """
    try:
        bundle = fetch_mobile_attack_bundle()
    except Exception as e:
        logger.error("[MITRE] Failed to download mobile-attack bundle: %s", e)
        return []

    objects = bundle.get("objects") or []
    if not isinstance(objects, list):
        logger.error("[MITRE] Unexpected 'objects' type: %s", type(objects))
        return []

    docs: List[Dict[str, Any]] = []

    seen_stix_ids: set[str] = set()
    n_existing = 0
    n_new = 0

    for obj in objects:
        if not isinstance(obj, dict):
            continue

        doc = stix_obj_to_doc(obj)
        if doc is None:
            continue

        stix_id = doc["stix_id"]

        # De-duplicate within this run
        if stix_id in seen_stix_ids:
            continue
        seen_stix_ids.add(stix_id)

        # Qdrant dedupe BEFORE indexing (using vendor + stix_id filter)
        if check_qdrant:
            try:
                if stix_id_already_ingested(client, collection_name, stix_id):
                    n_existing += 1
                    continue
            except Exception as e:
                logger.error("[MITRE] Qdrant check failed for %s: %s", stix_id, e)
                # Conservative: continue scraping anyway

        # New doc for this run
        n_new += 1
        docs.append(doc)

    logger.info(
        "[MITRE] Existing: %d | New: %d",
        n_existing, n_new
    )
    return docs


# -------------------------------
# CLI debug (optional)
# -------------------------------

if __name__ == "__main__":
    import os
    logging.basicConfig(level=logging.INFO)
    
    # For testing without Qdrant
    docs = scrape_mitre_mobile(client=None, check_qdrant=False)
    print(f"Total docs: {len(docs)}")
    if docs:
        print("\nExample non-relationship doc:")
        import pprint
        non_rel = next((d for d in docs if d["type"] != "relationship"), None)
        if non_rel:
            pprint.pprint(non_rel)
        
        print("\nExample relationship doc:")
        rel = next((d for d in docs if d["type"] == "relationship"), None)
        if rel:
            pprint.pprint(rel)
    else:
        print("No documents extracted.")
