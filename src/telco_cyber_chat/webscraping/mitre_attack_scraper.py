from __future__ import annotations

import os
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Set, Tuple

import requests
from llama_index.core.schema import TextNode
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

try:
    from qdrant_client.models import PayloadSelectorInclude
except Exception:
    PayloadSelectorInclude = None  # type: ignore


logger = logging.getLogger(__name__)

# ================== CONFIG ==================
VENDOR_VALUE = "mitre"
MITRE_MOBILE_JSON_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/mobile-attack/mobile-attack.json"
)

QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "Telco_CyberChat")

# Payload keys expected in Qdrant (for precheck)
VENDOR_KEY = "vendor"
STIX_ID_KEY = "stix_id"
URL_KEY = "url"


# -------------------------------
# QDRANT PRECHECK (filter-first)
# -------------------------------
def fetch_existing_mitre_stix_ids(
    client: QdrantClient,
    collection_name: str,
    page_size: int = 256,
) -> Set[str]:
    """
    Scroll Qdrant with filter vendor=mitre and collect payload['stix_id'].
    This avoids per-object Qdrant calls.
    """
    existing: Set[str] = set()
    offset = None

    vendor_filter = Filter(
        must=[FieldCondition(key=VENDOR_KEY, match=MatchValue(value=VENDOR_VALUE))]
    )

    while True:
        with_payload: Any = True
        if PayloadSelectorInclude is not None:
            with_payload = PayloadSelectorInclude(include=[STIX_ID_KEY])

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
            sid = payload.get(STIX_ID_KEY)
            if isinstance(sid, str) and sid.strip():
                existing.add(sid.strip())

        if offset is None:
            break

    return existing


# -------------------------------
# SCRAPE: download bundle JSON
# -------------------------------
def fetch_mobile_attack_bundle(timeout: int = 60) -> Dict[str, Any]:
    resp = requests.get(MITRE_MOBILE_JSON_URL, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError(f"[MITRE] Unexpected JSON root type: {type(data)}")
    return data


# -------------------------------
# Helpers: URL + external_id
# -------------------------------
def extract_primary_url(obj: Dict[str, Any]) -> Optional[str]:
    """
    Best effort to get a canonical MITRE URL from external_references.
    """
    refs = obj.get("external_references", []) or []
    if not isinstance(refs, list):
        return None

    # Prefer MITRE ATT&CK reference
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        src = (ref.get("source_name") or "").lower()
        url = ref.get("url")
        if ("mitre-attack" in src or "mitre-mobile-attack" in src) and isinstance(url, str):
            url = url.strip()
            if url:
                return url

    # Fallback: first URL anywhere
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        url = ref.get("url")
        if isinstance(url, str):
            url = url.strip()
            if url:
                return url

    return None


def extract_external_id(obj: Dict[str, Any]) -> Optional[str]:
    """
    external_id lives in external_references entries like:
      {"source_name": "mitre-attack", "external_id": "Txxxx", ...}
    """
    refs = obj.get("external_references", []) or []
    if not isinstance(refs, list):
        return None
    for ref in refs:
        if not isinstance(ref, dict):
            continue
        src = (ref.get("source_name") or "").lower()
        ext_id = ref.get("external_id")
        if ("mitre-attack" in src or "mitre-mobile-attack" in src) and isinstance(ext_id, str) and ext_id.strip():
            return ext_id.strip()
    return None


def _fmt_value(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (str, int, float, bool)):
        return str(v)
    try:
        return json.dumps(v, ensure_ascii=False)
    except Exception:
        return str(v)


# -------------------------------
# DOC BUILD (relationship vs non-relationship)
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

    created = obj.get("created")
    modified = obj.get("modified")
    spec_version = obj.get("spec_version")

    if stix_type == "relationship":
        return {
            "vendor": VENDOR_VALUE,
            "stix_id": stix_id.strip(),
            "type": "relationship",
            "scraped_date": scraped_date,
            "spec_version": spec_version,
            "created": created,
            "modified": modified,
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
        "external_id": extract_external_id(obj),
        "scraped_date": scraped_date,
        "spec_version": spec_version,
        "created": created,
        "modified": modified,
        "name": (obj.get("name") or "").strip(),
        "description": (obj.get("description") or "").strip(),
        "url": url,
    }


# -------------------------------
# TEXTNODE BUILD
# -------------------------------
def doc_to_textnode(doc: Dict[str, Any]) -> TextNode:
    """
    Build a TextNode:
      - text: key/value readable format for all fields except url
      - metadata: url + (vendor, stix_id, type) to preserve Qdrant precheck/filtering
    """
    url = doc.get("url")
    vendor = doc.get("vendor")
    stix_id = doc.get("stix_id")
    stix_type = doc.get("type")

    lines: List[str] = []
    for k, v in doc.items():
        if k == "url":
            continue
        val = _fmt_value(v)
        if val != "":
            lines.append(f"{k}: {val}")

    metadata: Dict[str, Any] = {}
    if isinstance(url, str) and url.strip():
        metadata["url"] = url.strip()

    # ✅ keep these so Qdrant payload still supports:
    # - filter-first precheck (stix_id)
    # - filtering (vendor)
    if isinstance(vendor, str) and vendor.strip():
        metadata["vendor"] = vendor.strip()
    if isinstance(stix_id, str) and stix_id.strip():
        metadata["stix_id"] = stix_id.strip()
    if isinstance(stix_type, str) and stix_type.strip():
        metadata["type"] = stix_type.strip()

    return TextNode(text="\n".join(lines).strip(), metadata=metadata)


def build_mitre_nodes_from_docs(docs: List[Dict[str, Any]]) -> Tuple[List[TextNode], List[TextNode]]:
    """
    Returns: (content_nodes, relationship_nodes)
    """
    content_nodes: List[TextNode] = []
    relationship_nodes: List[TextNode] = []

    for d in docs:
        n = doc_to_textnode(d)
        if (d.get("type") or "").strip() == "relationship":
            relationship_nodes.append(n)
        else:
            content_nodes.append(n)

    return content_nodes, relationship_nodes


# -------------------------------
# MAIN: SCRAPE + PRECHECK + BUILD NODES
# -------------------------------
def scrape_mitre_mobile_nodes(
    check_qdrant: bool = True,
) -> Tuple[List[TextNode], List[TextNode], List[TextNode]]:
    """
    Does:
      1) optional Qdrant precheck (filter-first) using vendor=mitre, reading payload['stix_id']
      2) download bundle
      3) keep only new STIX objects (by stix_id)
      4) build TextNodes + relationship TextNodes

    Returns:
      (content_nodes, relationship_nodes, all_nodes)
    """
    # local import to avoid circular imports at package import time
    from telco_cyber_chat.webscraping.scrape_core import get_qdrant_client

    existing_stix_ids: Set[str] = set()
    if check_qdrant:
        client = get_qdrant_client()
        existing_stix_ids = fetch_existing_mitre_stix_ids(client, QDRANT_COLLECTION)
        logger.info("[MITRE] Existing stix_id loaded from Qdrant: %d", len(existing_stix_ids))

    bundle = fetch_mobile_attack_bundle()
    objects = bundle.get("objects") or []
    if not isinstance(objects, list):
        logger.error("[MITRE] Unexpected 'objects' type: %s", type(objects))
        return [], [], []

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

        if check_qdrant and stix_id in existing_stix_ids:
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

    content_nodes, relationship_nodes = build_mitre_nodes_from_docs(docs)
    all_nodes = content_nodes + relationship_nodes

    logger.info(
        "[MITRE] Built nodes: content=%d relationship=%d total=%d",
        len(content_nodes), len(relationship_nodes), len(all_nodes),
    )

    return content_nodes, relationship_nodes, all_nodes


# -------------------------------
# ✅ Wrapper expected by your pipeline/graph
# -------------------------------
def scrape_mitre_mobile(check_qdrant: bool = True) -> List[TextNode]:
    """
    Public entrypoint: returns ALL nodes (content + relationship).
    """
    _, _, all_nodes = scrape_mitre_mobile_nodes(check_qdrant=check_qdrant)
    return all_nodes


# -------------------------------
# CLI debug
# -------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    content_nodes, relationship_nodes, all_nodes = scrape_mitre_mobile_nodes(check_qdrant=True)

    print(f"\n✅ Content nodes      : {len(content_nodes)}")
    print(f"✅ Relationship nodes : {len(relationship_nodes)}")
    print(f"✅ Total nodes        : {len(all_nodes)}")

    if all_nodes:
        n0 = all_nodes[0]
        print("\n--- SAMPLE NODE ---")
        print("metadata:", n0.metadata)
        print("text preview:\n", (n0.text[:800] + "...") if len(n0.text) > 800 else n0.text)
