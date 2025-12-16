"""
qdrant_ingest.py

Upsert nodes (TextNodes) with dense+sparse embeddings
into a Qdrant collection for hybrid search.

Payload strategy:
- Always store: vendor, url, text, scraped_date
- Store stix_id ONLY when vendor == "mitre"
- Do NOT store other fields in payload (they belong in node.text already)

IMPORTANT (matches your node_builder.py):
- node.metadata contains ONLY {"url": "..."} (or {})
- vendor is inside node.text as a line like: "vendor: mitre"
  so we parse vendor (and stix id) from node.text when missing.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from llama_index.core.schema import TextNode
from qdrant_client import QdrantClient, models as qmodels


# ---------------------------------------------------------
# Qdrant config
# ---------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "Telco_CyberChat")

# Your dense dim (BGE-M3 is 1024)
DENSE_DIM = int(os.getenv("DENSE_DIM", "1024"))


def get_qdrant_client() -> QdrantClient:
    if not QDRANT_URL:
        raise ValueError("QDRANT_URL env var is required")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
_VENDOR_RE = re.compile(r"(?im)^\s*vendor\s*:\s*(.+?)\s*$")
# Try common keys in the text. We also fallback to matching a real STIX id pattern.
_STIX_LINE_RE = re.compile(r"(?im)^\s*(stix_id|stixid|id)\s*:\s*(.+?)\s*$")
_STIX_VALUE_RE = re.compile(r"(?i)\b(stix--[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b")


def _now_iso_utc() -> str:
    # Qdrant DATETIME expects ISO-8601; use UTC and drop micros for cleaner payloads
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _node_id(node: Any) -> Any:
    # TextNode uses id_
    if hasattr(node, "id_"):
        return getattr(node, "id_")
    # fallback if something else slips in
    if hasattr(node, "id"):
        return getattr(node, "id")
    raise TypeError(f"Unsupported node type: {type(node)} (missing id/id_)")


def _node_text(node: Any) -> str:
    return str(getattr(node, "text", "") or "").strip()


def _node_meta(node: Any) -> Dict[str, Any]:
    return dict(getattr(node, "metadata", {}) or {})


def _parse_vendor_from_text(text: str) -> str:
    if not text:
        return ""
    m = _VENDOR_RE.search(text)
    return m.group(1).strip() if m else ""


def _get_vendor(node: Any) -> str:
    """
    node_builder puts vendor in node.text, not metadata.
    We still allow metadata vendor/source/dataset just in case other sources set it.
    """
    meta = _node_meta(node)
    v = str(meta.get("vendor") or meta.get("source") or meta.get("dataset") or "").strip()
    if v:
        return v
    v2 = _parse_vendor_from_text(_node_text(node))
    return v2 or "unknown"


def _extract_stix_id_from_text(text: str) -> str:
    if not text:
        return ""
    # 1) Look at id/stix_id/stixId lines
    m = _STIX_LINE_RE.search(text)
    if m:
        val = m.group(2).strip()
        m2 = _STIX_VALUE_RE.search(val)
        return m2.group(1) if m2 else ""

    # 2) fallback: find any stix--... anywhere
    m3 = _STIX_VALUE_RE.search(text)
    return m3.group(1) if m3 else ""


def _get_stix_id(node: Any) -> str:
    meta = _node_meta(node)
    s = str(meta.get("stix_id") or meta.get("stixId") or "").strip()
    if s:
        return s
    return _extract_stix_id_from_text(_node_text(node))


def _ensure_payload_index(
    client: QdrantClient,
    collection: str,
    field_name: str,
    field_schema: qmodels.PayloadSchemaType = qmodels.PayloadSchemaType.KEYWORD,
) -> None:
    try:
        client.create_payload_index(
            collection_name=collection,
            field_name=field_name,
            field_schema=field_schema,
        )
        print(f"✅ Created payload index: {field_name} ({field_schema})")
    except Exception as e:
        # likely already exists
        print(f"ℹ️ Payload index '{field_name}' may already exist: {e}")


def ensure_collection_and_indexes(
    client: QdrantClient,
    collection: str,
    *,
    needs_stix_id: bool,
) -> None:
    # Create collection if missing
    try:
        client.get_collection(collection)
    except Exception:
        print(f"[QDRANT] Creating collection '{collection}' ...")
        client.create_collection(
            collection_name=collection,
            vectors_config={
                "dense": qmodels.VectorParams(size=DENSE_DIM, distance=qmodels.Distance.COSINE)
            },
            sparse_vectors_config={
                "sparse": qmodels.SparseVectorParams(index=qmodels.SparseIndexParams(on_disk=False))
            },
        )

    # Always index vendor + url + scraped_date
    _ensure_payload_index(client, collection, "vendor", qmodels.PayloadSchemaType.KEYWORD)
    _ensure_payload_index(client, collection, "url", qmodels.PayloadSchemaType.KEYWORD)
    _ensure_payload_index(client, collection, "scraped_date", qmodels.PayloadSchemaType.DATETIME)

    # Index stix_id only if we are ingesting MITRE nodes
    if needs_stix_id:
        _ensure_payload_index(client, collection, "stix_id", qmodels.PayloadSchemaType.KEYWORD)


def _sparse_dict_to_qdrant(sparse: Dict[int, float]) -> qmodels.SparseVector:
    items = sorted(((int(k), float(v)) for k, v in sparse.items()), key=lambda x: x[0])
    return qmodels.SparseVector(
        indices=[k for k, _ in items],
        values=[v for _, v in items],
    )


def _build_payload(node: Any, default_scraped_date: str) -> Dict[str, Any]:
    """
    Payload strategy:
    - vendor (always) -> parsed from text if needed
    - url (always) -> metadata["url"]
    - scraped_date (always)
    - stix_id only for vendor == mitre
    - text (always)
    """
    meta = _node_meta(node)
    url = str(meta.get("url") or "").strip()

    vendor = _get_vendor(node)

    scraped_date = str(
        meta.get("scraped_date") or meta.get("scraped_at") or meta.get("scrape_date") or ""
    ).strip() or default_scraped_date

    payload: Dict[str, Any] = {
        "vendor": vendor,
        "url": url,
        "scraped_date": scraped_date,
        "text": _node_text(node),
    }

    if vendor.lower() == "mitre":
        stix_id = _get_stix_id(node)
        if stix_id:
            payload["stix_id"] = stix_id

    return payload


# ---------------------------------------------------------
# Main upsert
# ---------------------------------------------------------
def upsert_nodes_to_qdrant(
    nodes: List[TextNode],  # ✅ TextNodes only
    embeddings: Dict[str, Dict[str, Any]],
    collection_name: Optional[str] = None,
    batch_size: int = 64,
) -> int:
    if not nodes:
        return 0

    coll = collection_name or QDRANT_COLLECTION
    client = get_qdrant_client()

    # One timestamp for this ingestion run (used if node doesn't already have scraped_date)
    run_scraped_date = _now_iso_utc()

    # Do we need stix_id index?
    needs_stix_id = any(_get_vendor(n).lower() == "mitre" and _get_stix_id(n) for n in nodes)

    ensure_collection_and_indexes(client, coll, needs_stix_id=needs_stix_id)

    total = 0
    batch: List[qmodels.PointStruct] = []

    for node in nodes:
        nid = str(_node_id(node))  # embeddings dict uses string keys
        emb = embeddings.get(nid)
        if not emb:
            continue

        dense = emb.get("dense")
        sparse = emb.get("sparse")

        if dense is None:
            continue

        dense_vec = dense.tolist() if isinstance(dense, np.ndarray) else dense

        vectors: Dict[str, Any] = {"dense": dense_vec}
        if isinstance(sparse, dict) and sparse:
            vectors["sparse"] = _sparse_dict_to_qdrant(sparse)

        payload = _build_payload(node, default_scraped_date=run_scraped_date)

        batch.append(
            qmodels.PointStruct(
                id=_node_id(node),
                vector=vectors,
                payload=payload,
            )
        )

        if len(batch) >= batch_size:
            client.upsert(collection_name=coll, points=batch)
            total += len(batch)
            batch = []

    if batch:
        client.upsert(collection_name=coll, points=batch)
        total += len(batch)

    print(f"[QDRANT] Upserted {total} points into '{coll}'.")
    return total
