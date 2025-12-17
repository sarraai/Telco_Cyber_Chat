"""
qdrant_ingest.py

Upsert nodes (TextNodes) with dense+sparse embeddings into Qdrant for hybrid search.

Payload strategy (matches your node_builder.py):
- Always store: vendor, url, text, scraped_date
- Store stix_id ONLY when vendor == "mitre"
- Do NOT store other fields in payload (they belong in node.text already)

IMPORTANT:
- node.metadata contains ONLY {"url": "..."} (or {})
- vendor is inside node.text as a line like: "vendor: mitre"
  so we parse vendor (and stix id) from node.text when missing.

✅ UPDATED:
- Does NOT require node.id_ / node.id.
- Generates a stable ID if missing (prefer metadata["url"], else hash text+metadata).
"""

from __future__ import annotations

import os
import re
import json
import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

import numpy as np
from qdrant_client import QdrantClient, models as qmodels

# ---------------------------------------------------------
# Qdrant config
# ---------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "Telco_CyberChat")

# Your dense dim (BGE-M3 is 1024)
DENSE_DIM = int(os.getenv("DENSE_DIM", "1024"))

# ---------------------------------------------------------
# Text parsing helpers (because vendor/stix_id live in node.text)
# ---------------------------------------------------------
_VENDOR_RE = re.compile(r"(?im)^\s*vendor\s*:\s*(.+?)\s*$")
_STIX_LINE_RE = re.compile(r"(?im)^\s*(stix_id|stixid|id)\s*:\s*(.+?)\s*$")
_STIX_VALUE_RE = re.compile(
    r"(?i)\b(stix--[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b"
)


def get_qdrant_client() -> QdrantClient:
    if not QDRANT_URL:
        raise ValueError("QDRANT_URL env var is required")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)


def _now_iso_utc() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _node_text(node: Any) -> str:
    return str(getattr(node, "text", "") or "").strip()


def _node_meta(node: Any) -> Dict[str, Any]:
    md = getattr(node, "metadata", None)
    return md if isinstance(md, dict) else {}


def _get_existing_id(node: Any) -> Optional[str]:
    if hasattr(node, "id_"):
        v = getattr(node, "id_", None)
        v = str(v).strip() if v is not None else ""
        return v or None
    if hasattr(node, "id"):
        v = getattr(node, "id", None)
        v = str(v).strip() if v is not None else ""
        return v or None
    return None


def _set_node_id(node: Any, new_id: str) -> None:
    """Best-effort write-back so downstream stays consistent."""
    try:
        if hasattr(node, "id_"):
            setattr(node, "id_", new_id)
            return
        if hasattr(node, "id"):
            setattr(node, "id", new_id)
            return
        # If it has neither, we still try to attach id_ (works for many python objects)
        setattr(node, "id_", new_id)
    except Exception:
        pass


def _generate_stable_id(node: Any) -> str:
    md = _node_meta(node)
    url = md.get("url")
    if isinstance(url, str) and url.strip():
        raw = f"{type(node).__name__}|url|{url.strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    text = _node_text(node)
    try:
        md_norm = json.dumps(md, sort_keys=True, ensure_ascii=False)
    except Exception:
        md_norm = str(md)

    raw = f"{type(node).__name__}|text|{text}|meta|{md_norm}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _node_id(node: Any) -> str:
    """
    ✅ No longer requires id/id_.
    Returns existing ID if present, else generates a stable one and writes it back.
    """
    existing = _get_existing_id(node)
    if existing:
        return existing
    new_id = _generate_stable_id(node)
    _set_node_id(node, new_id)
    return new_id


def _parse_vendor_from_text(text: str) -> str:
    if not text:
        return ""
    m = _VENDOR_RE.search(text)
    return m.group(1).strip() if m else ""


def _get_vendor(node: Any) -> str:
    meta = _node_meta(node)
    v = str(meta.get("vendor") or meta.get("source") or meta.get("dataset") or "").strip()
    if v:
        return v
    v2 = _parse_vendor_from_text(_node_text(node))
    return v2 or "unknown"


def _extract_stix_id_from_text(text: str) -> str:
    if not text:
        return ""
    m = _STIX_LINE_RE.search(text)
    if m:
        val = m.group(2).strip()
        m2 = _STIX_VALUE_RE.search(val)
        return m2.group(1) if m2 else ""
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
    field_schema: qmodels.PayloadSchemaType,
) -> None:
    try:
        client.create_payload_index(
            collection_name=collection,
            field_name=field_name,
            field_schema=field_schema,
        )
        print(f"✅ Created payload index: {field_name} ({field_schema})")
    except Exception as e:
        print(f"ℹ️ Payload index '{field_name}' may already exist: {e}")


def ensure_collection_and_indexes(client: QdrantClient, collection: str, *, needs_stix_id: bool) -> None:
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

    _ensure_payload_index(client, collection, "vendor", qmodels.PayloadSchemaType.KEYWORD)
    _ensure_payload_index(client, collection, "url", qmodels.PayloadSchemaType.KEYWORD)
    _ensure_payload_index(client, collection, "scraped_date", qmodels.PayloadSchemaType.DATETIME)

    if needs_stix_id:
        _ensure_payload_index(client, collection, "stix_id", qmodels.PayloadSchemaType.KEYWORD)


def _sparse_dict_to_qdrant(sparse: Dict[int, float]) -> qmodels.SparseVector:
    items = sorted(((int(k), float(v)) for k, v in sparse.items()), key=lambda x: x[0])
    return qmodels.SparseVector(
        indices=[k for k, _ in items],
        values=[v for _, v in items],
    )


def _build_payload(node: Any, default_scraped_date: str) -> Dict[str, Any]:
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


def upsert_nodes_to_qdrant(
    nodes: List[Any],
    embeddings: Dict[str, Dict[str, Any]],
    collection_name: Optional[str] = None,
    batch_size: int = 64,
) -> int:
    if not nodes:
        return 0

    coll = collection_name or QDRANT_COLLECTION
    client = get_qdrant_client()

    run_scraped_date = _now_iso_utc()
    needs_stix_id = any(_get_vendor(n).lower() == "mitre" and _get_stix_id(n) for n in nodes)
    ensure_collection_and_indexes(client, coll, needs_stix_id=needs_stix_id)

    total = 0
    batch: List[qmodels.PointStruct] = []

    for node in nodes:
        nid = _node_id(node)  # ✅ works even if node had no id
        emb = embeddings.get(str(nid))  # embeddings keys are strings
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
                id=str(nid),  # ✅ Qdrant point id as string
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
