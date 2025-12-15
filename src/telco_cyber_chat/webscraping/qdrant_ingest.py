"""
qdrant_ingest.py

Upsert nodes (TextNodes + RelationshipNodes) with dense+sparse embeddings
into a Qdrant collection for hybrid search.

Payload strategy (NEW):
- Always store: vendor, url, text
- Store stix_id ONLY when vendor == "mitre"
- Do NOT store other fields in payload (they belong in node.text already)
"""

from __future__ import annotations

import os
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


def get_qdrant_client() -> QdrantClient:
    if not QDRANT_URL:
        raise ValueError("QDRANT_URL env var is required")
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)


# ---------------------------------------------------------
# Helpers
# ---------------------------------------------------------
def _node_id(node: Any) -> Any:
    # Qdrant supports int or UUID; your pipeline may already provide those.
    if hasattr(node, "id_"):
        return getattr(node, "id_")
    if hasattr(node, "id"):
        return getattr(node, "id")
    raise TypeError(f"Unsupported node type: {type(node)} (missing id/id_)")

def _node_text(node: Any) -> str:
    return str(getattr(node, "text", "") or "").strip()

def _node_meta(node: Any) -> Dict[str, Any]:
    return dict(getattr(node, "metadata", {}) or {})

def _safe_vendor(meta: Dict[str, Any]) -> str:
    # prefer vendor, fallback to source/dataset
    v = (meta.get("vendor") or meta.get("source") or meta.get("dataset") or "").strip()
    return v or "unknown"

def _ensure_payload_index(client: QdrantClient, collection: str, field_name: str) -> None:
    try:
        client.create_payload_index(
            collection_name=collection,
            field_name=field_name,
            field_schema=qmodels.PayloadSchemaType.KEYWORD,
        )
        print(f"✅ Created payload index: {field_name}")
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

    # Always index vendor + url
    _ensure_payload_index(client, collection, "vendor")
    _ensure_payload_index(client, collection, "url")

    # Index stix_id only if we are ingesting MITRE nodes
    if needs_stix_id:
        _ensure_payload_index(client, collection, "stix_id")


def _sparse_dict_to_qdrant(sparse: Dict[int, float]) -> qmodels.SparseVector:
    items = sorted(((int(k), float(v)) for k, v in sparse.items()), key=lambda x: x[0])
    return qmodels.SparseVector(
        indices=[k for k, _ in items],
        values=[v for _, v in items],
    )


def _build_payload(node: Any) -> Dict[str, Any]:
    """
    NEW payload strategy:
    - vendor (always)
    - url (always)
    - stix_id only for vendor == mitre
    - text (always)
    """
    meta = _node_meta(node)
    vendor = _safe_vendor(meta)
    url = (meta.get("url") or "").strip()

    payload: Dict[str, Any] = {
        "vendor": vendor,
        "url": url,
        "text": _node_text(node),
    }

    # keep stix_id ONLY for MITRE
    if vendor.lower() == "mitre":
        stix_id = (meta.get("stix_id") or meta.get("stixId") or "").strip()
        if stix_id:
            payload["stix_id"] = stix_id

    return payload


# ---------------------------------------------------------
# Main upsert
# ---------------------------------------------------------
def upsert_nodes_to_qdrant(
    nodes: List[Any],  # TextNode or RelationshipNode
    embeddings: Dict[str, Dict[str, Any]],
    collection_name: Optional[str] = None,
    batch_size: int = 64,
) -> int:
    if not nodes:
        return 0

    coll = collection_name or QDRANT_COLLECTION
    client = get_qdrant_client()

    # Do we need stix_id index?
    needs_stix_id = False
    for n in nodes:
        meta = _node_meta(n)
        vendor = _safe_vendor(meta).lower()
        if vendor == "mitre" and (meta.get("stix_id") or meta.get("stixId")):
            needs_stix_id = True
            break

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

        payload = _build_payload(node)

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
