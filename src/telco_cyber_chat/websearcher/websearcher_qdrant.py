# src/telco_cyber_chat/websearcher/websearcher_qdrant.py
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

from llama_index.core.schema import TextNode
from qdrant_client import AsyncQdrantClient, QdrantClient
from qdrant_client import models as qmodels

_UUID_NS = uuid.UUID("5b2f0b2c-7f55-4a3e-9ac2-2e2f3f3f5b4c")


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def stable_doc_id(*, data_type: str, doc_name: str) -> str:
    dt = (data_type or "").strip()
    dn = (doc_name or "").strip().lower()
    return str(uuid.uuid5(_UUID_NS, f"{dt}|{dn}"))


def stable_point_id(*, doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_UUID_NS, f"{doc_id}|{int(chunk_index)}"))


async def ensure_payload_indexes_async(client: AsyncQdrantClient, collection: str) -> None:
    """
    ✅ Create payload keyword indexes if missing (safe to call repeatedly).
    This makes doc_name and data_type filterable + fast in Qdrant.
    """
    try:
        from qdrant_client.http.models import PayloadIndexParams, PayloadSchemaType
    except Exception:
        return

    # ✅ These are the fields you asked to be "indexed"
    for key in ["doc_name", "data_type", "doc_id"]:
        try:
            await client.create_payload_index(
                collection_name=collection,
                field_name=key,
                field_schema=PayloadIndexParams(schema=PayloadSchemaType.KEYWORD),
            )
        except Exception:
            pass


async def ensure_collection_hybrid_async(
    client: AsyncQdrantClient,
    collection: str,
    *,
    dim: int,
    dense_name: str = "dense",
    sparse_name: str = "sparse",
) -> None:
    try:
        await client.get_collection(collection)
        return
    except Exception:
        await client.create_collection(
            collection_name=collection,
            vectors_config={
                dense_name: qmodels.VectorParams(size=int(dim), distance=qmodels.Distance.COSINE)
            },
            sparse_vectors_config={sparse_name: qmodels.SparseVectorParams()},
            on_disk_payload=True,
        )


def _as_list(vec: Any) -> Optional[List[float]]:
    if vec is None:
        return None
    try:
        if hasattr(vec, "tolist"):
            out = vec.tolist()
            return out if isinstance(out, list) else list(out)
    except Exception:
        pass
    try:
        return list(vec)
    except Exception:
        return None


def _node_id(node: TextNode) -> str:
    v = getattr(node, "id_", None)
    return str(v).strip() if v is not None else ""


def _infer_dim_from_emb_map(emb_map: Dict[str, Any]) -> Optional[int]:
    for emb in emb_map.values():
        dense = getattr(emb, "dense", None) if emb is not None else None
        dense_list = _as_list(dense)
        if dense_list:
            return len(dense_list)
    return None


def _sparse_to_qdrant(sparse: Any) -> qmodels.SparseVector:
    sparse = sparse or {}
    items: List[Tuple[int, float]] = []
    if isinstance(sparse, dict):
        for k, v in sparse.items():
            try:
                ki = int(k)
                vf = float(v)
                if vf:
                    items.append((ki, vf))
            except Exception:
                continue

    items.sort(key=lambda x: x[0])
    indices = [k for k, _ in items]
    values = [v for _, v in items]
    return qmodels.SparseVector(indices=indices, values=values)


async def upsert_nodes_hybrid_from_embeddings_async(
    *,
    nodes: List[TextNode],
    emb_map: Dict[str, Any],
    client: AsyncQdrantClient,
    collection: str,
    dense_name: str = "dense",
    sparse_name: str = "sparse",
) -> int:
    if not nodes:
        return 0

    dim = _infer_dim_from_emb_map(emb_map)
    if not dim:
        return 0

    await ensure_collection_hybrid_async(
        client, collection, dim=dim, dense_name=dense_name, sparse_name=sparse_name
    )
    await ensure_payload_indexes_async(client, collection)

    now = _utc_now_iso()
    points: List[qmodels.PointStruct] = []

    for n in nodes:
        meta = n.metadata or {}

        doc_name = str(meta.get("doc_name") or "").strip()  # ✅ should be Drive display name now
        data_type = str(meta.get("data_type") or "unstructured").strip()  # ✅ forced upstream
        scraped_date = str(meta.get("scraped_date") or now)

        if not doc_name:
            raise ValueError("Missing required metadata 'doc_name' on TextNode.")
        if "chunk_index" not in meta:
            raise ValueError("Missing required metadata 'chunk_index' on TextNode.")

        chunk_index = int(meta["chunk_index"])
        doc_id = stable_doc_id(data_type=data_type, doc_name=doc_name)
        pid = stable_point_id(doc_id=doc_id, chunk_index=chunk_index)

        nid = _node_id(n)
        emb = emb_map.get(nid)
        if not emb:
            continue

        dense_vec = _as_list(getattr(emb, "dense", None))
        if not dense_vec:
            continue

        sparse_vec = _sparse_to_qdrant(getattr(emb, "sparse", None))

        payload: Dict[str, Any] = {
            "doc_name": doc_name,          # ✅ indexed
            "data_type": data_type,        # ✅ indexed
            "doc_id": doc_id,              # ✅ indexed
            "scraped_date": scraped_date,
            "node_content": n.text or "",
            "text_len": len(n.text or ""),
            "chunk_index": chunk_index,
        }

        for k in ["drive_file_id", "drive_file_name", "url"]:
            if k in meta and meta.get(k) is not None:
                payload[k] = meta.get(k)

        points.append(
            qmodels.PointStruct(
                id=pid,
                vector={dense_name: dense_vec, sparse_name: sparse_vec},
                payload=payload,
            )
        )

    if not points:
        return 0

    await client.upsert(collection_name=collection, points=points, wait=True)
    return len(points)
