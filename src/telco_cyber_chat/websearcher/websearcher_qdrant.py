# src/telco_cyber_chat/websearcher/websearcher_qdrant.py
from __future__ import annotations

import uuid
import asyncio
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from llama_index.core.schema import TextNode
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient, AsyncQdrantClient  # ✅ Added Async
from qdrant_client import models as qmodels

_UUID_NS = uuid.UUID("5b2f0b2c-7f55-4a3e-9ac2-2e2f3f3f5b4c")
_MODEL_CACHE: Dict[str, BGEM3FlagModel] = {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cuda_available() -> bool:
    """Lazy check - only when called, not at import time"""
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _get_bgem3(model_name: str) -> BGEM3FlagModel:
    """Cache model - safe, just returns cached instance if exists"""
    m = _MODEL_CACHE.get(model_name)
    if m is None:
        m = BGEM3FlagModel(model_name, use_fp16=_cuda_available())
        _MODEL_CACHE[model_name] = m
    return m


def _stable_doc_id(data_type: str, doc_name: str) -> str:
    return str(uuid.uuid5(_UUID_NS, f"{data_type}|{doc_name}"))


def _stable_point_id(doc_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_UUID_NS, f"{doc_id}|{chunk_index}"))


async def _ensure_payload_indexes_async(client: AsyncQdrantClient, collection: str) -> None:
    """Async version - create keyword indexes if missing"""
    try:
        from qdrant_client.http.models import PayloadIndexParams, PayloadSchemaType
    except Exception:
        return

    for key in ["doc_name", "data_type", "doc_id"]:
        try:
            await client.create_payload_index(
                collection_name=collection,
                field_name=key,
                field_schema=PayloadIndexParams(schema=PayloadSchemaType.KEYWORD),
            )
        except Exception:
            pass  # Already exists


def _ensure_payload_indexes(client: QdrantClient, collection: str) -> None:
    """Sync version - kept for backward compatibility"""
    try:
        from qdrant_client.http.models import PayloadIndexParams, PayloadSchemaType
    except Exception:
        return

    for key in ["doc_name", "data_type", "doc_id"]:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=key,
                field_schema=PayloadIndexParams(schema=PayloadSchemaType.KEYWORD),
            )
        except Exception:
            pass


async def upsert_nodes_bgem3_hybrid_async(
    nodes: List[TextNode],
    qdrant_url: str,
    collection: str,
    qdrant_api_key: Optional[str] = None,
    model_name: str = "BAAI/bge-m3",
    dense_name: str = "dense",
    sparse_name: str = "sparse",
    client: Optional[AsyncQdrantClient] = None,  # ✅ Async client
    model: Optional[BGEM3FlagModel] = None,
) -> int:
    """
    ✅ ASYNC version of upsert_nodes_bgem3_hybrid
    
    Embeds TextNodes (BGE-M3 dense+sparse) and upserts to Qdrant asynchronously.
    """
    if not nodes:
        return 0

    should_close = False
    if client is None:
        client = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_api_key)
        should_close = True

    # Embedding is CPU-bound, run in thread pool
    model = model or _get_bgem3(model_name)
    
    def _embed():
        texts = [n.text or "" for n in nodes]
        return model.encode(texts, return_dense=True, return_sparse=True, return_colbert=False)
    
    loop = asyncio.get_event_loop()
    emb = await loop.run_in_executor(None, _embed)

    dense_vecs = emb["dense_vecs"]
    sparse_w = emb["lexical_weights"]
    dim = len(dense_vecs[0])

    # Ensure collection exists
    try:
        await client.get_collection(collection)
    except Exception:
        await client.create_collection(
            collection_name=collection,
            vectors_config={dense_name: qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE)},
            sparse_vectors_config={sparse_name: qmodels.SparseVectorParams()},
            on_disk_payload=True,
        )

    # Ensure indexes
    await _ensure_payload_indexes_async(client, collection)

    now = _utc_now_iso()
    points: List[qmodels.PointStruct] = []

    for n, dv, sw in zip(nodes, dense_vecs, sparse_w):
        meta = n.metadata or {}
        doc_name = str(meta.get("doc_name") or "").strip()
        data_type = str(meta.get("data_type") or meta.get("doc_type") or "unstructured").strip()
        scraped_date = str(meta.get("scraped_date") or now)

        if not doc_name:
            raise ValueError("Missing required metadata 'doc_name' on TextNode.")
        if "chunk_index" not in meta:
            raise ValueError("Missing required metadata 'chunk_index' on TextNode.")

        chunk_index = int(meta["chunk_index"])
        doc_id = _stable_doc_id(data_type=data_type, doc_name=doc_name)
        pid = _stable_point_id(doc_id=doc_id, chunk_index=chunk_index)

        items = sorted((int(k), float(v)) for k, v in (sw or {}).items() if v)
        indices = [k for k, _ in items]
        values = [v for _, v in items]

        payload: Dict[str, Any] = {
            "doc_name": doc_name,
            "data_type": data_type,
            "doc_id": doc_id,
            "scraped_date": scraped_date,
            "node_content": n.text or "",
            "text_len": len(n.text or ""),
        }

        points.append(
            qmodels.PointStruct(
                id=pid,
                vector={
                    dense_name: dv,
                    sparse_name: qmodels.SparseVector(indices=indices, values=values),
                },
                payload=payload,
            )
        )

    await client.upsert(collection_name=collection, points=points, wait=True)
    
    if should_close:
        await client.close()
    
    return len(points)


# ✅ Keep sync version for backward compatibility
def upsert_nodes_bgem3_hybrid(
    nodes: List[TextNode],
    qdrant_url: str,
    collection: str,
    qdrant_api_key: Optional[str] = None,
    model_name: str = "BAAI/bge-m3",
    dense_name: str = "dense",
    sparse_name: str = "sparse",
    client: Optional[QdrantClient] = None,
    model: Optional[BGEM3FlagModel] = None,
) -> int:
    """Sync version - kept for backward compatibility"""
    if not nodes:
        return 0

    client = client or QdrantClient(url=qdrant_url, api_key=qdrant_api_key)
    model = model or _get_bgem3(model_name)

    texts = [n.text or "" for n in nodes]
    emb = model.encode(texts, return_dense=True, return_sparse=True, return_colbert=False)

    dense_vecs = emb["dense_vecs"]
    sparse_w = emb["lexical_weights"]
    dim = len(dense_vecs[0])

    try:
        client.get_collection(collection)
    except Exception:
        client.create_collection(
            collection_name=collection,
            vectors_config={dense_name: qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE)},
            sparse_vectors_config={sparse_name: qmodels.SparseVectorParams()},
            on_disk_payload=True,
        )

    _ensure_payload_indexes(client, collection)

    now = _utc_now_iso()
    points: List[qmodels.PointStruct] = []

    for n, dv, sw in zip(nodes, dense_vecs, sparse_w):
        meta = n.metadata or {}
        doc_name = str(meta.get("doc_name") or "").strip()
        data_type = str(meta.get("data_type") or meta.get("doc_type") or "unstructured").strip()
        scraped_date = str(meta.get("scraped_date") or now)

        if not doc_name:
            raise ValueError("Missing required metadata 'doc_name'")
        if "chunk_index" not in meta:
            raise ValueError("Missing required metadata 'chunk_index'")

        chunk_index = int(meta["chunk_index"])
        doc_id = _stable_doc_id(data_type=data_type, doc_name=doc_name)
        pid = _stable_point_id(doc_id=doc_id, chunk_index=chunk_index)

        items = sorted((int(k), float(v)) for k, v in (sw or {}).items() if v)
        indices = [k for k, _ in items]
        values = [v for _, v in items]

        payload: Dict[str, Any] = {
            "doc_name": doc_name,
            "data_type": data_type,
            "doc_id": doc_id,
            "scraped_date": scraped_date,
            "node_content": n.text or "",
            "text_len": len(n.text or ""),
        }

        points.append(
            qmodels.PointStruct(
                id=pid,
                vector={
                    dense_name: dv,
                    sparse_name: qmodels.SparseVector(indices=indices, values=values),
                },
                payload=payload,
            )
        )

    client.upsert(collection_name=collection, points=points, wait=True)
    return len(points)
