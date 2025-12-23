# src/telco_cyber_chat/websearcher/websearcher_qdrant.py
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from llama_index.core.schema import TextNode
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

_UUID_NS = uuid.UUID("5b2f0b2c-7f55-4a3e-9ac2-2e2f3f3f5b4c")
_MODEL_CACHE: Dict[str, BGEM3FlagModel] = {}


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _cuda_available() -> bool:
    # Avoid importing torch at module import time (helps LangSmith startup)
    try:
        import torch  # type: ignore
        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _get_bgem3(model_name: str) -> BGEM3FlagModel:
    """
    Cache model so repeated calls in the same process don't reload weights.
    fp16 only when CUDA is available.
    """
    m = _MODEL_CACHE.get(model_name)
    if m is None:
        m = BGEM3FlagModel(model_name, use_fp16=_cuda_available())
        _MODEL_CACHE[model_name] = m
    return m


def _stable_doc_id(doc_type: str, doc_name: str) -> str:
    # stable per document (doc_type + doc_name)
    return str(uuid.uuid5(_UUID_NS, f"{doc_type}|{doc_name}"))


def _stable_point_id(doc_id: str, chunk_index: int) -> str:
    # stable per chunk inside a document
    return str(uuid.uuid5(_UUID_NS, f"{doc_id}|{chunk_index}"))


def _ensure_payload_indexes(client: QdrantClient, collection: str) -> None:
    """Create keyword indexes if missing (safe to call repeatedly)."""
    try:
        from qdrant_client.http.models import PayloadIndexParams, PayloadSchemaType
    except Exception:
        return

    for key in ["doc_name", "doc_type", "doc_id"]:
        try:
            client.create_payload_index(
                collection_name=collection,
                field_name=key,
                field_schema=PayloadIndexParams(schema=PayloadSchemaType.keyword),
            )
        except Exception:
            # already exists or backend doesn't support -> ignore
            pass


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
    """
    Embeds TextNodes (BGE-M3 dense+sparse) and upserts them into Qdrant.

    REQUIRED per-node metadata:
      - doc_name: str   (folder-derived name, no extension)
      - doc_type: str   (e.g., "unstructured")
      - chunk_index: int  (0..N-1 for that doc)

    Optional:
      - scraped_date: ISO str  (if missing -> set now)

    Payload written:
      doc_name (keyword), doc_type (keyword), doc_id (keyword), scraped_date, node_content, text_len
    """
    if not nodes:
        return 0

    client = client or QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Embed (reuse model if passed / cached)
    model = model or _get_bgem3(model_name)

    texts = [n.text or "" for n in nodes]
    emb = model.encode(texts, return_dense=True, return_sparse=True, return_colbert=False)

    dense_vecs = emb["dense_vecs"]          # List[List[float]]
    sparse_w = emb["lexical_weights"]       # List[Dict[int,float]]
    dim = len(dense_vecs[0])

    # Ensure collection exists (dense + sparse)
    try:
        client.get_collection(collection)
    except Exception:
        client.create_collection(
            collection_name=collection,
            vectors_config={dense_name: qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE)},
            sparse_vectors_config={sparse_name: qmodels.SparseVectorParams()},
            on_disk_payload=True,
        )

    # Ensure keyword indexes for filtering (prevents the 400 "Index required" issue)
    _ensure_payload_indexes(client, collection)

    now = _utc_now_iso()
    points: List[qmodels.PointStruct] = []

    for idx, (n, dv, sw) in enumerate(zip(nodes, dense_vecs, sparse_w)):
        meta = (n.metadata or {})

        doc_name = str(meta.get("doc_name") or "").strip()
        doc_type = str(meta.get("doc_type") or "unstructured").strip()
        scraped_date = str(meta.get("scraped_date") or now)

        if not doc_name:
            raise ValueError("Missing required metadata 'doc_name' on TextNode.")
        if "chunk_index" not in meta:
            # Strongly enforce determinism (prevents overwriting chunks).
            raise ValueError("Missing required metadata 'chunk_index' on TextNode.")
        try:
            chunk_index = int(meta.get("chunk_index"))  # type: ignore[arg-type]
        except Exception:
            raise ValueError(f"Invalid 'chunk_index' in metadata: {meta.get('chunk_index')!r}")

        doc_id = _stable_doc_id(doc_type=doc_type, doc_name=doc_name)
        pid = _stable_point_id(doc_id=doc_id, chunk_index=chunk_index)

        # Sparse vector: ensure ints/floats + sorted indices
        items = sorted((int(k), float(v)) for k, v in (sw or {}).items() if v)
        indices = [k for k, _ in items]
        values = [v for _, v in items]

        payload: Dict[str, Any] = {
            "doc_name": doc_name,          # keyword
            "doc_type": doc_type,          # keyword
            "doc_id": doc_id,              # keyword (stable)
            "scraped_date": scraped_date,  # ISO string
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
