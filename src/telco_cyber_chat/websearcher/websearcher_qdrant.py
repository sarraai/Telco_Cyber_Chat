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


def _stable_id(vendor: str, url: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_UUID_NS, f"{vendor}|{url}|{chunk_index}"))


def _extract_chunk_index(text: str) -> int:
    for line in (text.splitlines()[:15]):
        if line.startswith("chunk_index:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except Exception:
                return 0
    return 0


def _extract_header_value(text: str, key: str) -> Optional[str]:
    prefix = f"{key}:"
    for line in text.splitlines()[:30]:
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return None


def _get_bgem3(model_name: str) -> BGEM3FlagModel:
    # Cache model so repeated calls in same process don't reload weights
    m = _MODEL_CACHE.get(model_name)
    if m is None:
        m = BGEM3FlagModel(model_name, use_fp16=True)
        _MODEL_CACHE[model_name] = m
    return m


def upsert_nodes_bgem3_hybrid(
    nodes: List[TextNode],
    qdrant_url: str,
    collection: str,
    vendor: str,
    qdrant_api_key: Optional[str] = None,
    model_name: str = "BAAI/bge-m3",
    dense_name: str = "dense",
    sparse_name: str = "sparse",
    client: Optional[QdrantClient] = None,
    model: Optional[BGEM3FlagModel] = None,
) -> int:
    if not nodes:
        return 0

    client = client or QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Embed (reuse model if passed / cached)
    model = model or _get_bgem3(model_name)

    texts = [n.text for n in nodes]
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
            vectors_config={
                dense_name: qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE)
            },
            sparse_vectors_config={
                sparse_name: qmodels.SparseVectorParams()
            },
        )

    now = _utc_now_iso()
    points: List[qmodels.PointStruct] = []

    for n, dv, sw in zip(nodes, dense_vecs, sparse_w):
        url = (n.metadata or {}).get("url", "") or ""
        chunk_index = _extract_chunk_index(n.text)
        pid = _stable_id(vendor=vendor, url=url, chunk_index=chunk_index)

        indices = list(sw.keys())
        values = list(sw.values())

        # ✅ store doc_name for "skip by name"
        doc_name = _extract_header_value(n.text, "doc_name")

        payload: Dict[str, Any] = {
            "vendor": vendor,
            "url": url,
            "text": n.text,
            "scraped_date": now,
        }
        if doc_name:
            payload["doc_name"] = doc_name

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

    client.upsert(collection_name=collection, points=points)
    return len(points)
