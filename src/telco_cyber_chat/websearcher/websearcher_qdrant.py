from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from llama_index.core.schema import TextNode
from FlagEmbedding import BGEM3FlagModel
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels


_UUID_NS = uuid.UUID("5b2f0b2c-7f55-4a3e-9ac2-2e2f3f3f5b4c")

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def _stable_id(vendor: str, url: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_UUID_NS, f"{vendor}|{url}|{chunk_index}"))

def _extract_chunk_index(text: str) -> int:
    for line in (text.splitlines()[:12]):
        if line.startswith("chunk_index:"):
            try:
                return int(line.split(":", 1)[1].strip())
            except Exception:
                return 0
    return 0

def upsert_nodes_bgem3_hybrid(
    nodes: List[TextNode],
    qdrant_url: str,
    collection: str,
    vendor: str,
    qdrant_api_key: Optional[str] = None,
    model_name: str = "BAAI/bge-m3",
    dense_name: str = "dense",
    sparse_name: str = "sparse",
) -> int:
    if not nodes:
        return 0

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # Embed
    model = BGEM3FlagModel(model_name, use_fp16=True)
    texts = [n.text for n in nodes]
    emb = model.encode(texts, return_dense=True, return_sparse=True, return_colbert=False)

    dense_vecs = emb["dense_vecs"]                     # List[List[float]]
    sparse_w = emb["lexical_weights"]                  # List[Dict[int,float]]

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

    points: List[qmodels.PointStruct] = []
    now = _utc_now_iso()

    for n, dv, sw in zip(nodes, dense_vecs, sparse_w):
        url = (n.metadata or {}).get("url", "") or ""
        chunk_index = _extract_chunk_index(n.text)
        pid = _stable_id(vendor=vendor, url=url, chunk_index=chunk_index)

        indices = list(sw.keys())
        values = list(sw.values())

        payload: Dict[str, Any] = {
            "vendor": vendor,
            "url": url,
            "text": n.text,
            "scraped_date": now,
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

    client.upsert(collection_name=collection, points=points)
    return len(points)
