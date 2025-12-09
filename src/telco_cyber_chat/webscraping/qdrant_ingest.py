"""
qdrant_ingest.py

Upsert LlamaIndex TextNodes (with BGE-M3 dense + sparse embeddings)
into a Qdrant collection for hybrid search.
"""

from __future__ import annotations

import os
from typing import List, Dict, Any, Optional

import numpy as np
from qdrant_client import QdrantClient, models as qmodels
from llama_index.core.schema import TextNode


# ---------------------------------------------------------
# Qdrant config (reuse same env vars as your retriever)
# ---------------------------------------------------------
QDRANT_URL = os.getenv("QDRANT_URL", "").strip()
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", "").strip()
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "Telco_CyberChat")


def get_qdrant_client() -> QdrantClient:
    if not QDRANT_URL:
        raise ValueError("QDRANT_URL env var is required")
    # API key can be optional if you run Qdrant locally
    return QdrantClient(
        url=QDRANT_URL,
        api_key=QDRANT_API_KEY or None,
    )


# ---------------------------------------------------------
# Upsert helpers
# ---------------------------------------------------------
def _build_payload_from_node(node: TextNode) -> Dict[str, Any]:
    """
    Build Qdrant payload (metadata) from a TextNode.

    - text: full node content (title + description + extras)
    - url: taken from node.metadata["url"] if present
    - source/title/etc: directly from metadata
    """
    meta = dict(node.metadata or {})
    url = meta.get("url", "")

    payload: Dict[str, Any] = {
        "text": node.text,
        "url": url,
        # keep other metadata fields too (vendor, title, product, etc.)
        **meta,
    }
    return payload


def upsert_nodes_to_qdrant(
    nodes: List[TextNode],
    embeddings: Dict[str, Dict[str, Any]],
    collection_name: Optional[str] = None,
    batch_size: int = 64,
) -> int:
    """
    Upsert embedded nodes into Qdrant with hybrid (dense + sparse) vectors.

    Args:
        nodes: list of TextNode from node_builder.py
        embeddings: mapping node.id_ -> {"dense": np.ndarray | None,
                                         "sparse": Dict[int, float] | None}
        collection_name: Qdrant collection (defaults to QDRANT_COLLECTION)
        batch_size: upsert in small batches

    Returns:
        Total number of points successfully sent to Qdrant.
    """
    if not nodes:
        return 0

    coll = collection_name or QDRANT_COLLECTION
    client = get_qdrant_client()

    total = 0
    batch: List[qmodels.PointStruct] = []

    for node in nodes:
        node_id = node.id_
        emb = embeddings.get(node_id)
        if not emb:
            # no embedding computed for this node
            continue

        dense = emb.get("dense")
        sparse = emb.get("sparse")

        if dense is None:
            # we require at least dense vector
            continue

        # --- Build named vectors ---
        dense_vec = dense.tolist() if isinstance(dense, np.ndarray) else dense
        
        # Create named vectors dict
        vectors = {
            "dense": dense_vec,
        }

        # Add sparse vector if available
        if sparse:
            sparse_vec = qmodels.SparseVector(
                indices=list(sparse.keys()),
                values=[float(v) for v in sparse.values()],
            )
            vectors["sparse"] = sparse_vec

        payload = _build_payload_from_node(node)

        # Create point with named vectors
        point = qmodels.PointStruct(
            id=node_id,
            vector=vectors,  # Dict of named vectors (dense + sparse)
            payload=payload,
        )
        batch.append(point)

        # flush in batches
        if len(batch) >= batch_size:
            client.upsert(collection_name=coll, points=batch)
            total += len(batch)
            batch = []

    # flush remaining
    if batch:
        client.upsert(collection_name=coll, points=batch)
        total += len(batch)

    print(f"[QDRANT] Upserted {total} points into collection '{coll}'.")
    return total


# Alias for backward compatibility with ingest_pipeline.py
upsert_embeddings = upsert_nodes_to_qdrant
