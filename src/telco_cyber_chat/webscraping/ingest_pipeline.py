"""
ingest_pipeline.py

This module MUST expose: ingest_all_sources
because scraper_graph.py imports it like:

    from telco_cyber_chat.webscraping.ingest_pipeline import ingest_all_sources
"""

from __future__ import annotations

import os
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from llama_index.core.schema import TextNode

from telco_cyber_chat.webscraping.node_builder import build_vendor_nodes, build_mitre_nodes
from telco_cyber_chat.webscraping.node_embedder import embed_nodes_hybrid
from telco_cyber_chat.webscraping.qdrant_ingest import upsert_nodes_to_qdrant

logger = logging.getLogger(__name__)


def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_records_list(path: str) -> List[Dict[str, Any]]:
    data = _load_json(path)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        recs = data.get("records")
        if isinstance(recs, list):
            return [x for x in recs if isinstance(x, dict)]
    return []


def _load_mitre_df(mitre_json_path: str) -> pd.DataFrame:
    bundle = _load_json(mitre_json_path)
    objects = bundle.get("objects", []) if isinstance(bundle, dict) else []
    if not isinstance(objects, list):
        objects = []
    return pd.DataFrame([o for o in objects if isinstance(o, dict)])


def _vendor_inputs_from_env() -> List[Tuple[str, str]]:
    mapping = [
        ("nokia", os.getenv("NOKIA_JSON_PATH", "").strip()),
        ("huawei", os.getenv("HUAWEI_JSON_PATH", "").strip()),
        ("ericsson", os.getenv("ERICSSON_JSON_PATH", "").strip()),
        ("cisco", os.getenv("CISCO_JSON_PATH", "").strip()),
        ("variot", os.getenv("VARIOT_JSON_PATH", "").strip()),
    ]
    return [(v, p) for (v, p) in mapping if p]


async def ingest_all_sources(
    *args,
    **kwargs,
) -> Dict[str, Any]:
    """
    ✅ REQUIRED export for scraper_graph.py

    Reads already-scraped JSONs (paths via env vars),
    builds TextNodes, embeds, then upserts to Qdrant.
    """
    qdrant_collection: Optional[str] = kwargs.get("qdrant_collection")
    concurrency: int = int(kwargs.get("concurrency", 2))
    embed_batch_size: int = int(kwargs.get("embed_batch_size", 20))
    upsert_batch_size: int = int(kwargs.get("upsert_batch_size", 64))

    all_nodes: List[TextNode] = []

    # Vendor JSONs -> TextNodes
    for vendor, path in _vendor_inputs_from_env():
        try:
            records = _load_records_list(path)
            nodes = build_vendor_nodes(records, vendor=vendor)
            all_nodes.extend(nodes)
            logger.info("Loaded %d nodes for vendor=%s from %s", len(nodes), vendor, path)
        except Exception:
            logger.exception("Failed loading vendor=%s from %s", vendor, path)

    # MITRE bundle -> TextNodes
    mitre_path = os.getenv("MITRE_JSON_PATH", "").strip()
    if mitre_path:
        try:
            df = _load_mitre_df(mitre_path)
            content_nodes, relationship_nodes = build_mitre_nodes(df)
            all_nodes.extend(content_nodes)
            all_nodes.extend(relationship_nodes)
            logger.info(
                "Loaded MITRE nodes: content=%d rel=%d from %s",
                len(content_nodes), len(relationship_nodes), mitre_path
            )
        except Exception:
            logger.exception("Failed loading MITRE from %s", mitre_path)

    if not all_nodes:
        return {"ok": True, "nodes": 0, "upserted": 0, "message": "No nodes to ingest."}

    embeddings = await embed_nodes_hybrid(
        all_nodes,
        concurrency=concurrency,
        batch_size=embed_batch_size,
    )

    upserted = upsert_nodes_to_qdrant(
        all_nodes,
        embeddings=embeddings,
        collection_name=qdrant_collection,
        batch_size=upsert_batch_size,
    )

    return {
        "ok": True,
        "nodes": len(all_nodes),
        "upserted": int(upserted),
        "collection": qdrant_collection or os.getenv("QDRANT_COLLECTION", "Telco_CyberChat"),
    }
