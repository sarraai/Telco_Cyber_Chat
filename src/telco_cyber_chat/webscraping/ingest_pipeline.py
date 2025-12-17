"""
ingest_pipeline.py

This module MUST expose: ingest_all_sources
because scraper_graph.py imports it like:

    from telco_cyber_chat.webscraping.ingest_pipeline import ingest_all_sources

What this version fixes:
- Implements `check_qdrant=True` to SKIP already-ingested items (vendor+url; MITRE via stix_id when possible)
- Returns `per_source` so scraper_graph can display per-vendor new counts
- Canonicalizes URLs before checking and before building nodes (so dedupe is stable)
- Emits lightweight logs/prints so you can SEE it is running in Studio
"""

from __future__ import annotations

import os
import re
import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from llama_index.core.schema import TextNode

from qdrant_client import QdrantClient, models as qmodels

from telco_cyber_chat.webscraping.node_embed import embed_nodes_hybrid
from telco_cyber_chat.webscraping.qdrant_ingest import upsert_nodes_to_qdrant
from telco_cyber_chat.webscraping.scrape_core import canonical_url, normalize_vendor

logger = logging.getLogger(__name__)

MITRE_SOURCE_KEY = "mitre_mobile"
_STIX_RE = re.compile(r"(?i)\b(stix--[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b")


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


def _get_collection_name(explicit: Optional[str]) -> str:
    return (explicit or os.getenv("QDRANT_COLLECTION", "Telco_CyberChat")).strip()


def _get_qdrant_client_safe() -> Optional[QdrantClient]:
    qurl = os.getenv("QDRANT_URL", "").strip()
    if not qurl:
        return None
    qkey = os.getenv("QDRANT_API_KEY", "").strip() or None
    try:
        return QdrantClient(url=qurl, api_key=qkey)
    except Exception:
        logger.exception("Failed to create QdrantClient; proceeding without DB checks.")
        return None


def _qdrant_has_vendor_url(
    client: QdrantClient,
    collection: str,
    vendor: str,
    url: str,
) -> bool:
    # Assumes payload has keyword fields: vendor, url
    flt = qmodels.Filter(
        must=[
            qmodels.FieldCondition(key="vendor", match=qmodels.MatchValue(value=normalize_vendor(vendor))),
            qmodels.FieldCondition(key="url", match=qmodels.MatchValue(value=url)),
        ]
    )
    try:
        res = client.count(collection_name=collection, count_filter=flt, exact=False)
        return (res.count or 0) > 0
    except Exception:
        # If DB check fails, do NOT block ingestion
        logger.warning("Qdrant vendor+url check failed; proceeding as not ingested.")
        return False


def _qdrant_has_mitre_stix(
    client: QdrantClient,
    collection: str,
    stix_id: str,
) -> bool:
    flt = qmodels.Filter(
        must=[
            qmodels.FieldCondition(key="vendor", match=qmodels.MatchValue(value="mitre")),
            qmodels.FieldCondition(key="stix_id", match=qmodels.MatchValue(value=stix_id)),
        ]
    )
    try:
        res = client.count(collection_name=collection, count_filter=flt, exact=False)
        return (res.count or 0) > 0
    except Exception:
        logger.warning("Qdrant stix_id check failed; proceeding as not ingested.")
        return False


def _extract_stix_from_text(text: str) -> str:
    if not text:
        return ""
    m = _STIX_RE.search(text)
    return m.group(1) if m else ""


async def ingest_all_sources(*args, **kwargs) -> Dict[str, Any]:
    """
    ✅ REQUIRED export for scraper_graph.py

    Reads already-scraped JSONs (paths via env vars),
    builds TextNodes, (optionally) filters already-ingested content via Qdrant,
    embeds, then upserts to Qdrant.

    Returns:
      {
        ok: bool,
        nodes: int,               # nodes considered for ingestion (after filtering)
        upserted: int,            # points actually upserted
        per_source: {vendor: int, ...},
        collection: str,
        message?: str
      }
    """
    _ = args  # unused

    qdrant_collection: Optional[str] = kwargs.get("qdrant_collection")
    collection = _get_collection_name(qdrant_collection)

    check_qdrant: bool = bool(kwargs.get("check_qdrant", False))
    verbose: bool = bool(kwargs.get("verbose", True))

    concurrency: int = int(kwargs.get("concurrency", 2))
    embed_batch_size: int = int(kwargs.get("embed_batch_size", 20))
    upsert_batch_size: int = int(kwargs.get("upsert_batch_size", 64))

    def _say(msg: str) -> None:
        logger.info(msg)
        if verbose:
            print(msg, flush=True)

    _say(f"[INGEST] Starting ingest_all_sources(check_qdrant={check_qdrant}) into collection='{collection}'")

    client = _get_qdrant_client_safe() if check_qdrant else None
    if check_qdrant and client is None:
        _say("[INGEST] check_qdrant=True but QDRANT_URL not set or client init failed -> skipping DB checks.")

    per_source: Dict[str, int] = {}
    all_nodes: List[TextNode] = []

    # -------------------------
    # 1) Vendor JSONs -> TextNodes (filter via vendor+url)
    # -------------------------
    for vendor, path in _vendor_inputs_from_env():
        try:
            records = _load_records_list(path)

            # Canonicalize URL inside records so payload + checks match
            canon_records: List[Dict[str, Any]] = []
            skipped_existing = 0

            for rec in records:
                if not isinstance(rec, dict):
                    continue
                r2 = dict(rec)
                url = r2.get("url")
                if isinstance(url, str) and url.strip():
                    cu = canonical_url(url)
                    r2["url"] = cu

                    if client is not None:
                        if _qdrant_has_vendor_url(client, collection, vendor, cu):
                            skipped_existing += 1
                            continue

                canon_records.append(r2)

            nodes = build_vendor_nodes(canon_records, vendor=vendor)
            all_nodes.extend(nodes)
            per_source[vendor] = len(nodes)

            _say(
                f"[INGEST] vendor={vendor} file='{path}' "
                f"records={len(records)} kept={len(canon_records)} skipped_existing={skipped_existing} nodes={len(nodes)}"
            )
        except Exception:
            per_source[vendor] = 0
            logger.exception("Failed loading/processing vendor=%s from %s", vendor, path)
            if verbose:
                print(f"[INGEST] ERROR vendor={vendor} from '{path}' (see stacktrace above)", flush=True)

    # -------------------------
    # 2) MITRE bundle -> TextNodes (filter via stix_id when possible)
    # -------------------------
    mitre_path = os.getenv("MITRE_JSON_PATH", "").strip()
    if mitre_path:
        try:
            df = _load_mitre_df(mitre_path)
            content_nodes, relationship_nodes = build_mitre_nodes(df)

            mitre_nodes_all = list(content_nodes) + list(relationship_nodes)

            if client is not None:
                kept: List[TextNode] = []
                skipped = 0

                for n in mitre_nodes_all:
                    # Prefer stix_id check; fallback: no check (keep)
                    stix = ""
                    # Many MITRE nodes use stix--... as id_
                    nid = str(getattr(n, "id_", "") or "")
                    if nid.lower().startswith("stix--"):
                        stix = nid
                    else:
                        stix = _extract_stix_from_text(getattr(n, "text", "") or "")

                    if stix:
                        if _qdrant_has_mitre_stix(client, collection, stix):
                            skipped += 1
                            continue

                    kept.append(n)

                mitre_nodes_new = kept
                _say(
                    f"[INGEST] mitre file='{mitre_path}' "
                    f"nodes_all={len(mitre_nodes_all)} kept={len(mitre_nodes_new)} skipped_existing={skipped}"
                )
            else:
                mitre_nodes_new = mitre_nodes_all
                _say(
                    f"[INGEST] mitre file='{mitre_path}' nodes_all={len(mitre_nodes_all)} (no DB check)"
                )

            all_nodes.extend(mitre_nodes_new)
            per_source[MITRE_SOURCE_KEY] = len(mitre_nodes_new)

        except Exception:
            per_source[MITRE_SOURCE_KEY] = 0
            logger.exception("Failed loading MITRE from %s", mitre_path)
            if verbose:
                print(f"[INGEST] ERROR mitre from '{mitre_path}' (see stacktrace above)", flush=True)
    else:
        per_source[MITRE_SOURCE_KEY] = 0
        _say("[INGEST] MITRE_JSON_PATH not set -> skipping MITRE")

    if not all_nodes:
        _say("[INGEST] No new nodes to ingest (after filtering).")
        return {
            "ok": True,
            "nodes": 0,
            "upserted": 0,
            "per_source": per_source,
            "collection": collection,
            "message": "No nodes to ingest.",
        }

    _say(f"[INGEST] Embedding {len(all_nodes)} nodes (concurrency={concurrency}, batch={embed_batch_size}) ...")
    embeddings = await embed_nodes_hybrid(
        all_nodes,
        concurrency=concurrency,
        batch_size=embed_batch_size,
    )

    _say(f"[INGEST] Upserting to Qdrant (batch={upsert_batch_size}) ...")
    upserted = upsert_nodes_to_qdrant(
        all_nodes,
        embeddings=embeddings,
        collection_name=collection,
        batch_size=upsert_batch_size,
    )

    _say(f"[INGEST] Done. upserted={int(upserted)} nodes={len(all_nodes)} per_source={per_source}")

    return {
        "ok": True,
        "nodes": len(all_nodes),
        "upserted": int(upserted),
        "per_source": per_source,
        "collection": collection,
    }
