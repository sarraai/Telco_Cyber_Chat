"""
ingest_pipeline.py

Exposes BOTH:
1) ingest_all_sources()              -> full end-to-end (scrape + embed + upsert)
2) scrape_all_sources_only()         -> Stage 1
3) embed_nodes_only()                -> Stage 3
4) upsert_nodes_only()               -> Stage 4

Notes:
- Scrapers dedupe internally via check_qdrant=True
- We avoid logger.exception() to prevent ERROR-looking logs in LangSmith.
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger("telco_cyber_chat.webscraping.ingest")

MITRE_SOURCE_KEY = "mitre_mobile"


def _get_collection_name(explicit: Optional[str]) -> str:
    return (explicit or os.getenv("QDRANT_COLLECTION", "Telco_CyberChat")).strip()


def _verbose_print(verbose: bool, msg: str) -> None:
    # Keep printing for LangSmith logs visibility
    if verbose:
        print(msg, flush=True)


def _say(verbose: bool, msg: str) -> None:
    logger.info(msg)
    _verbose_print(verbose, msg)


def _warn(verbose: bool, msg: str, exc: Optional[BaseException] = None) -> None:
    # WARNING level (not ERROR) so it won't look like a failure banner.
    if exc is not None:
        logger.warning(msg, exc_info=True)
    else:
        logger.warning(msg)
    _verbose_print(verbose, msg)


# ------------------------- STAGE 1: SCRAPE ONLY -------------------------

async def scrape_all_sources_only(
    *,
    check_qdrant: bool = True,
    verbose: bool = True,
    qdrant_collection: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Calls ALL scrapers and returns NEW TextNodes (scrapers dedupe internally).

    Returns:
      {
        "ok": bool,
        "nodes": List[TextNode],
        "per_source": { vendor: int },
        "collection": str
      }
    """
    collection = _get_collection_name(qdrant_collection)

    # Lazy imports to reduce startup/import time
    from telco_cyber_chat.webscraping.cisco_scraper import scrape_cisco
    from telco_cyber_chat.webscraping.nokia_scraper import scrape_nokia
    from telco_cyber_chat.webscraping.ericsson_scraper import scrape_ericsson
    from telco_cyber_chat.webscraping.huawei_scraper import scrape_huawei_nodes
    from telco_cyber_chat.webscraping.variot_scraper import scrape_variot_nodes
    from telco_cyber_chat.webscraping.mitre_attack_scraper import scrape_mitre_mobile

    _say(verbose, f"[INGEST] 🟣 Stage 1: scraping (check_qdrant={check_qdrant}) → collection='{collection}'")

    per_source: Dict[str, int] = {}
    all_nodes: List[Any] = []

    # ---- Cisco ----
    try:
        _say(verbose, "[INGEST] Cisco: scraping…")
        nodes = scrape_cisco(check_qdrant=check_qdrant)
        all_nodes.extend(nodes)
        per_source["cisco"] = len(nodes)
        _say(verbose, f"[INGEST] Cisco: {len(nodes)} new nodes")
    except Exception as e:
        per_source["cisco"] = 0
        _warn(verbose, f"[INGEST] ⚠ Cisco scraper skipped: {e}", exc=e)

    # ---- Nokia ----
    try:
        _say(verbose, "[INGEST] Nokia: scraping…")
        nodes = scrape_nokia(check_qdrant=check_qdrant)
        all_nodes.extend(nodes)
        per_source["nokia"] = len(nodes)
        _say(verbose, f"[INGEST] Nokia: {len(nodes)} new nodes")
    except Exception as e:
        per_source["nokia"] = 0
        _warn(verbose, f"[INGEST] ⚠ Nokia scraper skipped: {e}", exc=e)

    # ---- Ericsson ----
    try:
        _say(verbose, "[INGEST] Ericsson: scraping…")
        nodes = scrape_ericsson(check_qdrant=check_qdrant)
        all_nodes.extend(nodes)
        per_source["ericsson"] = len(nodes)
        _say(verbose, f"[INGEST] Ericsson: {len(nodes)} new nodes")
    except Exception as e:
        per_source["ericsson"] = 0
        _warn(verbose, f"[INGEST] ⚠ Ericsson scraper skipped: {e}", exc=e)

    # ---- Huawei ----
    try:
        _say(verbose, "[INGEST] Huawei: scraping…")
        nodes = scrape_huawei_nodes(check_qdrant=check_qdrant)
        all_nodes.extend(nodes)
        per_source["huawei"] = len(nodes)
        _say(verbose, f"[INGEST] Huawei: {len(nodes)} new nodes")
    except Exception as e:
        per_source["huawei"] = 0
        _warn(verbose, f"[INGEST] ⚠ Huawei scraper skipped: {e}", exc=e)

    # ---- VARIoT ----
    try:
        _say(verbose, "[INGEST] VARIoT: scraping…")
        variot_result = scrape_variot_nodes(check_qdrant=check_qdrant)

        # Your variot scraper returns dict: {"nodes": [...], ...}
        variot_nodes = (variot_result or {}).get("nodes", []) or []
        all_nodes.extend(variot_nodes)
        per_source["variot"] = len(variot_nodes)
        _say(verbose, f"[INGEST] VARIoT: {len(variot_nodes)} new nodes")
    except Exception as e:
        per_source["variot"] = 0
        _warn(verbose, f"[INGEST] ⚠ VARIoT scraper skipped: {e}", exc=e)

    # ---- MITRE ----
    mitre_enabled = os.getenv("ENABLE_MITRE_SCRAPING", "true").lower() == "true"
    if mitre_enabled:
        try:
            _say(verbose, "[INGEST] MITRE Mobile ATT&CK: scraping…")
            nodes = scrape_mitre_mobile(check_qdrant=check_qdrant)
            all_nodes.extend(nodes)
            per_source[MITRE_SOURCE_KEY] = len(nodes)
            _say(verbose, f"[INGEST] MITRE: {len(nodes)} new nodes")
        except Exception as e:
            per_source[MITRE_SOURCE_KEY] = 0
            _warn(verbose, f"[INGEST] ⚠ MITRE scraper skipped: {e}", exc=e)
    else:
        per_source[MITRE_SOURCE_KEY] = 0
        _say(verbose, "[INGEST] MITRE scraping disabled (ENABLE_MITRE_SCRAPING=true to enable)")

    _say(verbose, f"[INGEST] 🟣 Stage 1 done: total_new_nodes={len(all_nodes)} per_source={per_source}")

    return {
        "ok": True,
        "nodes": all_nodes,
        "per_source": per_source,
        "collection": collection,
    }


# ------------------------- STAGE 3: EMBED ONLY -------------------------

async def embed_nodes_only(
    nodes: List[Any],
    *,
    embed_batch_size: int = 32,
    concurrency: int = 2,
    verbose: bool = True,
) -> Any:
    """
    Embeds nodes (hybrid dense+sparse) and returns embeddings object (whatever embedder returns).
    """
    if not nodes:
        _say(verbose, "[INGEST] 🟣 Stage 3: embedding skipped (0 nodes)")
        return None

    from telco_cyber_chat.webscraping.node_embed import embed_nodes_hybrid

    _say(verbose, f"[INGEST] 🟣 Stage 3: embedding {len(nodes)} nodes (concurrency={concurrency}, batch={embed_batch_size})…")
    embeddings = await embed_nodes_hybrid(nodes, concurrency=concurrency, batch_size=embed_batch_size)
    _say(verbose, f"[INGEST] 🟣 Stage 3 done: embedded={len(nodes)}")
    return embeddings


# ------------------------- STAGE 4: UPSERT ONLY -------------------------

async def upsert_nodes_only(
    *,
    nodes: List[Any],
    embeddings: Any,
    upsert_batch_size: int = 64,
    qdrant_collection: Optional[str] = None,
    verbose: bool = True,
) -> Dict[str, Any]:
    """
    Upserts nodes+embeddings into Qdrant.
    Returns: {"ok": True, "upserted": int, "collection": str}
    """
    collection = _get_collection_name(qdrant_collection)

    if not nodes:
        _say(verbose, "[INGEST] 🟣 Stage 4: upsert skipped (0 nodes)")
        return {"ok": True, "upserted": 0, "collection": collection}

    from telco_cyber_chat.webscraping.qdrant_ingest import upsert_nodes_to_qdrant

    _say(verbose, f"[INGEST] 🟣 Stage 4: upserting {len(nodes)} nodes (batch={upsert_batch_size}) → '{collection}'…")
    upserted = upsert_nodes_to_qdrant(
        nodes,
        embeddings=embeddings,
        collection_name=collection,
        batch_size=upsert_batch_size,
    )
    upserted_i = int(upserted or 0)
    _say(verbose, f"[INGEST] ✅ Stage 4 done: upserted={upserted_i} collection='{collection}'")

    return {"ok": True, "upserted": upserted_i, "collection": collection}


# ------------------------- FULL PIPELINE (BACKCOMPAT) -------------------------

async def ingest_all_sources(*args, **kwargs) -> Dict[str, Any]:
    """
    Backward-compatible: does scrape + embed + upsert in one call.

    Returns:
      {
        "ok": True,
        "nodes": int,
        "upserted": int,
        "per_source": {vendor: int},
        "collection": str,
      }
    """
    _ = args  # unused

    check_qdrant: bool = bool(kwargs.get("check_qdrant", True))
    verbose: bool = bool(kwargs.get("verbose", True))
    concurrency: int = int(kwargs.get("concurrency", 2))
    embed_batch_size: int = int(kwargs.get("embed_batch_size", 32))
    upsert_batch_size: int = int(kwargs.get("upsert_batch_size", 64))
    qdrant_collection: Optional[str] = kwargs.get("qdrant_collection")

    stage1 = await scrape_all_sources_only(
        check_qdrant=check_qdrant,
        verbose=verbose,
        qdrant_collection=qdrant_collection,
    )

    nodes = stage1.get("nodes", []) or []
    per_source = dict(stage1.get("per_source", {}) or {})
    collection = stage1.get("collection")

    if not nodes:
        _say(verbose, "[INGEST] ✅ Done: 0 new nodes (nothing to embed/upsert).")
        return {
            "ok": True,
            "nodes": 0,
            "upserted": 0,
            "per_source": per_source,
            "collection": collection,
            "message": "No new nodes found across all sources.",
        }

    embeddings = await embed_nodes_only(
        nodes,
        embed_batch_size=embed_batch_size,
        concurrency=concurrency,
        verbose=verbose,
    )

    upsert_res = await upsert_nodes_only(
        nodes=nodes,
        embeddings=embeddings,
        upsert_batch_size=upsert_batch_size,
        qdrant_collection=collection,
        verbose=verbose,
    )

    upserted = int(upsert_res.get("upserted", 0) or 0)

    _say(verbose, f"[INGEST] ✅ Done. upserted={upserted} nodes={len(nodes)} per_source={per_source}")
    return {
        "ok": True,
        "nodes": len(nodes),
        "upserted": upserted,
        "per_source": per_source,
        "collection": collection,
    }
