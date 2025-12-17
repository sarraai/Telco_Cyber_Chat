"""
ingest_pipeline.py - FIXED VERSION

MUST expose: ingest_all_sources

This version:
- ACTUALLY CALLS the scraper functions (cisco, nokia, etc.)
- Dedupes against Qdrant BEFORE scraping (via check_qdrant=True)
- Embeds and upserts new nodes only
"""

from __future__ import annotations

import os
import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

MITRE_SOURCE_KEY = "mitre_mobile"


def _get_collection_name(explicit: Optional[str]) -> str:
    return (explicit or os.getenv("QDRANT_COLLECTION", "Telco_CyberChat")).strip()


async def ingest_all_sources(*args, **kwargs) -> Dict[str, Any]:
    """
    FIXED VERSION:
    - Calls each scraper function directly
    - Scrapers handle Qdrant deduplication internally via check_qdrant=True
    - Embeds and upserts only NEW nodes
    """
    _ = args  # unused

    # Lazy imports (avoid slow startup)
    from telco_cyber_chat.webscraping.node_embed import embed_nodes_hybrid
    from telco_cyber_chat.webscraping.qdrant_ingest import upsert_nodes_to_qdrant

    # Import scraper functions
    from telco_cyber_chat.webscraping.cisco_scraper import scrape_cisco
    from telco_cyber_chat.webscraping.nokia_scraper import scrape_nokia
    from telco_cyber_chat.webscraping.ericsson_scraper import scrape_ericsson
    from telco_cyber_chat.webscraping.huawei_scraper import scrape_huawei_nodes
    from telco_cyber_chat.webscraping.variot_scraper import scrape_variot_nodes
    from telco_cyber_chat.webscraping.mitre_attack_scraper import scrape_mitre_mobile

    qdrant_collection: Optional[str] = kwargs.get("qdrant_collection")
    collection = _get_collection_name(qdrant_collection)

    check_qdrant: bool = bool(kwargs.get("check_qdrant", True))  # Default TRUE
    verbose: bool = bool(kwargs.get("verbose", True))

    concurrency: int = int(kwargs.get("concurrency", 2))
    embed_batch_size: int = int(kwargs.get("embed_batch_size", 20))
    upsert_batch_size: int = int(kwargs.get("upsert_batch_size", 64))

    def _say(msg: str) -> None:
        logger.info(msg)
        if verbose:
            print(msg, flush=True)

    _say(f"[INGEST] Starting ingest_all_sources(check_qdrant={check_qdrant}) into collection='{collection}'")

    per_source: Dict[str, int] = {}
    all_nodes: List[Any] = []

    # ========== CISCO ==========
    try:
        _say("[INGEST] Scraping Cisco...")
        cisco_nodes = scrape_cisco(check_qdrant=check_qdrant)
        all_nodes.extend(cisco_nodes)
        per_source["cisco"] = len(cisco_nodes)
        _say(f"[INGEST] Cisco: {len(cisco_nodes)} new nodes")
    except Exception as e:
        per_source["cisco"] = 0
        logger.exception("Failed scraping Cisco")
        if verbose:
            print(f"[INGEST] ERROR Cisco: {e}", flush=True)

    # ========== NOKIA ==========
    try:
        _say("[INGEST] Scraping Nokia...")
        nokia_nodes = scrape_nokia(check_qdrant=check_qdrant)
        all_nodes.extend(nokia_nodes)
        per_source["nokia"] = len(nokia_nodes)
        _say(f"[INGEST] Nokia: {len(nokia_nodes)} new nodes")
    except Exception as e:
        per_source["nokia"] = 0
        logger.exception("Failed scraping Nokia")
        if verbose:
            print(f"[INGEST] ERROR Nokia: {e}", flush=True)

    # ========== ERICSSON ==========
    try:
        _say("[INGEST] Scraping Ericsson...")
        ericsson_nodes = scrape_ericsson(check_qdrant=check_qdrant)
        all_nodes.extend(ericsson_nodes)
        per_source["ericsson"] = len(ericsson_nodes)
        _say(f"[INGEST] Ericsson: {len(ericsson_nodes)} new nodes")
    except Exception as e:
        per_source["ericsson"] = 0
        logger.exception("Failed scraping Ericsson")
        if verbose:
            print(f"[INGEST] ERROR Ericsson: {e}", flush=True)

    # ========== HUAWEI ==========
    try:
        _say("[INGEST] Scraping Huawei...")
        huawei_nodes = scrape_huawei_nodes(check_qdrant=check_qdrant)
        all_nodes.extend(huawei_nodes)
        per_source["huawei"] = len(huawei_nodes)
        _say(f"[INGEST] Huawei: {len(huawei_nodes)} new nodes")
    except Exception as e:
        per_source["huawei"] = 0
        logger.exception("Failed scraping Huawei")
        if verbose:
            print(f"[INGEST] ERROR Huawei: {e}", flush=True)

    # ========== VARIOT ==========
    try:
        _say("[INGEST] Scraping VARIoT...")
        variot_result = scrape_variot_nodes(check_qdrant=check_qdrant)
        variot_nodes = variot_result.get("nodes", [])
        all_nodes.extend(variot_nodes)
        per_source["variot"] = len(variot_nodes)
        _say(f"[INGEST] VARIoT: {len(variot_nodes)} new nodes")
    except Exception as e:
        per_source["variot"] = 0
        logger.exception("Failed scraping VARIoT")
        if verbose:
            print(f"[INGEST] ERROR VARIoT: {e}", flush=True)

    # ========== MITRE ==========
    mitre_enabled = os.getenv("ENABLE_MITRE_SCRAPING", "true").lower() == "true"
    if mitre_enabled:
        try:
            _say("[INGEST] Scraping MITRE Mobile ATT&CK...")
            mitre_nodes = scrape_mitre_mobile(check_qdrant=check_qdrant)
            all_nodes.extend(mitre_nodes)
            per_source[MITRE_SOURCE_KEY] = len(mitre_nodes)
            _say(f"[INGEST] MITRE: {len(mitre_nodes)} new nodes")
        except Exception as e:
            per_source[MITRE_SOURCE_KEY] = 0
            logger.exception("Failed scraping MITRE")
            if verbose:
                print(f"[INGEST] ERROR MITRE: {e}", flush=True)
    else:
        per_source[MITRE_SOURCE_KEY] = 0
        _say("[INGEST] MITRE scraping disabled (set ENABLE_MITRE_SCRAPING=true to enable)")

    if not all_nodes:
        _say("[INGEST] No new nodes to ingest (after filtering).")
        return {
            "ok": True,
            "nodes": 0,
            "upserted": 0,
            "per_source": per_source,
            "collection": collection,
            "message": "No new nodes found across all sources.",
        }

    _say(f"[INGEST] Embedding {len(all_nodes)} nodes (concurrency={concurrency}, batch={embed_batch_size})...")
    embeddings = await embed_nodes_hybrid(all_nodes, concurrency=concurrency, batch_size=embed_batch_size)

    _say(f"[INGEST] Upserting to Qdrant (batch={upsert_batch_size})...")
    upserted = upsert_nodes_to_qdrant(
        all_nodes,
        embeddings=embeddings,
        collection_name=collection,
        batch_size=upsert_batch_size,
    )

    _say(f"[INGEST] ✅ Done. upserted={int(upserted)} nodes={len(all_nodes)} per_source={per_source}")
    return {
        "ok": True,
        "nodes": len(all_nodes),
        "upserted": int(upserted),
        "per_source": per_source,
        "collection": collection,
    }
