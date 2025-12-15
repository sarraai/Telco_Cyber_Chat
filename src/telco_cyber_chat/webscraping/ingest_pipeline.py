"""
ingest_pipeline.py

High-level orchestration of the vendor + VARIoT + MITRE ingestion pipeline.

NEW STRATEGY:
- For ALL sources: build TextNodes where:
    node.text     = ALL fields except "url" (key/value readable text)
    node.metadata = {"url": "<canonical_url>", "vendor": "<source>"}  # vendor kept for filtering

- For MITRE:
    - Non-relationship objects -> TextNodes (vendor="mitre", stix_id set)
    - Relationship objects     -> RelationshipNodes (vendor="mitre", stix_id set)

- Embed BOTH TextNodes + RelationshipNodes using node_embedder
- Upsert BOTH into Qdrant using qdrant_ingest
"""

from __future__ import annotations

import asyncio
import logging
from typing import Dict, List, Any, Tuple, Union

from llama_index.core.schema import TextNode

# ---- Scrapers ----
from telco_cyber_chat.webscraping.nokia_scraper import scrape_nokia
from telco_cyber_chat.webscraping.cisco_scraper import scrape_cisco
from telco_cyber_chat.webscraping.ericsson_scraper import scrape_ericsson
from telco_cyber_chat.webscraping.huawei_scraper import scrape_huawei
from telco_cyber_chat.webscraping.variot_scraper import scrape_variot

# MITRE scraper should return a DataFrame OR list of raw STIX dicts.
from telco_cyber_chat.webscraping.mitre_attack_scraper import scrape_mitre_mobile

# ---- Node building ----
from telco_cyber_chat.webscraping.node_builder import (
    build_vendor_nodes,
    build_mitre_nodes,          # expects df or list[dict]
    RelationshipNode,           # dataclass with id_, text, metadata
)

# ---- Embedding ----
from telco_cyber_chat.webscraping.node_embedder import embed_nodes_hybrid

# ---- Qdrant upsert ----
from telco_cyber_chat.webscraping.qdrant_ingest import upsert_nodes_to_qdrant


logger = logging.getLogger(__name__)

ScrapedRecord = Dict[str, Any]
Summary = Dict[str, Any]
EmbeddableNode = Union[TextNode, RelationshipNode]


# =============================================================================
# Helper: run scrapers with optional "check_qdrant" flag
# =============================================================================
def _run_scraper(name: str, fn, check_qdrant: bool = True):
    """
    Run one scraper safely.

    We try calling with check_qdrant=<bool>. If not supported, call without args.
    """
    logger.info("Running scraper: %s (check_qdrant=%s)", name, check_qdrant)
    try:
        try:
            out = fn(check_qdrant=check_qdrant)
        except TypeError:
            out = fn()

        return out
    except Exception as exc:
        logger.exception("Scraper %s failed: %s", name, exc)
        return None


def _ensure_vendor_field(records: List[ScrapedRecord], vendor: str) -> List[ScrapedRecord]:
    """
    Ensure each record has vendor=<vendor>.
    (You still keep vendor inside node.text; vendor is also needed for Qdrant filtering.)
    """
    out: List[ScrapedRecord] = []
    for r in records or []:
        if not isinstance(r, dict):
            continue
        rr = dict(r)
        rr["vendor"] = vendor
        out.append(rr)
    return out


# =============================================================================
# Main orchestration
# =============================================================================
async def ingest_all_sources(
    check_qdrant: bool = True,
    embed_batch_size: int = 32,
) -> Summary:
    """
    Pipeline:
      1) Scrape all sources
      2) Build TextNodes + RelationshipNodes
      3) Embed them (dense+sparse)
      4) Upsert into Qdrant
    """
    logger.info("=== [1/4] Scraping all sources ===")

    vendor_outputs: Dict[str, List[ScrapedRecord]] = {}

    # ---- Vendor sources (list[dict]) ----
    for vendor, fn in [
        ("cisco", scrape_cisco),
        ("nokia", scrape_nokia),
        ("ericsson", scrape_ericsson),
        ("huawei", scrape_huawei),
        ("variot", scrape_variot),
    ]:
        raw = _run_scraper(vendor, fn, check_qdrant=check_qdrant)
        if isinstance(raw, list):
            vendor_outputs[vendor] = _ensure_vendor_field(raw, vendor)
        else:
            vendor_outputs[vendor] = []

    # ---- MITRE source (DataFrame or list[dict]) ----
    mitre_raw = _run_scraper("mitre", scrape_mitre_mobile, check_qdrant=check_qdrant)

    per_source_counts = {k: len(v) for k, v in vendor_outputs.items()}

    # -------------------------------------------------------------------------
    # 2) Build nodes
    # -------------------------------------------------------------------------
    logger.info("=== [2/4] Building nodes (TextNodes + RelationshipNodes) ===")

    all_nodes: List[EmbeddableNode] = []

    # Vendor nodes
    for vendor, records in vendor_outputs.items():
        if not records:
            continue
        try:
            nodes = build_vendor_nodes(records, vendor=vendor)
            all_nodes.extend(nodes)
            logger.info("Built %d TextNodes for vendor=%s", len(nodes), vendor)
        except Exception as exc:
            logger.exception("Failed building nodes for %s: %s", vendor, exc)

    # MITRE nodes (content + relationships)
    mitre_content_nodes: List[TextNode] = []
    mitre_relationship_nodes: List[RelationshipNode] = []

    try:
        if mitre_raw is not None:
            mitre_content_nodes, mitre_relationship_nodes = build_mitre_nodes(mitre_raw)
            all_nodes.extend(mitre_content_nodes)
            all_nodes.extend(mitre_relationship_nodes)

        per_source_counts["mitre_content"] = len(mitre_content_nodes)
        per_source_counts["mitre_relationships"] = len(mitre_relationship_nodes)

        logger.info(
            "Built MITRE nodes: %d content + %d relationships",
            len(mitre_content_nodes),
            len(mitre_relationship_nodes),
        )
    except Exception as exc:
        logger.exception("Failed building MITRE nodes: %s", exc)
        per_source_counts["mitre_content"] = 0
        per_source_counts["mitre_relationships"] = 0

    total_nodes = len(all_nodes)
    if total_nodes == 0:
        logger.warning("No nodes built. Aborting ingestion.")
        return {
            "per_source": per_source_counts,
            "total_nodes": 0,
            "total_embedded": 0,
            "upserted": 0,
        }

    # -------------------------------------------------------------------------
    # 3) Embed nodes (works for BOTH TextNode + RelationshipNode)
    # -------------------------------------------------------------------------
    logger.info("=== [3/4] Embedding nodes via remote BGE ===")

    embeddings = await embed_nodes_hybrid(
        all_nodes,               # <- supports relationship nodes now
        batch_size=embed_batch_size,
        concurrency=2,
    )

    if not embeddings:
        logger.warning("Embedding returned no results. Aborting ingestion.")
        return {
            "per_source": per_source_counts,
            "total_nodes": total_nodes,
            "total_embedded": 0,
            "upserted": 0,
        }

    # -------------------------------------------------------------------------
    # 4) Upsert into Qdrant
    # -------------------------------------------------------------------------
    logger.info("=== [4/4] Upserting into Qdrant ===")

    upserted = upsert_nodes_to_qdrant(all_nodes, embeddings)

    logger.info("Done. Upserted %d points.", upserted)

    return {
        "per_source": per_source_counts,
        "total_nodes": total_nodes,
        "total_embedded": len(embeddings),
        "upserted": upserted,
    }


# =============================================================================
# CLI entrypoint
# =============================================================================
def _configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> None:
    _configure_logging()
    summary = asyncio.run(ingest_all_sources(check_qdrant=True, embed_batch_size=32))
    logger.info("Ingestion summary: %s", summary)


if __name__ == "__main__":
    main()
