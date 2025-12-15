"""
ingest_pipeline.py

High-level orchestration of the vendor + VARIoT + MITRE ingestion pipeline.

Steps:
  1. Run all scraper modules (Nokia, Cisco, Ericsson, Huawei, VARIoT, MITRE-Mobile).
     Each scraper returns a list of dicts: {"url", "title", "description"}.
     BEFORE scraping heavy content, each scraper is expected to skip
     URLs that already exist in Qdrant via `url_already_ingested(...)`.
  2. Convert scraped documents to LlamaIndex TextNodes via node_builder.
     - node.content = "title\n\ndescription"
     - node.metadata contains at least {"url", "source", "title"}.
  3. Embed nodes using the remote BGE endpoint through node_embedder.
  4. Upsert the embedded points into Qdrant via qdrant_ingest.
  5. Return a summary dict with per-source counts so the LangGraph scraper_graph
     can display "X new documents" per vendor node.

Run as a script:
    python -m telco_cyber_chat.webscraping.ingest_pipeline
"""

import asyncio
import logging
from typing import Dict, List, Any

from llama_index.core.schema import TextNode

# ---- Scrapers (each returns List[Dict[str, str]] with keys: url, title, description) ----
from telco_cyber_chat.webscraping.nokia_scraper import scrape_nokia
from telco_cyber_chat.webscraping.cisco_scraper import scrape_cisco
from telco_cyber_chat.webscraping.ericsson_scraper import scrape_ericsson
from telco_cyber_chat.webscraping.huawei_scraper import scrape_huawei
from telco_cyber_chat.webscraping.variot_scraper import scrape_variot
from telco_cyber_chat.webscraping.mitre_attack_scraper import scrape_mitre_mobile

# ---- Node building / embedding / Qdrant upsert helpers ----
from telco_cyber_chat.webscraping.node_builder import build_vendor_nodes
from telco_cyber_chat.webscraping.node_embed import embed_nodes_hybrid
from telco_cyber_chat.webscraping.qdrant_ingest import upsert_nodes_to_qdrant


logger = logging.getLogger(__name__)

ScrapedDoc = Dict[str, str]
Summary = Dict[str, Any]


# =============================================================================
# Helper: run scrapers with optional "check_qdrant" flag
# =============================================================================
def _run_scraper(
    name: str,
    fn,
    check_qdrant: bool = True,
) -> List[ScrapedDoc]:
    """
    Run one scraper safely.

    We *try* to call the scraper with check_qdrant=<bool>. If the scraper
    does not accept that keyword (TypeError), we call it without arguments.

    Each scraper is expected to:
      - Return a list of dicts with keys: "url", "title", "description".
      - Internally skip URLs that are already in Qdrant if check_qdrant=True.
    """
    logger.info("Running scraper: %s (check_qdrant=%s)", name, check_qdrant)
    try:
        # First try with keyword (for scrapers that support it).
        try:
            docs = fn(check_qdrant=check_qdrant)
        except TypeError:
            # Scraper does not support the keyword → call without it.
            logger.debug(
                "Scraper %s does not accept 'check_qdrant' keyword, "
                "calling without it.",
                name,
            )
            docs = fn()

        if not isinstance(docs, list):
            logger.warning(
                "Scraper %s returned non-list result (%r). Treating as empty.",
                name,
                type(docs),
            )
            return []

        logger.info("Scraper %s produced %d documents.", name, len(docs))
        return docs

    except Exception as exc:
        logger.exception("Scraper %s failed: %s", name, exc)
        return []


# =============================================================================
# Main orchestration
# =============================================================================
async def ingest_all_sources(
    check_qdrant: bool = True,
    batch_size: int = 32,
) -> Summary:
    """
    High-level pipeline:
      1. Scrape all sources (Nokia, Cisco, Ericsson, Huawei, VARIoT, MITRE-Mobile).
      2. Build TextNodes from the {url, title, description} records.
      3. Embed nodes using remote BGE via node_embedder.
      4. Upsert embeddings into Qdrant via qdrant_ingest.
      5. Return a summary dict with per-source counts and upserted points.

    Args:
        check_qdrant: If True, each scraper should skip URLs already in Qdrant.
        batch_size:   Batch size to use in node_embedder (if supported).

    Returns:
        {
          "per_source": { "cisco": int, "nokia": int, ... },
          "total_scraped_docs": int,
          "total_textnodes": int,
          "total_embedded_nodes": int,
          "upserted": int,
        }
    """
    # -------------------------------------------------------------------------
    # 1) Scrape all vendors / sources
    # -------------------------------------------------------------------------
    logger.info("=== [1/4] Scraping all sources ===")

    scraped: Dict[str, List[ScrapedDoc]] = {
        "cisco":        _run_scraper("cisco", scrape_cisco, check_qdrant=check_qdrant),
        "nokia":        _run_scraper("nokia", scrape_nokia, check_qdrant=check_qdrant),
        "ericsson":     _run_scraper("ericsson", scrape_ericsson, check_qdrant=check_qdrant),
        "huawei":       _run_scraper("huawei", scrape_huawei, check_qdrant=check_qdrant),
        "variot":       _run_scraper("variot", scrape_variot, check_qdrant=check_qdrant),
        "mitre_mobile": _run_scraper("mitre_mobile", scrape_mitre_mobile, check_qdrant=check_qdrant),
    }

    per_source_counts = {name: len(docs) for name, docs in scraped.items()}
    total_docs = sum(per_source_counts.values())

    if total_docs == 0:
        logger.warning("No documents scraped from any source. Aborting ingestion.")
        return {
            "per_source": per_source_counts,
            "total_scraped_docs": 0,
            "total_textnodes": 0,
            "total_embedded_nodes": 0,
            "upserted": 0,
        }

    logger.info("Total scraped documents across all sources: %d", total_docs)

    # -------------------------------------------------------------------------
    # 2) Build TextNodes (title + description as content, URL in metadata)
    # -------------------------------------------------------------------------
    logger.info("=== [2/4] Building TextNodes ===")

    all_nodes: List[TextNode] = []

    for source_name, docs in scraped.items():
        if not docs:
            continue

        try:
            nodes = build_vendor_nodes(docs, source=source_name)
        except Exception as exc:
            logger.exception(
                "Failed to build nodes for source '%s': %s", source_name, exc
            )
            continue

        logger.info(
            "Source '%s': %d documents → %d TextNodes",
            source_name,
            len(docs),
            len(nodes),
        )
        all_nodes.extend(nodes)

    if not all_nodes:
        logger.warning("No TextNodes built from scraped data. Aborting ingestion.")
        return {
            "per_source": per_source_counts,
            "total_scraped_docs": total_docs,
            "total_textnodes": 0,
            "total_embedded_nodes": 0,
            "upserted": 0,
        }

    logger.info("Total TextNodes to embed: %d", len(all_nodes))

    # -------------------------------------------------------------------------
    # 3) Embed nodes using remote BGE (dense + sparse)
    # -------------------------------------------------------------------------
    logger.info("=== [3/4] Embedding nodes via remote BGE ===")

    embeddings_dict = await embed_nodes_hybrid(all_nodes)

    if not embeddings_dict:
        logger.warning("Embedding step returned no results. Aborting ingestion.")
        return {
            "per_source": per_source_counts,
            "total_scraped_docs": total_docs,
            "total_textnodes": len(all_nodes),
            "total_embedded_nodes": 0,
            "upserted": 0,
        }

    logger.info("Embedded %d nodes.", len(embeddings_dict))

    # -------------------------------------------------------------------------
    # 4) Upsert embeddings into Qdrant
    # -------------------------------------------------------------------------
    logger.info("=== [4/4] Upserting into Qdrant ===")

    upserted = upsert_nodes_to_qdrant(all_nodes, embeddings_dict)

    logger.info("Qdrant upsert complete. Upserted %d points.", upserted)
    logger.info("Ingestion pipeline finished successfully.")

    return {
        "per_source": per_source_counts,
        "total_scraped_docs": total_docs,
        "total_textnodes": len(all_nodes),
        "total_embedded_nodes": len(embeddings_dict),
        "upserted": upserted,
    }


# =============================================================================
# CLI entrypoint
# =============================================================================
def _configure_logging() -> None:
    level_name = "INFO"
    logging.basicConfig(
        level=getattr(logging, level_name, logging.INFO),
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
    )


def main() -> None:
    """Entrypoint used when running this module as a script."""
    _configure_logging()
    logger.info("Starting full ingestion pipeline...")
    summary = asyncio.run(ingest_all_sources(check_qdrant=True, batch_size=32))
    logger.info("Ingestion summary: %s", summary)


if __name__ == "__main__":
    main()
