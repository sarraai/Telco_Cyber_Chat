# telco_cyber_chat/webscraping/scraper_graph.py

import operator
from typing import Optional, Annotated, Dict
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

from telco_cyber_chat.webscraping.ingest_pipeline import ingest_all_sources


class ScraperState(TypedDict, total=False):
    # Accumulate status messages from parallel nodes
    status: Annotated[list[str], operator.add]

    # Ingestion summary
    inserted: Optional[int]
    per_source: Dict[str, int]
    collection: Optional[str]
    ok: bool
    error_message: Optional[str]


# ---------------------- SINGLE INGESTION NODE ----------------------

async def run_ingestion_node(state: ScraperState) -> ScraperState:
    """
    Runs the COMPLETE pipeline:
    - Calls ALL scrapers (cisco, nokia, ericsson, huawei, variot, mitre)
    - Each scraper dedupes against Qdrant internally
    - Embeds all new nodes
    - Upserts to Qdrant
    
    Returns summary with per-source counts.
    """
    try:
        summary = await ingest_all_sources(
            check_qdrant=True,       # Enable Qdrant deduplication
            embed_batch_size=32,
            upsert_batch_size=64,
            verbose=True,
        )
    except Exception as e:
        return {
            "status": [f"❌ Ingestion failed: {e}"],
            "inserted": 0,
            "per_source": {},
            "ok": False,
            "error_message": str(e),
        }

    if not isinstance(summary, dict):
        return {
            "status": ["❌ Unexpected response from ingest_all_sources"],
            "inserted": 0,
            "per_source": {},
            "ok": False,
        }

    per_source = dict(summary.get("per_source", {}) or {})
    upserted = int(summary.get("upserted", 0) or 0)
    collection = summary.get("collection", "Telco_CyberChat")

    # Build status message
    status_lines = [
        f"✅ Ingestion completed: {upserted} nodes upserted to '{collection}'",
        f"📊 Per-source breakdown:",
    ]
    
    for vendor, count in per_source.items():
        status_lines.append(f"  • {vendor}: {count} new nodes")

    return {
        "status": status_lines,
        "inserted": upserted,
        "per_source": per_source,
        "collection": collection,
        "ok": True,
    }


# ---------------------- BUILD GRAPH ----------------------

graph_builder = StateGraph(ScraperState)

# Single node that does everything
graph_builder.add_node("run_ingestion", run_ingestion_node)

# Simple linear flow
graph_builder.add_edge(START, "run_ingestion")
graph_builder.add_edge("run_ingestion", END)

graph = graph_builder.compile()
