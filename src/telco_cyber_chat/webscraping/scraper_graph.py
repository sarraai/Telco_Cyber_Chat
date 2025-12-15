import operator
from typing import Optional, Annotated, Dict, Any
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

from telco_cyber_chat.webscraping.ingest_pipeline import ingest_all_sources


class ScraperState(TypedDict, total=False):
    # Accumulate status messages from parallel nodes
    status: Annotated[list[str], operator.add]

    # Ingestion summary
    inserted: Optional[int]
    per_source: Dict[str, int]

    # Per-source flags and counts
    cisco_done: bool
    cisco_scraped: int

    nokia_done: bool
    nokia_scraped: int

    ericsson_done: bool
    ericsson_scraped: int

    huawei_done: bool
    huawei_scraped: int

    variot_done: bool
    variot_scraped: int

    mitre_mobile_done: bool
    mitre_mobile_scraped: int


# ---------------------- INGESTION NODE (RUN ONCE) ----------------------

async def run_ingestion_node(state: ScraperState) -> ScraperState:
    """
    Runs the full pipeline once (scrape → build nodes → embed → upsert).
    Stores per-source counts in state so vendor nodes can report them.
    """
    try:
        summary = await ingest_all_sources(check_qdrant=True, batch_size=32)
    except Exception as e:
        return {
            "status": [f"ingestion_failed: {e}"],
            "inserted": None,
            "per_source": {},
        }

    per_source = {}
    inserted = None

    if isinstance(summary, dict):
        per_source = dict(summary.get("per_source", {}) or {})
        try:
            inserted = int(summary.get("upserted", 0) or 0)
        except Exception:
            inserted = None

    return {
        "status": [f"ingestion_completed (upserted: {inserted})"],
        "inserted": inserted,
        "per_source": per_source,
    }


# ---------------------- VENDOR REPORT NODES ----------------------

def _count(state: ScraperState, vendor: str) -> int:
    return int((state.get("per_source") or {}).get(vendor, 0) or 0)


def scrape_cisco_node(state: ScraperState) -> ScraperState:
    n = _count(state, "cisco")
    return {"status": [f"cisco_new: {n}"], "cisco_done": True, "cisco_scraped": n}


def scrape_nokia_node(state: ScraperState) -> ScraperState:
    n = _count(state, "nokia")
    return {"status": [f"nokia_new: {n}"], "nokia_done": True, "nokia_scraped": n}


def scrape_ericsson_node(state: ScraperState) -> ScraperState:
    n = _count(state, "ericsson")
    return {"status": [f"ericsson_new: {n}"], "ericsson_done": True, "ericsson_scraped": n}


def scrape_huawei_node(state: ScraperState) -> ScraperState:
    n = _count(state, "huawei")
    return {"status": [f"huawei_new: {n}"], "huawei_done": True, "huawei_scraped": n}


def scrape_variot_node(state: ScraperState) -> ScraperState:
    n = _count(state, "variot")
    return {"status": [f"variot_new: {n}"], "variot_done": True, "variot_scraped": n}


def scrape_mitre_mobile_node(state: ScraperState) -> ScraperState:
    # IMPORTANT: must match the key used by ingest_pipeline summary ("mitre_mobile")
    n = _count(state, "mitre_mobile")
    return {"status": [f"mitre_mobile_new: {n}"], "mitre_mobile_done": True, "mitre_mobile_scraped": n}


# ---------------------- AGGREGATION NODE ----------------------

def aggregate_vendors_node(state: ScraperState) -> ScraperState:
    total_scraped = (
        state.get("cisco_scraped", 0)
        + state.get("nokia_scraped", 0)
        + state.get("ericsson_scraped", 0)
        + state.get("huawei_scraped", 0)
        + state.get("variot_scraped", 0)
        + state.get("mitre_mobile_scraped", 0)
    )
    return {"status": [f"all_sources_reported (total_new: {total_scraped})"]}


# ---------------------- BUILD GRAPH ----------------------

graph_builder = StateGraph(ScraperState)

# Run ingestion once first
graph_builder.add_node("run_ingestion", run_ingestion_node)

# Vendor reporting nodes (parallel, after ingestion)
graph_builder.add_node("scrape_cisco", scrape_cisco_node)
graph_builder.add_node("scrape_nokia", scrape_nokia_node)
graph_builder.add_node("scrape_ericsson", scrape_ericsson_node)
graph_builder.add_node("scrape_huawei", scrape_huawei_node)
graph_builder.add_node("scrape_variot", scrape_variot_node)
graph_builder.add_node("scrape_mitre_mobile", scrape_mitre_mobile_node)

graph_builder.add_node("aggregate_vendors", aggregate_vendors_node)

# Edges
graph_builder.add_edge(START, "run_ingestion")

graph_builder.add_edge("run_ingestion", "scrape_cisco")
graph_builder.add_edge("run_ingestion", "scrape_nokia")
graph_builder.add_edge("run_ingestion", "scrape_ericsson")
graph_builder.add_edge("run_ingestion", "scrape_huawei")
graph_builder.add_edge("run_ingestion", "scrape_variot")
graph_builder.add_edge("run_ingestion", "scrape_mitre_mobile")

graph_builder.add_edge("scrape_cisco", "aggregate_vendors")
graph_builder.add_edge("scrape_nokia", "aggregate_vendors")
graph_builder.add_edge("scrape_ericsson", "aggregate_vendors")
graph_builder.add_edge("scrape_huawei", "aggregate_vendors")
graph_builder.add_edge("scrape_variot", "aggregate_vendors")
graph_builder.add_edge("scrape_mitre_mobile", "aggregate_vendors")

graph_builder.add_edge("aggregate_vendors", END)

graph = graph_builder.compile()
