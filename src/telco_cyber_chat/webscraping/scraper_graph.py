import asyncio
from typing import Optional, Annotated
import operator

from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END

from telco_cyber_chat.webscraping.ingest_pipeline import ingest_all_sources


class ScraperState(TypedDict, total=False):
    """
    Minimal state for the scraper graph.

    We mainly track which logical step has been "reached" so that
    the graph in LangSmith is readable, plus an optional count of
    inserted points returned by ingest_all_sources.
    """
    # Use Annotated with operator.add to accumulate status messages from parallel nodes
    status: Annotated[list[str], operator.add]
    inserted: Optional[int]

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
    mitre_mobile_done: bool        # NEW
    mitre_mobile_scraped: int      # NEW

    # Pipeline phase flags
    textnodes_built: bool
    embedded: bool


# ---------------------- HELPER FUNCTION ----------------------

def get_scrape_count_for_source(source_name: str) -> int:
    """
    Get the count of newly scraped URLs for a specific source.
    This is a placeholder - you'll need to implement the actual logic
    to track scraped URLs per source.
    
    Options:
    1. Query Qdrant for documents with matching source added in last N minutes
    2. Track URLs during scraping in a temporary store
    3. Return count from your scraper functions
    """
    # TODO: Implement actual counting logic
    # For now, return 0 as placeholder
    return 0


# ---------------------- NODES: VENDORS ----------------------


def scrape_cisco_node(state: ScraperState) -> ScraperState:
    """
    Logical node for 'Cisco scraping step'.

    NOTE: The actual scraping + ingest is still orchestrated inside
    ingest_all_sources; this node is mainly here so that the graph
    shows a dedicated Cisco step.
    """
    scraped_count = get_scrape_count_for_source("cisco")
    
    return {
        "status": [f"cisco_scrape_step_reached (scraped: {scraped_count} new URLs)"],
        "cisco_done": True,
        "cisco_scraped": scraped_count,
    }


def scrape_nokia_node(state: ScraperState) -> ScraperState:
    """Logical Nokia scraping step."""
    scraped_count = get_scrape_count_for_source("nokia")
    
    return {
        "status": [f"nokia_scrape_step_reached (scraped: {scraped_count} new URLs)"],
        "nokia_done": True,
        "nokia_scraped": scraped_count,
    }


def scrape_ericsson_node(state: ScraperState) -> ScraperState:
    """Logical Ericsson scraping step."""
    scraped_count = get_scrape_count_for_source("ericsson")
    
    return {
        "status": [f"ericsson_scrape_step_reached (scraped: {scraped_count} new URLs)"],
        "ericsson_done": True,
        "ericsson_scraped": scraped_count,
    }


def scrape_huawei_node(state: ScraperState) -> ScraperState:
    """Logical Huawei scraping step."""
    scraped_count = get_scrape_count_for_source("huawei")
    
    return {
        "status": [f"huawei_scrape_step_reached (scraped: {scraped_count} new URLs)"],
        "huawei_done": True,
        "huawei_scraped": scraped_count,
    }


def scrape_variot_node(state: ScraperState) -> ScraperState:
    """Logical VARIoT scraping step."""
    scraped_count = get_scrape_count_for_source("variot")
    
    return {
        "status": [f"variot_scrape_step_reached (scraped: {scraped_count} new URLs)"],
        "variot_done": True,
        "variot_scraped": scraped_count,
    }


def scrape_mitre_mobile_node(state: ScraperState) -> ScraperState:
    """Logical MITRE mobile ATT&CK scraping step."""
    scraped_count = get_scrape_count_for_source("mitre_mobile")
    
    return {
        "status": [f"mitre_mobile_scrape_step_reached (scraped: {scraped_count} new URLs)"],
        "mitre_mobile_done": True,
        "mitre_mobile_scraped": scraped_count,
    }


# ---------------------- AGGREGATION NODE ----------------------


def aggregate_vendors_node(state: ScraperState) -> ScraperState:
    """
    Aggregation node that waits for all vendor scraping nodes to complete.
    This acts as a synchronization point before moving to the next phase.
    """
    total_scraped = (
        state.get("cisco_scraped", 0) +
        state.get("nokia_scraped", 0) +
        state.get("ericsson_scraped", 0) +
        state.get("huawei_scraped", 0) +
        state.get("variot_scraped", 0) +
        state.get("mitre_mobile_scraped", 0)  # NEW
    )
    
    return {
        "status": [f"all_vendors_scraped (total: {total_scraped} new URLs)"],
    }


# ---------------------- NODES: PIPELINE PHASES ----------------------


def build_textnodes_node(state: ScraperState) -> ScraperState:
    """
    Logical 'TextNode creation' step.

    The actual TextNode creation is implemented in the ingest pipeline;
    this node is for visibility in the graph.
    """
    return {
        "status": ["textnodes_build_step_reached"],
        "textnodes_built": True,
    }


def embed_nodes_node(state: ScraperState) -> ScraperState:
    """
    Logical 'Embedding' step.

    Actual embedding is done inside ingest_all_sources; here we just
    mark that the graph reached the embedding phase.
    """
    return {
        "status": ["embed_step_reached"],
        "embedded": True,
    }


def ingest_qdrant_node(state: ScraperState) -> ScraperState:
    """
    Final ingestion node: runs the full scraping + TextNode creation +
    embeddings + Qdrant upsert via ingest_all_sources.

    NOTE:
      - ingest_all_sources currently returns None (it just logs).
      - We still keep the 'inserted' field in case you later modify
        ingest_all_sources to return a summary dict like:
        {"upserted": <int>, "per_source": {...}}
    """
    summary = asyncio.run(ingest_all_sources(check_qdrant=True, batch_size=32))

    inserted: Optional[int] = None
    if isinstance(summary, dict):
        try:
            inserted = int(summary.get("upserted", 0) or 0)
        except Exception:
            inserted = None

    status_msg = (
        f"ingestion_completed (upserted: {inserted} points)"
        if inserted is not None
        else "ingestion_completed"
    )
    
    return {
        "status": [status_msg],
        "inserted": inserted,
    }


# ---------------------- BUILD GRAPH ----------------------


graph_builder = StateGraph(ScraperState)

# Vendor nodes (these will run in parallel)
graph_builder.add_node("scrape_cisco", scrape_cisco_node)
graph_builder.add_node("scrape_nokia", scrape_nokia_node)
graph_builder.add_node("scrape_ericsson", scrape_ericsson_node)
graph_builder.add_node("scrape_huawei", scrape_huawei_node)
graph_builder.add_node("scrape_variot", scrape_variot_node)
graph_builder.add_node("scrape_mitre_mobile", scrape_mitre_mobile_node)  # NEW

# Aggregation node to synchronize parallel execution
graph_builder.add_node("aggregate_vendors", aggregate_vendors_node)

# Pipeline phase nodes
graph_builder.add_node("build_textnodes", build_textnodes_node)
graph_builder.add_node("embed_nodes", embed_nodes_node)
graph_builder.add_node("ingest_qdrant", ingest_qdrant_node)

# Edges: Parallel execution of all vendors from START
graph_builder.add_edge(START, "scrape_cisco")
graph_builder.add_edge(START, "scrape_nokia")
graph_builder.add_edge(START, "scrape_ericsson")
graph_builder.add_edge(START, "scrape_huawei")
graph_builder.add_edge(START, "scrape_variot")
graph_builder.add_edge(START, "scrape_mitre_mobile")  # NEW

# All vendors converge to the aggregation node
graph_builder.add_edge("scrape_cisco", "aggregate_vendors")
graph_builder.add_edge("scrape_nokia", "aggregate_vendors")
graph_builder.add_edge("scrape_ericsson", "aggregate_vendors")
graph_builder.add_edge("scrape_huawei", "aggregate_vendors")
graph_builder.add_edge("scrape_variot", "aggregate_vendors")
graph_builder.add_edge("scrape_mitre_mobile", "aggregate_vendors")  # NEW

# Sequential pipeline after aggregation
graph_builder.add_edge("aggregate_vendors", "build_textnodes")
graph_builder.add_edge("build_textnodes", "embed_nodes")
graph_builder.add_edge("embed_nodes", "ingest_qdrant")
graph_builder.add_edge("ingest_qdrant", END)

graph = graph_builder.compile()
