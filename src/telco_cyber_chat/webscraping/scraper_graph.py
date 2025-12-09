import asyncio
from typing import Optional

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
    status: str
    inserted: Optional[int]

    # Per-source flags (purely informational for the graph)
    cisco_done: bool
    nokia_done: bool
    ericsson_done: bool
    huawei_done: bool
    variot_done: bool

    # Pipeline phase flags
    textnodes_built: bool
    embedded: bool


# ---------------------- NODES: VENDORS ----------------------


def scrape_cisco_node(state: ScraperState) -> ScraperState:
    """
    Logical node for 'Cisco scraping step'.

    NOTE: The actual scraping + ingest is still orchestrated inside
    ingest_all_sources; this node is mainly here so that the graph
    shows a dedicated Cisco step.
    """
    return {
        **state,
        "status": "cisco_scrape_step_reached",
        "cisco_done": True,
    }


def scrape_nokia_node(state: ScraperState) -> ScraperState:
    """Logical Nokia scraping step."""
    return {
        **state,
        "status": "nokia_scrape_step_reached",
        "nokia_done": True,
    }


def scrape_ericsson_node(state: ScraperState) -> ScraperState:
    """Logical Ericsson scraping step."""
    return {
        **state,
        "status": "ericsson_scrape_step_reached",
        "ericsson_done": True,
    }


def scrape_huawei_node(state: ScraperState) -> ScraperState:
    """Logical Huawei scraping step."""
    return {
        **state,
        "status": "huawei_scrape_step_reached",
        "huawei_done": True,
    }


def scrape_variot_node(state: ScraperState) -> ScraperState:
    """Logical VARIoT scraping step."""
    return {
        **state,
        "status": "variot_scrape_step_reached",
        "variot_done": True,
    }


# ---------------------- NODES: PIPELINE PHASES ----------------------


def build_textnodes_node(state: ScraperState) -> ScraperState:
    """
    Logical 'TextNode creation' step.

    The actual TextNode creation is implemented in the ingest pipeline;
    this node is for visibility in the graph.
    """
    return {
        **state,
        "status": "textnodes_build_step_reached",
        "textnodes_built": True,
    }


def embed_nodes_node(state: ScraperState) -> ScraperState:
    """
    Logical 'Embedding' step.

    Actual embedding is done inside ingest_all_sources; here we just
    mark that the graph reached the embedding phase.
    """
    return {
        **state,
        "status": "embed_step_reached",
        "embedded": True,
    }


def ingest_qdrant_node(state: ScraperState) -> ScraperState:
    """
    Final ingestion node: runs the full scraping + TextNode creation +
    embeddings + Qdrant upsert via ingest_all_sources.

    We capture the 'upserted' count if ingest_all_sources returns a
    summary dict.
    """
    summary = asyncio.run(ingest_all_sources(check_qdrant=True, batch_size=32))

    inserted: Optional[int] = None
    if isinstance(summary, dict):
        try:
            inserted = int(summary.get("upserted", 0) or 0)
        except Exception:
            inserted = None

    return {
        **state,
        "status": "ingestion_completed",
        "inserted": inserted,
    }


# ---------------------- BUILD GRAPH ----------------------


graph_builder = StateGraph(ScraperState)

# Vendor nodes
graph_builder.add_node("scrape_cisco", scrape_cisco_node)
graph_builder.add_node("scrape_nokia", scrape_nokia_node)
graph_builder.add_node("scrape_ericsson", scrape_ericsson_node)
graph_builder.add_node("scrape_huawei", scrape_huawei_node)
graph_builder.add_node("scrape_variot", scrape_variot_node)

# Pipeline phase nodes
graph_builder.add_node("build_textnodes", build_textnodes_node)
graph_builder.add_node("embed_nodes", embed_nodes_node)
graph_builder.add_node("ingest_qdrant", ingest_qdrant_node)

# Edges: linear pipeline for clear visualization
graph_builder.add_edge(START, "scrape_cisco")
graph_builder.add_edge("scrape_cisco", "scrape_nokia")
graph_builder.add_edge("scrape_nokia", "scrape_ericsson")
graph_builder.add_edge("scrape_ericsson", "scrape_huawei")
graph_builder.add_edge("scrape_huawei", "scrape_variot")
graph_builder.add_edge("scrape_variot", "build_textnodes")
graph_builder.add_edge("build_textnodes", "embed_nodes")
graph_builder.add_edge("embed_nodes", "ingest_qdrant")
graph_builder.add_edge("ingest_qdrant", END)

graph = graph_builder.compile()
