from typing import TypedDict, Optional

from langgraph.graph import StateGraph, START, END

from telco_cyber_chat.webscraping.ingest_pipeline import run_full_ingest


class ScraperState(TypedDict, total=False):
    """Minimal state for the scraper graph."""
    status: str
    inserted: Optional[int]


def run_ingest_node(state: ScraperState) -> ScraperState:
    """
    Single node that runs the full scraping + ingest pipeline and
    returns a small summary in the state.
    """
    inserted = run_full_ingest()   # <- your function from ingest_pipeline.py
    return {
        "status": "scraper_run_completed",
        "inserted": inserted,
    }


# Build the graph
graph_builder = StateGraph(ScraperState)

graph_builder.add_node("run_ingest", run_ingest_node)
graph_builder.add_edge(START, "run_ingest")
graph_builder.add_edge("run_ingest", END)

graph = graph_builder.compile()
