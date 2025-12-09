import asyncio
from typing_extensions import TypedDict, Optional
from langgraph.graph import StateGraph, START, END
from telco_cyber_chat.webscraping.ingest_pipeline import ingest_all_sources

class ScraperState(TypedDict, total=False):
    """Minimal state for the scraper graph."""
    status: str
    inserted: Optional[int]

def run_ingest_node(state: ScraperState) -> ScraperState:
    """
    Single node that runs the full scraping + ingest pipeline and
    returns a small summary in the state.
    """
    # Run the async function synchronously
    asyncio.run(ingest_all_sources(check_qdrant=True, batch_size=32))
    
    return {
        "status": "scraper_run_completed",
        "inserted": None,  # ingest_all_sources doesn't return a count
    }

# Build the graph
graph_builder = StateGraph(ScraperState)
graph_builder.add_node("run_ingest", run_ingest_node)
graph_builder.add_edge(START, "run_ingest")
graph_builder.add_edge("run_ingest", END)
graph = graph_builder.compile()
