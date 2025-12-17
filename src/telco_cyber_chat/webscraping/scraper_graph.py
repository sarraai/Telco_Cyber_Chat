# telco_cyber_chat/webscraping/scraper_graph.py

from __future__ import annotations

import operator
from typing import Optional, Annotated, Dict, List
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from llama_index.core.schema import TextNode


class ScraperState(TypedDict, total=False):
    # Status lines (accumulate)
    status: Annotated[List[str], operator.add]

    # Pipeline artifacts
    nodes: List[TextNode]
    embeddings: object  # keep generic (whatever your embedder returns)

    # Summary
    per_source: Dict[str, int]
    inserted: int
    collection: Optional[str]
    ok: bool
    error_message: Optional[str]


# ---------------------- STAGE 1: SCRAPE (ALL SOURCES) ----------------------

async def stage_scrape_all(state: ScraperState) -> ScraperState:
    """
    Calls ALL scrapers and returns NEW TextNodes (dedup already handled in each scraper).
    """
    try:
        # Import inside node to reduce import-time startup delays
        from telco_cyber_chat.webscraping.ingest_pipeline import scrape_all_sources_only

        out = await scrape_all_sources_only(check_qdrant=True, verbose=True)
        nodes = out.get("nodes", []) or []
        per_source = out.get("per_source", {}) or {}
        collection = out.get("collection")

        lines = [
            "🟣 Stage 1/4: Scrape complete",
            f"📦 New nodes collected: {len(nodes)}",
            f"📊 Per-source: {per_source}",
        ]

        return {
            "status": lines,
            "nodes": nodes,
            "per_source": dict(per_source),
            "collection": collection,
            "ok": True,
        }

    except Exception as e:
        return {
            "status": [f"❌ Stage 1/4 failed (scrape): {e}"],
            "nodes": [],
            "per_source": {},
            "inserted": 0,
            "ok": False,
            "error_message": str(e),
        }


# ---------------------- STAGE 2: TEXTNODE CREATION ----------------------
# If your scrapers already return TextNodes, this stage is effectively a no-op.
# Keep it for observability / future refactor (records -> nodes).

async def stage_textnodes(state: ScraperState) -> ScraperState:
    nodes = state.get("nodes", []) or []
    return {
        "status": [f"🟣 Stage 2/4: TextNodes ready ({len(nodes)})"],
        "nodes": nodes,
        "ok": True,
    }


# ---------------------- STAGE 3: EMBED ----------------------

async def stage_embed(state: ScraperState) -> ScraperState:
    try:
        nodes = state.get("nodes", []) or []
        if not nodes:
            return {"status": ["🟣 Stage 3/4: Nothing to embed (0 nodes)"], "embeddings": None, "ok": True}

        from telco_cyber_chat.webscraping.ingest_pipeline import embed_nodes_only

        embeddings = await embed_nodes_only(nodes, embed_batch_size=32)

        return {
            "status": [f"🟣 Stage 3/4: Embedded {len(nodes)} nodes"],
            "embeddings": embeddings,
            "ok": True,
        }

    except Exception as e:
        return {
            "status": [f"❌ Stage 3/4 failed (embed): {e}"],
            "ok": False,
            "error_message": str(e),
        }


# ---------------------- STAGE 4: UPSERT ----------------------

async def stage_upsert(state: ScraperState) -> ScraperState:
    try:
        nodes = state.get("nodes", []) or []
        embeddings = state.get("embeddings", None)

        if not nodes:
            return {
                "status": ["🟣 Stage 4/4: Nothing to upsert (0 nodes)"],
                "inserted": 0,
                "ok": True,
            }

        from telco_cyber_chat.webscraping.ingest_pipeline import upsert_nodes_only

        result = await upsert_nodes_only(
            nodes=nodes,
            embeddings=embeddings,
            upsert_batch_size=64,
        )

        upserted = int(result.get("upserted", 0) or 0)
        collection = result.get("collection") or state.get("collection")

        return {
            "status": [f"✅ Stage 4/4: Upserted {upserted} nodes to '{collection}'"],
            "inserted": upserted,
            "collection": collection,
            "ok": True,
        }

    except Exception as e:
        return {
            "status": [f"❌ Stage 4/4 failed (upsert): {e}"],
            "inserted": 0,
            "ok": False,
            "error_message": str(e),
        }


# ---------------------- BUILD GRAPH ----------------------

graph_builder = StateGraph(ScraperState)

graph_builder.add_node("scrape_all", stage_scrape_all)
graph_builder.add_node("textnodes", stage_textnodes)
graph_builder.add_node("embed", stage_embed)
graph_builder.add_node("upsert", stage_upsert)

graph_builder.add_edge(START, "scrape_all")
graph_builder.add_edge("scrape_all", "textnodes")
graph_builder.add_edge("textnodes", "embed")
graph_builder.add_edge("embed", "upsert")
graph_builder.add_edge("upsert", END)

graph = graph_builder.compile()
