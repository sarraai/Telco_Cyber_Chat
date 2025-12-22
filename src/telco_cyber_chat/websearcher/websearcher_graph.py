from __future__ import annotations

import operator
import os
from typing import Annotated, List, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

from .config import WebsearcherConfig
from .pipeline import run_websearcher_build_nodes
from .websearcher_qdrant import upsert_nodes_bgem3_hybrid


class WebsearcherState(TypedDict, total=False):
    # -------- inputs (these become the "tool schema" when exposed via MCP) --------
    seed_urls: List[str]                     # required for crawl mode
    allowed_domains: Optional[List[str]]     # optional domain allowlist
    max_pages: int
    max_depth: int
    tier: str                                # llamaparse tier: agentic/cost_effective/...
    vendor: str                              # payload vendor tag
    collection: Optional[str]                # qdrant collection override

    # -------- outputs / logs --------
    status: Annotated[List[str], operator.add]
    built_nodes: int
    inserted: int
    ok: bool
    error_message: Optional[str]


def _cfg_from_state(state: WebsearcherState) -> WebsearcherConfig:
    return WebsearcherConfig(
        seed_urls=state.get("seed_urls") or [],
        allowed_domains=state.get("allowed_domains"),
        max_pages=int(state.get("max_pages") or 50),
        max_depth=int(state.get("max_depth") or 2),
        parse_tier=str(state.get("tier") or "agentic"),
        vendor=str(state.get("vendor") or "websearcher"),
        collection=state.get("collection"),
    )


def stage_build_nodes(state: WebsearcherState) -> WebsearcherState:
    try:
        cfg = _cfg_from_state(state)
        if not cfg.seed_urls:
            raise ValueError("seed_urls is required (provide at least one URL).")

        nodes = run_websearcher_build_nodes(cfg)

        return {
            "status": [f"✅ Built {len(nodes)} TextNodes"],
            "built_nodes": len(nodes),
            "nodes": nodes,  # keep internal
            "ok": True,
        }
    except Exception as e:
        return {"status": [f"❌ build_nodes failed: {e}"], "ok": False, "error_message": str(e)}


def stage_upsert(state: WebsearcherState) -> WebsearcherState:
    if not state.get("ok"):
        return state

    try:
        nodes = state.get("nodes") or []
        cfg = _cfg_from_state(state)

        qdrant_url = os.environ["QDRANT_URL"]
        qdrant_key = os.environ.get("QDRANT_API_KEY")
        collection = cfg.collection or os.environ.get("QDRANT_COLLECTION", "telco_whitepapers")

        inserted = upsert_nodes_bgem3_hybrid(
            nodes=nodes,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_key,
            collection=collection,
            vendor=cfg.vendor,
        )

        return {
            "status": [f"✅ Upserted {inserted} points into collection={collection}"],
            "inserted": inserted,
            "collection": collection,
            "ok": True,
        }
    except Exception as e:
        return {"status": [f"❌ upsert failed: {e}"], "ok": False, "error_message": str(e)}


def _strip_internal(state: WebsearcherState) -> WebsearcherState:
    # remove internal big objects so responses are small
    state.pop("nodes", None)
    return state


builder = StateGraph(WebsearcherState)
builder.add_node("build_nodes", stage_build_nodes)
builder.add_node("upsert", stage_upsert)
builder.add_node("finalize", _strip_internal)

builder.add_edge(START, "build_nodes")
builder.add_edge("build_nodes", "upsert")
builder.add_edge("upsert", "finalize")
builder.add_edge("finalize", END)

graph = builder.compile()
