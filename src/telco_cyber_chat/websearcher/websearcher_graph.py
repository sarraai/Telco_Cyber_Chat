# src/telco_cyber_chat/websearcher/websearcher_graph.py
from __future__ import annotations

import operator
from typing import Annotated, List, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

from .config import WebsearcherConfig
from .pipeline import ingest_drive_folder_async  # ✅ Import async version


class WebsearcherState(TypedDict, total=False):
    # -------- inputs (MCP tool schema) --------
    drive_folder_id: Optional[str]
    max_files: int
    collection: Optional[str]
    data_type: str
    chunk_size: int
    chunk_overlap: int
    
    # -------- outputs / logs --------
    status: Annotated[List[str], operator.add]
    result: dict
    ok: bool
    error_message: Optional[str]


async def stage_ingest_drive(state: WebsearcherState) -> WebsearcherState:
    """✅ Now async - LangGraph supports async nodes"""
    try:
        cfg = WebsearcherConfig(
            collection=state.get("collection"),
            data_type=str(state.get("data_type") or "unstructured"),
            chunk_size=int(state.get("chunk_size") or 2000),
            chunk_overlap=int(state.get("chunk_overlap") or 200),
            max_files=int(state.get("max_files") or 200),
        )
        
        # ✅ Await async function
        res = await ingest_drive_folder_async(
            cfg,
            drive_folder_id=state.get("drive_folder_id"),
            max_files=state.get("max_files"),
        )
        
        return {
            "status": [
                "✅ Drive ingestion finished",
                f"files_seen={res.get('files_seen')} skipped={res.get('skipped')} processed={res.get('processed')}",
                f"built_nodes={res.get('built_nodes')} inserted={res.get('inserted')} collection={res.get('collection')}",
            ],
            "result": res,
            "ok": True,
        }
    except Exception as e:
        return {
            "status": [f"❌ Drive ingestion failed: {e}"],
            "ok": False,
            "error_message": str(e),
        }


# ✅ Build graph with async node
builder = StateGraph(WebsearcherState)
builder.add_node("ingest_drive", stage_ingest_drive)
builder.add_edge(START, "ingest_drive")
builder.add_edge("ingest_drive", END)

graph = builder.compile()
