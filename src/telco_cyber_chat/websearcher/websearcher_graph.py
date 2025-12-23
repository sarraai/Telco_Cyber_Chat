# src/telco_cyber_chat/websearcher/websearcher_graph.py
from __future__ import annotations

import operator
from typing import Annotated, List, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END

from .config import WebsearcherConfig


class WebsearcherState(TypedDict, total=False):
    # -------- inputs (tool schema) --------
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
    """
    Async LangGraph node.

    Key goals:
    - Avoid importing heavy modules at import time (LangSmith startup time).
    - Give a clearer error if pipeline import fails (e.g., missing llama-cloud-services).
    """
    status: List[str] = []

    try:
        # ✅ Lazy import to reduce LangSmith startup time
        from . import pipeline as pipeline_mod

        ingest_fn = getattr(pipeline_mod, "ingest_drive_folder_async", None)
        if ingest_fn is None or not callable(ingest_fn):
            available = [x for x in dir(pipeline_mod) if "ingest" in x.lower()]
            raise ImportError(
                "pipeline.py loaded but `ingest_drive_folder_async` was not found. "
                f"Available ingest-like symbols: {available}"
            )

        cfg = WebsearcherConfig(
            collection=state.get("collection"),
            data_type=str(state.get("data_type") or "unstructured"),
            chunk_size=int(state.get("chunk_size") or 2000),
            chunk_overlap=int(state.get("chunk_overlap") or 200),
            max_files=int(state.get("max_files") or 200),
        )

        # ✅ Run pipeline
        res = await ingest_fn(
            cfg,
            drive_folder_id=state.get("drive_folder_id"),
            max_files=state.get("max_files"),
        )

        # Status lines (support both old + new keys)
        files_seen = res.get("files_seen")
        skipped = res.get("skipped_existing", res.get("skipped"))
        processed = res.get("processed")
        built_nodes = res.get("built_nodes")
        inserted = res.get("inserted")
        collection = res.get("collection")

        drive_unique = res.get("drive_unique")
        qdrant_unique = res.get("qdrant_unique")
        missing = res.get("missing")
        dup_drive = res.get("duplicates_on_drive")

        status.append("✅ Drive ingestion finished")

        if drive_unique is not None and qdrant_unique is not None and missing is not None:
            status.append(
                f"drive_unique={drive_unique} qdrant_unique={qdrant_unique} "
                f"missing={missing} dup_drive={dup_drive}"
            )

        status.append(f"files_seen={files_seen} skipped={skipped} processed={processed}")
        status.append(f"built_nodes={built_nodes} inserted={inserted} collection={collection}")

        return {"status": status, "result": res, "ok": True}

    except Exception as e:
        # include a hint for the most common failure mode
        msg = str(e)
        hint = ""
        if "llama-cloud-services" in msg or "LlamaParse SDK not installed" in msg:
            hint = " | Hint: ensure `llama-cloud-services==0.5.1` is installed in THIS assistant requirements.txt."
        return {
            "status": [f"❌ Drive ingestion failed: {msg}{hint}"],
            "ok": False,
            "error_message": msg,
        }


# Build graph
builder = StateGraph(WebsearcherState)
builder.add_node("ingest_drive", stage_ingest_drive)
builder.add_edge(START, "ingest_drive")
builder.add_edge("ingest_drive", END)

graph = builder.compile()
