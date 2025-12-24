# src/telco_cyber_chat/websearcher/websearcher_graph.py
from __future__ import annotations

import operator
from typing import Annotated, Any, Dict, List, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from .config import WebsearcherConfig


# -----------------------------------------------------------------------------
# State
# -----------------------------------------------------------------------------
class WebsearcherState(TypedDict, total=False):
    # -------- inputs (tool schema) --------
    drive_folder_id: Optional[str]
    max_files: int
    collection: Optional[str]
    data_type: str
    chunk_size: int
    chunk_overlap: int

    # -------- pipeline artifacts --------
    cfg: Any  # WebsearcherConfig

    # -------- outputs / logs --------
    status: Annotated[List[str], operator.add]
    result: dict
    ok: bool
    error_message: Optional[str]

    # summary fields (optional but useful)
    inserted: int
    processed: int
    failed: int
    drive_unique: int
    qdrant_unique: int
    missing: int


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------
def _mk_cfg(state: WebsearcherState) -> WebsearcherConfig:
    return WebsearcherConfig(
        collection=state.get("collection"),
        data_type=str(state.get("data_type") or "unstructured"),
        chunk_size=int(state.get("chunk_size") or 2000),
        chunk_overlap=int(state.get("chunk_overlap") or 200),
        max_files=int(state.get("max_files") or 200),
    )


def _route_ok_or_finalize(state: WebsearcherState) -> str:
    return "finalize" if state.get("ok") is False else "ok"


# -----------------------------------------------------------------------------
# Nodes (lazy imports inside)
# -----------------------------------------------------------------------------
async def stage_start(state: WebsearcherState) -> WebsearcherState:
    """
    Build config only (lightweight).
    """
    try:
        cfg = _mk_cfg(state)
        return {
            "cfg": cfg,
            "ok": True,
            "status": [
                "▶️ Start",
                f"🧩 Config ready: collection={cfg.collection} data_type={cfg.data_type} "
                f"chunk_size={cfg.chunk_size} chunk_overlap={cfg.chunk_overlap} max_files={cfg.max_files}",
            ],
        }
    except Exception as e:
        msg = str(e)
        return {"ok": False, "error_message": msg, "status": [f"❌ Start failed: {msg}"]}


async def stage_run_pipeline(state: WebsearcherState) -> WebsearcherState:
    """
    Run your existing pipeline.py AS-IS:
    - Drive list + today filter
    - Qdrant inventory
    - Download
    - Agentic LlamaParse
    - Chunk
    - TextNodes
    - Embed (hybrid)
    - Upsert
    """
    try:
        from . import pipeline as pipeline_mod  # ✅ lazy import

        cfg: WebsearcherConfig = state.get("cfg") or _mk_cfg(state)

        status: List[str] = [
            "⚙️ Running pipeline.ingest_drive_folder_async (uses pipeline.py as-is) ..."
        ]

        res: Dict[str, Any] = await pipeline_mod.ingest_drive_folder_async(
            cfg,
            drive_folder_id=state.get("drive_folder_id"),
            max_files=state.get("max_files"),
        )

        ok = bool(res.get("ok", True))
        msg = res.get("message")
        if msg:
            status.append(f"ℹ️ {msg}")

        # Map common returned metrics into state (no changes to pipeline.py required)
        inserted = int(res.get("inserted", 0) or 0)
        processed = int(res.get("processed", 0) or 0)
        failed = int(res.get("failed", 0) or 0)

        drive_unique = int(res.get("drive_unique", 0) or 0)
        qdrant_unique = int(res.get("qdrant_unique", 0) or 0)
        missing = int(res.get("missing", 0) or 0)

        status.append("✅ pipeline.ingest_drive_folder_async finished")
        status.append(
            f"📊 drive_unique={drive_unique} qdrant_unique={qdrant_unique} missing={missing} "
            f"processed={processed} failed={failed} inserted={inserted}"
        )

        return {
            "ok": ok,
            "result": res,
            "inserted": inserted,
            "processed": processed,
            "failed": failed,
            "drive_unique": drive_unique,
            "qdrant_unique": qdrant_unique,
            "missing": missing,
            "status": status,
        }

    except Exception as e:
        msg = str(e)
        return {"ok": False, "error_message": msg, "status": [f"❌ Pipeline run failed: {msg}"]}


async def stage_finalize(state: WebsearcherState) -> WebsearcherState:
    """
    Final summary (always reached).
    """
    ok = bool(state.get("ok", True))
    err = state.get("error_message")
    res = state.get("result") or {}

    status: List[str] = ["🏁 Websearcher finished"]
    if ok:
        status.append("✅ OK")
    else:
        status.append("❌ FAILED")
        if err:
            status.append(f"error_message={err}")

    # keep a small normalized summary
    summary = {
        "ok": ok,
        "error_message": err,
        "inserted": int(state.get("inserted") or 0),
        "processed": int(state.get("processed") or 0),
        "failed": int(state.get("failed") or 0),
        "drive_unique": int(state.get("drive_unique") or 0),
        "qdrant_unique": int(state.get("qdrant_unique") or 0),
        "missing": int(state.get("missing") or 0),
        "raw": res,
    }

    status.append(
        f"📌 inserted={summary['inserted']} processed={summary['processed']} failed={summary['failed']} "
        f"missing={summary['missing']}"
    )

    return {"ok": ok, "error_message": err, "result": summary, "status": status}


# -----------------------------------------------------------------------------
# Build graph
# -----------------------------------------------------------------------------
builder = StateGraph(WebsearcherState)

builder.add_node("start", stage_start)
builder.add_node("run_pipeline", stage_run_pipeline)
builder.add_node("finalize", stage_finalize)

builder.add_edge(START, "start")
builder.add_conditional_edges(
    "start",
    _route_ok_or_finalize,
    {"ok": "run_pipeline", "finalize": "finalize"},
)
builder.add_conditional_edges(
    "run_pipeline",
    _route_ok_or_finalize,
    {"ok": "finalize", "finalize": "finalize"},
)
builder.add_edge("finalize", END)

graph = builder.compile()
