# src/telco_cyber_chat/websearcher/websearcher_graph.py
from __future__ import annotations

import operator
from datetime import datetime, timezone
from typing import Annotated, Any, Dict, List, Optional, Tuple
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
    cfg: Any  # WebsearcherConfig (kept Any to avoid heavy imports / serialization issues)
    drive_files: List[Dict[str, Any]]          # raw metadata from Drive listing
    drive_files_today: List[Dict[str, Any]]    # filtered "today" subset
    drive_unique: int
    duplicates_on_drive: int

    qdrant_existing_doc_names: List[str]
    qdrant_unique: int
    missing_files: List[Dict[str, Any]]        # files not found in Qdrant

    parsed_docs: List[Dict[str, Any]]          # [{doc_name, text, metadata}]
    chunks: List[Dict[str, Any]]               # [{doc_name, chunk_id, text, metadata}]
    nodes: Any                                  # List[TextNode]
    embeddings: Any                             # embedding outputs (list/np/etc.)

    inserted: int
    processed: int
    failed: int

    # -------- outputs / logs --------
    status: Annotated[List[str], operator.add]
    result: dict
    ok: bool
    error_message: Optional[str]

    # internal short-circuit flag (if nothing to do)
    terminal: bool


# -----------------------------------------------------------------------------
# Small helpers (no heavy imports here)
# -----------------------------------------------------------------------------
def _available_symbols(mod: Any) -> List[str]:
    return sorted([x for x in dir(mod) if not x.startswith("_")])


def _pick_callable(mod: Any, candidates: List[str]) -> Tuple[str, Any]:
    for name in candidates:
        fn = getattr(mod, name, None)
        if callable(fn):
            return name, fn
    raise AttributeError(
        f"None of the expected callables exist: {candidates}. "
        f"Available symbols: {_available_symbols(mod)}"
    )


def _doc_name_from_file(f: Dict[str, Any]) -> str:
    return str(
        f.get("doc_name")
        or f.get("name")
        or f.get("filename")
        or f.get("title")
        or f.get("id")
        or ""
    )


def _best_url_from_file(f: Dict[str, Any]) -> Optional[str]:
    return (
        f.get("download_url")
        or f.get("url")
        or f.get("webContentLink")
        or f.get("webViewLink")
        or f.get("link")
    )


def _is_created_today(file_meta: Dict[str, Any]) -> bool:
    """
    Drive typically returns createdTime / modifiedTime in RFC3339 (e.g. 2025-12-23T12:34:56.000Z).
    Compare to today's UTC date for stable server behavior.
    """
    created = file_meta.get("createdTime") or file_meta.get("created_time") or file_meta.get("created")
    if not created:
        return True  # if unknown, don't filter out

    try:
        s = str(created).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        return dt.astimezone(timezone.utc).date() == datetime.now(timezone.utc).date()
    except Exception:
        return True


def _mk_cfg(state: WebsearcherState) -> WebsearcherConfig:
    return WebsearcherConfig(
        collection=state.get("collection"),
        data_type=str(state.get("data_type") or "unstructured"),
        chunk_size=int(state.get("chunk_size") or 2000),
        chunk_overlap=int(state.get("chunk_overlap") or 200),
        max_files=int(state.get("max_files") or 200),
    )


def _simple_chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> List[str]:
    """
    Fallback chunker (char-based sliding window).
    Used if your .chunker module doesn't expose a chunk function.
    """
    if not text:
        return []
    chunk_size = max(200, int(chunk_size or 2000))
    chunk_overlap = max(0, int(chunk_overlap or 200))
    if chunk_overlap >= chunk_size:
        chunk_overlap = max(0, chunk_size // 5)

    out: List[str] = []
    i = 0
    n = len(text)
    step = max(1, chunk_size - chunk_overlap)

    while i < n:
        out.append(text[i : i + chunk_size])
        i += step
    return out


# -----------------------------------------------------------------------------
# Nodes (all heavy imports happen INSIDE nodes)
# -----------------------------------------------------------------------------
async def stage_start(state: WebsearcherState) -> WebsearcherState:
    """
    START node:
    - Build config
    - List Drive files (via pipeline helper)
    - Filter "created today" (UTC)
    Produces: cfg, drive_files, drive_files_today, drive_unique, duplicates_on_drive, terminal
    """
    try:
        from . import pipeline as pipeline_mod  # ✅ lazy import

        cfg: WebsearcherConfig = state.get("cfg") or _mk_cfg(state)

        candidates = [
            "list_drive_pdfs_async",
            "list_drive_files_async",
            "drive_list_pdfs_async",
            "get_drive_files_async",
        ]
        fn_name, list_fn = _pick_callable(pipeline_mod, candidates)

        status: List[str] = [
            "▶️ Start: config + Drive listing",
            f"🧩 Config: collection={cfg.collection} data_type={cfg.data_type} "
            f"chunk_size={cfg.chunk_size} chunk_overlap={cfg.chunk_overlap} max_files={cfg.max_files}",
            f"📥 Drive listing via pipeline.{fn_name}() ...",
        ]

        res = await list_fn(
            cfg,
            drive_folder_id=state.get("drive_folder_id"),
            max_files=state.get("max_files"),
        )

        files: List[Dict[str, Any]] = (
            res.get("files")
            or res.get("drive_files")
            or res.get("items")
            or res.get("documents")
            or []
        )
        files_seen = res.get("files_seen", len(files))

        files_today = [f for f in files if _is_created_today(f)]
        drive_names = [_doc_name_from_file(f) for f in files_today]
        drive_unique = len(set(drive_names))
        dup_drive = max(0, len(drive_names) - drive_unique)

        status.append("✅ Drive listing finished")
        status.append(f"📁 Total files seen: {files_seen}")
        status.append(f"🆕 Files created today (UTC): {len(files_today)}")
        if len(files_today) == 0:
            status.append("🎯 No new PDFs added today (terminal=True). Next stages will skip.")

        return {
            "ok": True,
            "terminal": (len(files_today) == 0),
            "cfg": cfg,
            "drive_files": files,
            "drive_files_today": files_today,
            "drive_unique": drive_unique,
            "duplicates_on_drive": dup_drive,
            "processed": 0,
            "failed": 0,
            "inserted": 0,
            "status": status,
        }

    except Exception as e:
        msg = str(e)
        hint = ""
        if "None of the expected callables" in msg:
            hint = " | Hint: add a Drive listing helper in pipeline.py (e.g. list_drive_pdfs_async)."
        return {"ok": False, "error_message": msg, "status": [f"❌ Start failed: {msg}{hint}"]}


async def stage_check_database(state: WebsearcherState) -> WebsearcherState:
    """
    Qdrant check:
    - For today's Drive docs, check which doc_name already exists in Qdrant
    Produces: qdrant_existing_doc_names, qdrant_unique, missing_files, terminal
    """
    status: List[str] = []
    try:
        if state.get("terminal"):
            return {"ok": True, "status": ["⏭️ Qdrant check skipped (terminal=True)"]}

        cfg: WebsearcherConfig = state.get("cfg") or _mk_cfg(state)
        files_today = state.get("drive_files_today") or []
        if not files_today:
            return {"ok": True, "terminal": True, "status": ["⏭️ Qdrant check skipped (no files_today)"]}

        import os
        from qdrant_client import QdrantClient
        from qdrant_client.http import models as qmodels

        qdrant_url = os.getenv("QDRANT_URL", "").strip()
        qdrant_api_key = os.getenv("QDRANT_API_KEY", "").strip() or None
        if not qdrant_url:
            raise RuntimeError("QDRANT_URL is missing in environment.")

        collection = cfg.collection or os.getenv("QDRANT_COLLECTION", "").strip()
        if not collection:
            raise RuntimeError("Collection missing: set state.collection or QDRANT_COLLECTION env var.")

        doc_key = os.getenv("QDRANT_DOCNAME_FIELD", "doc_name").strip()

        client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

        status.append("🗄️ Checking Qdrant for existing documents ...")
        existing: List[str] = []
        missing_files: List[Dict[str, Any]] = []

        seen_names: set = set()
        for f in files_today:
            doc_name = _doc_name_from_file(f)
            if not doc_name:
                missing_files.append(f)
                continue
            if doc_name in seen_names:
                continue
            seen_names.add(doc_name)

            flt = qmodels.Filter(
                must=[qmodels.FieldCondition(key=doc_key, match=qmodels.MatchValue(value=doc_name))]
            )

            points, _ = client.scroll(
                collection_name=collection,
                scroll_filter=flt,
                limit=1,
                with_payload=False,
                with_vectors=False,
            )

            if points:
                existing.append(doc_name)
            else:
                missing_files.append(f)

        qdrant_unique = len(set(existing))
        missing = len(missing_files)

        status.append("✅ Qdrant check finished")
        status.append(
            f"drive_unique={state.get('drive_unique', len(seen_names))} "
            f"qdrant_unique={qdrant_unique} missing={missing} "
            f"dup_drive={state.get('duplicates_on_drive', 0)}"
        )

        if missing == 0:
            status.append("🎯 All new PDFs already in Qdrant (terminal=True). Next stages will skip.")

        return {
            "ok": True,
            "qdrant_existing_doc_names": existing,
            "qdrant_unique": qdrant_unique,
            "missing_files": missing_files,
            "terminal": (missing == 0),
            "status": status,
        }

    except Exception as e:
        msg = str(e)
        return {"ok": False, "error_message": msg, "status": [f"❌ Qdrant check failed: {msg}"]}


async def stage_agentic_llamaparse(state: WebsearcherState) -> WebsearcherState:
    """
    Agentic LlamaParse stage:
    - Parse missing files (URLs) into text/markdown
    Produces: parsed_docs, processed, (maybe terminal if nothing parsed)
    """
    status: List[str] = []
    try:
        if state.get("terminal"):
            return {"ok": True, "status": ["⏭️ LlamaParse skipped (terminal=True)"]}

        cfg: WebsearcherConfig = state.get("cfg") or _mk_cfg(state)
        missing_files = state.get("missing_files") or []
        if not missing_files:
            return {"ok": True, "terminal": True, "status": ["⏭️ LlamaParse skipped (no missing_files)"]}

        from . import llamaparse_url as lp_mod  # ✅ lazy import

        candidates = [
            "parse_urls_agentic_async",
            "parse_url_agentic_async",
            "llamaparse_urls_async",
            "parse_urls_async",
        ]
        fn_name, parse_fn = _pick_callable(lp_mod, candidates)

        items: List[Dict[str, Any]] = []
        url_missing = 0
        for f in missing_files:
            url = _best_url_from_file(f)
            if not url:
                url_missing += 1
            items.append({"doc_name": _doc_name_from_file(f), "url": url, "file": f})

        status.append(f"🤖 Agentic LlamaParse via {lp_mod.__name__}.{fn_name}() ...")
        if url_missing:
            status.append(f"⚠️ Missing URL for {url_missing} file(s); parser helper should handle/skip them.")

        parsed = await parse_fn(cfg, items)

        parsed_docs: List[Dict[str, Any]] = parsed.get("docs") if isinstance(parsed, dict) else parsed
        if not isinstance(parsed_docs, list):
            raise RuntimeError(
                f"LlamaParse helper returned unexpected type: {type(parsed_docs)}. "
                "Expected list of docs or dict with key 'docs'."
            )

        ok_docs = len(parsed_docs)
        status.append("✅ LlamaParse finished")
        status.append(f"📄 Parsed documents: {ok_docs}")

        if ok_docs == 0:
            status.append("🎯 Nothing parsed (terminal=True). Next stages will skip.")
            return {"ok": True, "parsed_docs": parsed_docs, "processed": 0, "terminal": True, "status": status}

        return {"ok": True, "parsed_docs": parsed_docs, "processed": ok_docs, "status": status}

    except Exception as e:
        msg = str(e)
        hint = ""
        if "llama-cloud-services" in msg or "LlamaParse SDK" in msg:
            hint = " | Hint: ensure `llama-cloud-services>=0.5.0` is installed in requirements.txt."
        return {"ok": False, "error_message": msg, "status": [f"❌ LlamaParse failed: {msg}{hint}"]}


async def stage_chunking(state: WebsearcherState) -> WebsearcherState:
    """
    Chunk parsed docs -> chunk dicts.
    Produces: chunks, (maybe terminal if no chunks)
    """
    status: List[str] = []
    try:
        if state.get("terminal"):
            return {"ok": True, "status": ["⏭️ Chunking skipped (terminal=True)"]}

        cfg: WebsearcherConfig = state.get("cfg") or _mk_cfg(state)
        parsed_docs = state.get("parsed_docs") or []
        if not parsed_docs:
            return {"ok": True, "terminal": True, "status": ["⏭️ Chunking skipped (no parsed_docs)"]}

        chunk_size = int(cfg.chunk_size or 2000)
        chunk_overlap = int(cfg.chunk_overlap or 200)

        chunks: List[Dict[str, Any]] = []

        try:
            from . import chunker as chunker_mod  # ✅ lazy import
            candidates = ["chunk_docs", "chunk_markdown_docs", "chunk_text", "chunk_markdown"]
            fn_name, chunk_fn = _pick_callable(chunker_mod, candidates)

            status.append(f"🧱 Chunking via {chunker_mod.__name__}.{fn_name}() ...")
            chunked = chunk_fn(parsed_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            if isinstance(chunked, dict) and "chunks" in chunked:
                chunks = chunked["chunks"]
            elif isinstance(chunked, list):
                chunks = chunked
            else:
                raise RuntimeError(f"chunker output type not supported: {type(chunked)}")

        except Exception:
            status.append("🧱 Chunking via fallback chunker (char sliding window) ...")
            for d in parsed_docs:
                doc_name = str(d.get("doc_name") or "")
                text = str(d.get("text") or "")
                meta = d.get("metadata") or {}
                parts = _simple_chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                for i, p in enumerate(parts):
                    chunks.append({"doc_name": doc_name, "chunk_id": i, "text": p, "metadata": dict(meta)})

        status.append("✅ Chunking finished")
        status.append(f"🧩 Total chunks: {len(chunks)}")

        if len(chunks) == 0:
            status.append("🎯 No chunks created (terminal=True). Next stages will skip.")
            return {"ok": True, "chunks": chunks, "terminal": True, "status": status}

        return {"ok": True, "chunks": chunks, "status": status}

    except Exception as e:
        msg = str(e)
        return {"ok": False, "error_message": msg, "status": [f"❌ Chunking failed: {msg}"]}


async def stage_build_textnodes(state: WebsearcherState) -> WebsearcherState:
    """
    Convert chunks -> LlamaIndex TextNodes.
    Produces: nodes, (maybe terminal if none)
    """
    status: List[str] = []
    try:
        if state.get("terminal"):
            return {"ok": True, "status": ["⏭️ TextNodes skipped (terminal=True)"]}

        chunks = state.get("chunks") or []
        if not chunks:
            return {"ok": True, "terminal": True, "status": ["⏭️ TextNodes skipped (no chunks)"]}

        from llama_index.core.schema import TextNode  # ✅ lazy import

        cfg: WebsearcherConfig = state.get("cfg") or _mk_cfg(state)

        status.append("🧱 Building TextNodes ...")
        nodes: List[TextNode] = []

        for c in chunks:
            doc_name = str(c.get("doc_name") or "")
            chunk_id = c.get("chunk_id")
            meta = dict(c.get("metadata") or {})
            meta.update(
                {
                    "doc_name": doc_name,
                    "chunk_id": chunk_id,
                    "data_type": str(cfg.data_type),
                    "source": "drive_websearcher",
                }
            )

            node_id = f"{doc_name}::chunk::{chunk_id}"
            nodes.append(TextNode(id_=node_id, text=str(c.get("text") or ""), metadata=meta))

        status.append("✅ TextNodes ready")
        status.append(f"🧱 built_nodes={len(nodes)}")

        if len(nodes) == 0:
            status.append("🎯 No nodes built (terminal=True). Next stages will skip.")
            return {"ok": True, "nodes": nodes, "terminal": True, "status": status}

        return {"ok": True, "nodes": nodes, "status": status}

    except Exception as e:
        msg = str(e)
        return {"ok": False, "error_message": msg, "status": [f"❌ TextNodes stage failed: {msg}"]}


async def stage_embedding(state: WebsearcherState) -> WebsearcherState:
    """
    Embed TextNodes using your remote embedding helper.
    Produces: embeddings, (maybe terminal if none)
    """
    status: List[str] = []
    try:
        if state.get("terminal"):
            return {"ok": True, "status": ["⏭️ Embedding skipped (terminal=True)"]}

        cfg: WebsearcherConfig = state.get("cfg") or _mk_cfg(state)
        nodes = state.get("nodes") or []
        if not nodes:
            return {"ok": True, "terminal": True, "status": ["⏭️ Embedding skipped (no nodes)"]}

        embed_mod = None
        for mod_name in [".embed_webscraper", ".embed_loader"]:
            try:
                embed_mod = __import__(f"{__package__}{mod_name}", fromlist=["*"])
                break
            except Exception:
                embed_mod = None

        if embed_mod is None:
            raise ImportError(
                "No embedding helper found. Expected telco_cyber_chat.websearcher.embed_webscraper "
                "or embed_loader module."
            )

        candidates = ["embed_textnodes_async", "embed_nodes_async", "embed_async"]
        fn_name, embed_fn = _pick_callable(embed_mod, candidates)

        status.append(f"🧠 Embedding via {embed_mod.__name__}.{fn_name}() ...")
        emb = await embed_fn(cfg, nodes)

        status.append("✅ Embedding finished")
        if emb is None:
            status.append("🎯 Embedding returned None (terminal=True). Next stages will skip.")
            return {"ok": True, "embeddings": emb, "terminal": True, "status": status}

        return {"ok": True, "embeddings": emb, "status": status}

    except Exception as e:
        msg = str(e)
        return {"ok": False, "error_message": msg, "status": [f"❌ Embedding failed: {msg}"]}


async def stage_upsert(state: WebsearcherState) -> WebsearcherState:
    """
    Upsert into Qdrant (delegates to pipeline.py helper).
    Produces: inserted, result
    """
    status: List[str] = []
    try:
        if state.get("terminal"):
            return {"ok": True, "status": ["⏭️ Upsert skipped (terminal=True)"]}

        cfg: WebsearcherConfig = state.get("cfg") or _mk_cfg(state)
        nodes = state.get("nodes") or []
        emb = state.get("embeddings")

        if not nodes:
            return {"ok": True, "terminal": True, "status": ["⏭️ Upsert skipped (no nodes)"]}

        from . import pipeline as pipeline_mod  # ✅ lazy import

        candidates = ["upsert_nodes_async", "upsert_to_qdrant_async", "qdrant_upsert_async"]
        fn_name, upsert_fn = _pick_callable(pipeline_mod, candidates)

        status.append(f"⬆️ Upserting via pipeline.{fn_name}() ...")
        res = await upsert_fn(cfg, nodes, emb)

        inserted = res.get("inserted", res.get("upserted", 0)) if isinstance(res, dict) else 0
        status.append("✅ Upsert finished")
        status.append(f"inserted={inserted} collection={cfg.collection}")

        return {"ok": True, "inserted": inserted, "result": (res if isinstance(res, dict) else {}), "status": status}

    except Exception as e:
        msg = str(e)
        hint = ""
        if "None of the expected callables" in msg:
            hint = " | Hint: add an upsert helper in pipeline.py (e.g. upsert_nodes_async(cfg, nodes, embeddings))."
        return {"ok": False, "error_message": msg, "status": [f"❌ Upsert failed: {msg}{hint}"]}


async def stage_finalize(state: WebsearcherState) -> WebsearcherState:
    """
    Final summary.
    Runs on both success and failure (graph routes here on ok=False).
    """
    try:
        ok = bool(state.get("ok", True))
        err = state.get("error_message")

        files_today = state.get("drive_files_today") or []
        missing = state.get("missing_files") or []
        built_nodes = len(state.get("nodes") or []) if state.get("nodes") is not None else 0

        inserted = int(state.get("inserted") or 0)
        processed = int(state.get("processed") or 0)
        failed = int(state.get("failed") or 0)

        res = state.get("result") or {}

        summary: Dict[str, Any] = {
            "ok": ok,
            "error_message": err,
            "files_today": len(files_today),
            "drive_unique": state.get("drive_unique"),
            "duplicates_on_drive": state.get("duplicates_on_drive"),
            "qdrant_unique": state.get("qdrant_unique"),
            "missing": len(missing),
            "processed": processed,
            "built_nodes": built_nodes,
            "inserted": inserted,
            "failed": failed,
            "collection": (state.get("cfg").collection if state.get("cfg") else state.get("collection")),
            "terminal": bool(state.get("terminal", False)),
            "raw": res,
        }

        status: List[str] = ["🏁 Websearcher pipeline finished (sequential graph)"]
        status.append(
            f"📁 today={len(files_today)} missing={len(missing)} processed={processed} "
            f"built_nodes={built_nodes} inserted={inserted} failed={failed} terminal={bool(state.get('terminal'))}"
        )

        if not ok and err:
            status.append(f"❌ Pipeline ended with error: {err}")
        else:
            if len(files_today) == 0:
                status.append("🎯 No new PDFs added today - ingestion skipped")
            elif len(missing) == 0:
                status.append("🎯 All new PDFs already in Qdrant - nothing to process")
            elif inserted > 0:
                status.append(f"🎯 Successfully upserted {inserted} chunk(s) into Qdrant")

        return {"ok": ok, "error_message": err, "result": summary, "status": status}

    except Exception as e:
        msg = str(e)
        return {"ok": False, "error_message": msg, "status": [f"❌ Finalize failed: {msg}"]}


# -----------------------------------------------------------------------------
# Routing helpers
# -----------------------------------------------------------------------------
def _route_next_or_finalize(state: WebsearcherState) -> str:
    """
    If a stage failed -> go finalize.
    Otherwise -> proceed to the next sequential stage.
    """
    return "finalize" if state.get("ok") is False else "next"


# -----------------------------------------------------------------------------
# Build graph (STRICTLY SEQUENTIAL)
# START -> start -> qdrant_check -> llamaparse -> chunk -> textnodes -> embed -> upsert -> finalize -> END
# -----------------------------------------------------------------------------
builder = StateGraph(WebsearcherState)

builder.add_node("start", stage_start)
builder.add_node("qdrant_check", stage_check_database)
builder.add_node("llamaparse", stage_agentic_llamaparse)
builder.add_node("chunking", stage_chunking)
builder.add_node("textnodes", stage_build_textnodes)
builder.add_node("embedding", stage_embedding)
builder.add_node("upsert", stage_upsert)
builder.add_node("finalize", stage_finalize)

# START -> start
builder.add_edge(START, "start")

# start -> qdrant_check OR finalize (if failed)
builder.add_conditional_edges(
    "start",
    _route_next_or_finalize,
    {"next": "qdrant_check", "finalize": "finalize"},
)

# qdrant_check -> llamaparse OR finalize (if failed)
builder.add_conditional_edges(
    "qdrant_check",
    _route_next_or_finalize,
    {"next": "llamaparse", "finalize": "finalize"},
)

# llamaparse -> chunking OR finalize (if failed)
builder.add_conditional_edges(
    "llamaparse",
    _route_next_or_finalize,
    {"next": "chunking", "finalize": "finalize"},
)

# chunking -> textnodes OR finalize (if failed)
builder.add_conditional_edges(
    "chunking",
    _route_next_or_finalize,
    {"next": "textnodes", "finalize": "finalize"},
)

# textnodes -> embedding OR finalize (if failed)
builder.add_conditional_edges(
    "textnodes",
    _route_next_or_finalize,
    {"next": "embedding", "finalize": "finalize"},
)

# embedding -> upsert OR finalize (if failed)
builder.add_conditional_edges(
    "embedding",
    _route_next_or_finalize,
    {"next": "upsert", "finalize": "finalize"},
)

# upsert -> finalize (success) OR finalize (failure)
builder.add_conditional_edges(
    "upsert",
    _route_next_or_finalize,
    {"next": "finalize", "finalize": "finalize"},
)

# finalize -> END
builder.add_edge("finalize", END)

graph = builder.compile()
