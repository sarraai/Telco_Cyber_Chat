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
    return (
        str(f.get("doc_name") or f.get("name") or f.get("filename") or f.get("title") or f.get("id") or "")
    )


def _best_url_from_file(f: Dict[str, Any]) -> Optional[str]:
    # Try common Drive / ingestion fields (your pipeline can set one of these)
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
    We compare to today's date in UTC to keep it simple & stable in servers.
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
    Fallback chunker (char-based sliding window) used if your .chunker module
    doesn't expose a chunk function.
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
async def stage_build_config(state: WebsearcherState) -> WebsearcherState:
    try:
        cfg = _mk_cfg(state)
        return {
            "cfg": cfg,
            "ok": True,
            "terminal": False,
            "status": [
                "🧩 Config ready",
                f"collection={cfg.collection} data_type={cfg.data_type} "
                f"chunk_size={cfg.chunk_size} chunk_overlap={cfg.chunk_overlap} max_files={cfg.max_files}",
            ],
        }
    except Exception as e:
        msg = str(e)
        return {"ok": False, "error_message": msg, "status": [f"❌ Config stage failed: {msg}"]}


async def stage_fetch_drive_files(state: WebsearcherState) -> WebsearcherState:
    """
    Start -> fetch/list files from Drive folder (lazy import).
    Prefer using pipeline.py helper for listing (fast, centralized),
    then we do a "created today" filter here unless your pipeline already does it.
    """
    try:
        from . import pipeline as pipeline_mod  # ✅ lazy import

        cfg: WebsearcherConfig = state.get("cfg") or _mk_cfg(state)

        # expected helper names (implement one in pipeline.py)
        candidates = [
            "list_drive_pdfs_async",
            "list_drive_files_async",
            "drive_list_pdfs_async",
            "get_drive_files_async",
        ]

        fn_name, list_fn = _pick_callable(pipeline_mod, candidates)

        status: List[str] = [f"📥 Drive listing via pipeline.{fn_name}() ..."]

        res = await list_fn(
            cfg,
            drive_folder_id=state.get("drive_folder_id"),
            max_files=state.get("max_files"),
        )

        # Normalize
        files: List[Dict[str, Any]] = (
            res.get("files")
            or res.get("drive_files")
            or res.get("items")
            or res.get("documents")
            or []
        )

        files_seen = res.get("files_seen", len(files))

        # Local "today" filter (unless you already filtered in the listing fn)
        files_today = [f for f in files if _is_created_today(f)]
        drive_names = [_doc_name_from_file(f) for f in files_today]
        drive_unique = len(set(drive_names))
        dup_drive = max(0, len(drive_names) - drive_unique)

        status.append("✅ Drive listing finished")
        status.append(f"📁 Total files seen: {files_seen}")
        status.append(f"🆕 Files created today (UTC): {len(files_today)}")
        if len(files_today) == 0:
            status.append("🎯 No new PDFs added today - nothing to process")

        return {
            "ok": True,
            "cfg": cfg,
            "drive_files": files,
            "drive_files_today": files_today,
            "drive_unique": drive_unique,
            "duplicates_on_drive": dup_drive,
            "processed": 0,
            "failed": 0,
            "inserted": 0,
            "terminal": (len(files_today) == 0),
            "status": status,
        }

    except Exception as e:
        msg = str(e)
        hint = ""
        if "pipeline" in msg and "list_drive" in msg:
            hint = " | Hint: add a Drive listing helper in pipeline.py (e.g. list_drive_pdfs_async)."
        return {
            "ok": False,
            "error_message": msg,
            "status": [f"❌ Drive listing failed: {msg}{hint}"],
        }


async def stage_check_database(state: WebsearcherState) -> WebsearcherState:
    """
    Check Qdrant (DB) to find which docs already exist.
    We do an efficient per-doc_name existence check (no full collection scan).
    """
    status: List[str] = []
    try:
        if state.get("terminal"):
            return {"ok": True, "status": ["⏭️ DB check skipped (no new files today)"]}

        cfg: WebsearcherConfig = state.get("cfg") or _mk_cfg(state)
        files_today = state.get("drive_files_today") or []
        if not files_today:
            return {"ok": True, "terminal": True, "status": ["⏭️ DB check skipped (no files to check)"]}

        # Lazy import qdrant client here
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

        # De-duplicate Drive doc_names for efficient checks
        seen_names: set = set()
        for f in files_today:
            doc_name = _doc_name_from_file(f)
            if not doc_name:
                # if no name, treat as missing to avoid silently skipping
                missing_files.append(f)
                continue
            if doc_name in seen_names:
                continue
            seen_names.add(doc_name)

            flt = qmodels.Filter(
                must=[
                    qmodels.FieldCondition(
                        key=doc_key,
                        match=qmodels.MatchValue(value=doc_name),
                    )
                ]
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
            status.append("🎯 All new PDFs already in Qdrant - nothing to process")

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
        return {"ok": False, "error_message": msg, "status": [f"❌ DB check failed: {msg}"]}


async def stage_agentic_llamaparse(state: WebsearcherState) -> WebsearcherState:
    """
    Agentic LlamaParse stage:
    - For each missing file, obtain URL (download_url/webViewLink/etc.) and parse to markdown/text.
    - Delegates to your llamaparse_url.py (recommended) if present.
    """
    status: List[str] = []
    try:
        if state.get("terminal"):
            return {"ok": True, "status": ["⏭️ LlamaParse skipped (nothing to parse)"]}

        cfg: WebsearcherConfig = state.get("cfg") or _mk_cfg(state)
        missing_files = state.get("missing_files") or []
        if not missing_files:
            return {"ok": True, "terminal": True, "status": ["⏭️ LlamaParse skipped (no missing files)"]}

        # Prefer your existing helper module
        from . import llamaparse_url as lp_mod  # ✅ lazy import

        # expected helper names (implement one in llamaparse_url.py)
        candidates = [
            "parse_urls_agentic_async",
            "parse_url_agentic_async",
            "llamaparse_urls_async",
            "parse_urls_async",
        ]
        fn_name, parse_fn = _pick_callable(lp_mod, candidates)

        # Build URL list + metadata
        items: List[Dict[str, Any]] = []
        for f in missing_files:
            url = _best_url_from_file(f)
            items.append(
                {
                    "doc_name": _doc_name_from_file(f),
                    "url": url,
                    "file": f,
                }
            )

        status.append(f"🤖 Agentic LlamaParse via {lp_mod.__name__}.{fn_name}() ...")
        parsed = await parse_fn(cfg, items)  # your helper should return list[{doc_name,text,metadata}]

        # Normalize output
        parsed_docs: List[Dict[str, Any]] = parsed.get("docs") if isinstance(parsed, dict) else parsed
        if not isinstance(parsed_docs, list):
            raise RuntimeError(
                f"LlamaParse helper returned unexpected type: {type(parsed_docs)}. "
                "Expected list of docs or dict with key 'docs'."
            )

        ok_docs = len(parsed_docs)
        status.append("✅ LlamaParse finished")
        status.append(f"📄 Parsed documents: {ok_docs}")

        return {
            "ok": True,
            "parsed_docs": parsed_docs,
            "processed": ok_docs,
            "status": status,
        }

    except Exception as e:
        msg = str(e)
        hint = ""
        if "llama-cloud-services" in msg or "LlamaParse SDK not installed" in msg:
            hint = " | Hint: ensure `llama-cloud-services>=0.5.0` is installed in requirements.txt."
        return {"ok": False, "error_message": msg, "status": [f"❌ LlamaParse failed: {msg}{hint}"]}


async def stage_chunking(state: WebsearcherState) -> WebsearcherState:
    """
    Chunk parsed markdown/text into chunk dicts.
    Uses .chunker helper if present; otherwise uses fallback chunker in this file.
    """
    status: List[str] = []
    try:
        if state.get("terminal"):
            return {"ok": True, "status": ["⏭️ Chunking skipped (nothing to chunk)"]}

        cfg: WebsearcherConfig = state.get("cfg") or _mk_cfg(state)
        parsed_docs = state.get("parsed_docs") or []
        if not parsed_docs:
            return {"ok": True, "terminal": True, "status": ["⏭️ Chunking skipped (no parsed docs)"]}

        chunk_size = int(cfg.chunk_size or 2000)
        chunk_overlap = int(cfg.chunk_overlap or 200)

        # Try using your chunker.py first
        chunks: List[Dict[str, Any]] = []
        try:
            from . import chunker as chunker_mod  # ✅ lazy import

            candidates = ["chunk_docs", "chunk_markdown_docs", "chunk_text", "chunk_markdown"]
            fn_name, chunk_fn = _pick_callable(chunker_mod, candidates)

            status.append(f"🧱 Chunking via {chunker_mod.__name__}.{fn_name}() ...")
            chunked = chunk_fn(parsed_docs, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

            # normalize
            if isinstance(chunked, dict) and "chunks" in chunked:
                chunks = chunked["chunks"]
            elif isinstance(chunked, list):
                chunks = chunked
            else:
                raise RuntimeError(f"chunker output type not supported: {type(chunked)}")

        except Exception:
            # Fallback
            status.append("🧱 Chunking via fallback chunker (char sliding window) ...")
            for d in parsed_docs:
                doc_name = str(d.get("doc_name") or "")
                text = str(d.get("text") or "")
                meta = d.get("metadata") or {}
                parts = _simple_chunk_text(text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
                for i, p in enumerate(parts):
                    chunks.append(
                        {
                            "doc_name": doc_name,
                            "chunk_id": i,
                            "text": p,
                            "metadata": dict(meta),
                        }
                    )

        status.append("✅ Chunking finished")
        status.append(f"🧩 Total chunks: {len(chunks)}")

        return {"ok": True, "chunks": chunks, "status": status}

    except Exception as e:
        msg = str(e)
        return {"ok": False, "error_message": msg, "status": [f"❌ Chunking failed: {msg}"]}


async def stage_build_textnodes(state: WebsearcherState) -> WebsearcherState:
    """
    Convert chunks -> LlamaIndex TextNodes.
    """
    status: List[str] = []
    try:
        if state.get("terminal"):
            return {"ok": True, "status": ["⏭️ TextNodes skipped (nothing to build)"]}

        chunks = state.get("chunks") or []
        if not chunks:
            return {"ok": True, "terminal": True, "status": ["⏭️ TextNodes skipped (no chunks)"]}

        # Lazy import TextNode
        from llama_index.core.schema import TextNode

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

            # stable-ish string id (you can replace with your SHA/xxhash scheme later)
            node_id = f"{doc_name}::chunk::{chunk_id}"

            nodes.append(TextNode(id_=node_id, text=str(c.get("text") or ""), metadata=meta))

        status.append("✅ TextNodes ready")
        status.append(f"🧱 built_nodes={len(nodes)}")

        return {"ok": True, "nodes": nodes, "status": status}

    except Exception as e:
        msg = str(e)
        return {"ok": False, "error_message": msg, "status": [f"❌ TextNodes stage failed: {msg}"]}


async def stage_embedding(state: WebsearcherState) -> WebsearcherState:
    """
    Embed TextNodes using your remote embedding helper (recommended).
    """
    status: List[str] = []
    try:
        if state.get("terminal"):
            return {"ok": True, "status": ["⏭️ Embedding skipped (nothing to embed)"]}

        cfg: WebsearcherConfig = state.get("cfg") or _mk_cfg(state)
        nodes = state.get("nodes") or []
        if not nodes:
            return {"ok": True, "terminal": True, "status": ["⏭️ Embedding skipped (no nodes)"]}

        # Prefer your embed_webscraper.py (you said you created it)
        # Fallback to embed_loader.py if needed.
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

        candidates = [
            "embed_textnodes_async",
            "embed_nodes_async",
            "embed_async",
        ]
        fn_name, embed_fn = _pick_callable(embed_mod, candidates)

        status.append(f"🧠 Embedding via {embed_mod.__name__}.{fn_name}() ...")
        emb = await embed_fn(cfg, nodes)

        status.append("✅ Embedding finished")
        return {"ok": True, "embeddings": emb, "status": status}

    except Exception as e:
        msg = str(e)
        return {"ok": False, "error_message": msg, "status": [f"❌ Embedding failed: {msg}"]}


async def stage_upsert(state: WebsearcherState) -> WebsearcherState:
    """
    Upsert into Qdrant.
    This is schema-dependent, so we delegate to pipeline.py upsert helper.
    """
    status: List[str] = []
    try:
        if state.get("terminal"):
            return {"ok": True, "status": ["⏭️ Upsert skipped (nothing to upsert)"]}

        cfg: WebsearcherConfig = state.get("cfg") or _mk_cfg(state)
        nodes = state.get("nodes") or []
        emb = state.get("embeddings")
        if not nodes:
            return {"ok": True, "terminal": True, "status": ["⏭️ Upsert skipped (no nodes)"]}

        from . import pipeline as pipeline_mod  # ✅ lazy import

        candidates = [
            "upsert_nodes_async",
            "upsert_to_qdrant_async",
            "qdrant_upsert_async",
        ]
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
    Final summary + consistent status lines.
    """
    try:
        files_today = state.get("drive_files_today") or []
        missing = state.get("missing_files") or []
        built_nodes = len(state.get("nodes") or []) if state.get("nodes") is not None else 0

        inserted = int(state.get("inserted") or 0)
        processed = int(state.get("processed") or 0)
        failed = int(state.get("failed") or 0)

        res = state.get("result") or {}
        # keep a normalized summary, even if pipeline returns extra fields
        summary = {
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
            "raw": res,
        }

        status: List[str] = ["🏁 Websearcher pipeline finished"]
        status.append(
            f"📁 today={len(files_today)} missing={len(missing)} processed={processed} "
            f"built_nodes={built_nodes} inserted={inserted} failed={failed}"
        )

        if len(files_today) == 0:
            status.append("🎯 No new PDFs added today - ingestion skipped")
        elif len(missing) == 0:
            status.append("🎯 All new PDFs already in Qdrant - nothing to process")
        elif inserted > 0:
            status.append(f"🎯 Successfully upserted {inserted} chunk(s) into Qdrant")

        return {"ok": True, "result": summary, "status": status}

    except Exception as e:
        msg = str(e)
        return {"ok": False, "error_message": msg, "status": [f"❌ Finalize failed: {msg}"]}


# -----------------------------------------------------------------------------
# Routing helpers
# -----------------------------------------------------------------------------
def _route_ok_or_end(state: WebsearcherState) -> str:
    return "end" if state.get("ok") is False else "ok"


def _route_terminal_or_next(state: WebsearcherState) -> str:
    if state.get("ok") is False:
        return "end"
    if state.get("terminal"):
        return "finalize"
    return "next"


# -----------------------------------------------------------------------------
# Build graph
# -----------------------------------------------------------------------------
builder = StateGraph(WebsearcherState)

builder.add_node("build_config", stage_build_config)
builder.add_node("fetch_drive", stage_fetch_drive_files)
builder.add_node("check_db", stage_check_database)
builder.add_node("llamaparse", stage_agentic_llamaparse)
builder.add_node("chunking", stage_chunking)
builder.add_node("textnodes", stage_build_textnodes)
builder.add_node("embedding", stage_embedding)
builder.add_node("upsert", stage_upsert)
builder.add_node("finalize", stage_finalize)

# START -> build_config
builder.add_edge(START, "build_config")
builder.add_conditional_edges(
    "build_config",
    _route_ok_or_end,
    {"ok": "fetch_drive", "end": END},
)

# fetch_drive -> (terminal?) finalize else check_db
builder.add_conditional_edges(
    "fetch_drive",
    _route_terminal_or_next,
    {"next": "check_db", "finalize": "finalize", "end": END},
)

# check_db -> (terminal?) finalize else llamaparse
builder.add_conditional_edges(
    "check_db",
    _route_terminal_or_next,
    {"next": "llamaparse", "finalize": "finalize", "end": END},
)

# llamaparse -> chunking
builder.add_conditional_edges(
    "llamaparse",
    _route_ok_or_end,
    {"ok": "chunking", "end": END},
)

# chunking -> textnodes
builder.add_conditional_edges(
    "chunking",
    _route_ok_or_end,
    {"ok": "textnodes", "end": END},
)

# textnodes -> embedding
builder.add_conditional_edges(
    "textnodes",
    _route_ok_or_end,
    {"ok": "embedding", "end": END},
)

# embedding -> upsert
builder.add_conditional_edges(
    "embedding",
    _route_ok_or_end,
    {"ok": "upsert", "end": END},
)

# upsert -> finalize
builder.add_conditional_edges(
    "upsert",
    _route_ok_or_end,
    {"ok": "finalize", "end": END},
)

# finalize -> END
builder.add_edge("finalize", END)

graph = builder.compile()
