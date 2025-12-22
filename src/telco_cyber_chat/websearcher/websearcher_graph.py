from __future__ import annotations

import io
import json
import operator
import os
from datetime import datetime, timezone
from typing import Annotated, List, Optional
from typing_extensions import TypedDict

from langgraph.graph import StateGraph, START, END
from llama_index.core.schema import TextNode

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from pypdf import PdfReader

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from .config import WebsearcherConfig
from .chunker import chunk_text
from .websearcher_qdrant import upsert_nodes_bgem3_hybrid


class WebsearcherState(TypedDict, total=False):
    # ------- inputs -------
    drive_folder_id: Optional[str]      # default: env GDRIVE_FOLDER_ID
    max_files: int                      # default: 200
    vendor: str                         # default: "websearcher"
    collection: Optional[str]           # overrides QDRANT_COLLECTION

    # chunking overrides (optional)
    chunk_size: int
    chunk_overlap: int

    # ------- outputs / logs -------
    status: Annotated[List[str], operator.add]
    files_seen: int
    skipped: int
    processed: int
    built_nodes: int
    inserted: int
    ok: bool
    error_message: Optional[str]


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _drive_client():
    sa_json = os.environ["GDRIVE_SA_JSON"]
    creds = service_account.Credentials.from_service_account_info(
        json.loads(sa_json),
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds)


def _list_pdfs(drive, folder_id: str, max_files: int) -> List[dict]:
    q = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    files: List[dict] = []
    page_token = None

    while True:
        resp = drive.files().list(
            q=q,
            fields="nextPageToken, files(id,name,modifiedTime)",
            pageToken=page_token,
            pageSize=200,
        ).execute()
        files.extend(resp.get("files", []))
        page_token = resp.get("nextPageToken")
        if not page_token or len(files) >= max_files:
            break

    return files[:max_files]


def _download_pdf_bytes(drive, file_id: str) -> bytes:
    request = drive.files().get_media(fileId=file_id)
    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return fh.getvalue()


def _pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts: List[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            parts.append(t)
    return "\n\n".join(parts).strip()


def _cfg_from_state(state: WebsearcherState) -> WebsearcherConfig:
    return WebsearcherConfig(
        seed_urls=[],  # unused in Drive strategy
        vendor=str(state.get("vendor") or "websearcher"),
        collection=state.get("collection"),
        chunk_size=int(state.get("chunk_size") or 2000),
        chunk_overlap=int(state.get("chunk_overlap") or 200),
    )


def _exists_in_qdrant_by_name(client: QdrantClient, collection: str, pdf_name: str) -> bool:
    # Name-only dedupe: requires payload["doc_name"] to exist.
    flt = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="doc_name",
                match=qmodels.MatchValue(value=pdf_name),
            )
        ]
    )
    pts, _ = client.scroll(
        collection_name=collection,
        scroll_filter=flt,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    return len(pts) > 0


def stage_build_nodes_from_drive(state: WebsearcherState) -> WebsearcherState:
    try:
        cfg = _cfg_from_state(state)

        folder_id = state.get("drive_folder_id") or os.environ.get("GDRIVE_FOLDER_ID")
        if not folder_id:
            raise ValueError("Missing drive_folder_id (input) and GDRIVE_FOLDER_ID (env).")

        max_files = int(state.get("max_files") or 200)

        qdrant_url = os.environ["QDRANT_URL"]
        qdrant_key = os.environ.get("QDRANT_API_KEY")
        collection = cfg.collection or os.environ.get("QDRANT_COLLECTION", "telco_whitepapers")
        qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key)

        drive = _drive_client()
        files = _list_pdfs(drive, folder_id, max_files=max_files)

        nodes_all: List[TextNode] = []
        skipped = 0
        processed = 0
        now = _utc_now_iso()

        for f in files:
            pdf_name = f.get("name") or ""
            file_id = f.get("id") or ""
            modified_time = f.get("modifiedTime") or ""

            if not pdf_name or not file_id:
                continue

            # ✅ skip if already ingested by name
            if _exists_in_qdrant_by_name(qdrant, collection, pdf_name):
                skipped += 1
                continue

            # Download + extract
            pdf_bytes = _download_pdf_bytes(drive, file_id)
            text = _pdf_bytes_to_text(pdf_bytes)
            if not text:
                # no text extracted → skip quietly (or log if you want)
                continue

            # Use a stable "url" field even though dedupe is name-only
            source_url = f"gdrive:{folder_id}:{pdf_name}"

            # Put doc_name in header so upsert can store it in payload
            header = (
                f"vendor: {cfg.vendor}\n"
                f"doc_name: {pdf_name}\n"
                f"drive_file_id: {file_id}\n"
                f"drive_modified_time: {modified_time}\n"
                f"url: {source_url}\n"
                f"scraped_date: {now}\n"
            )

            chunks = chunk_text(header + "\n" + text, chunk_size=cfg.chunk_size, overlap=cfg.chunk_overlap)
            for i, ch in enumerate(chunks):
                txt = header + f"chunk_index: {i}\n\n{ch}"
                nodes_all.append(TextNode(text=txt, metadata={"url": source_url}))

            processed += 1

        return {
            "status": [
                f"📄 Drive files seen: {len(files)}",
                f"⏭️ Skipped by name: {skipped}",
                f"✅ New PDFs processed: {processed}",
                f"🧩 Built nodes: {len(nodes_all)}",
            ],
            "files_seen": len(files),
            "skipped": skipped,
            "processed": processed,
            "built_nodes": len(nodes_all),
            "nodes": nodes_all,  # internal only
            "ok": True,
            "collection": collection,
        }

    except Exception as e:
        return {"status": [f"❌ build_nodes_from_drive failed: {e}"], "ok": False, "error_message": str(e)}


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
    state.pop("nodes", None)
    return state


builder = StateGraph(WebsearcherState)
builder.add_node("build_nodes_from_drive", stage_build_nodes_from_drive)
builder.add_node("upsert", stage_upsert)
builder.add_node("finalize", _strip_internal)

builder.add_edge(START, "build_nodes_from_drive")
builder.add_edge("build_nodes_from_drive", "upsert")
builder.add_edge("upsert", "finalize")
builder.add_edge("finalize", END)

graph = builder.compile()
