from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional

from llama_index.core.schema import TextNode
from pypdf import PdfReader

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from .config import WebsearcherConfig
from .chunker import chunk_text
from .websearcher_qdrant import upsert_nodes_bgem3_hybrid, _get_bgem3


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


def _exists_in_qdrant_by_name(client: QdrantClient, collection: str, pdf_name: str) -> bool:
    flt = qmodels.Filter(
        must=[qmodels.FieldCondition(key="doc_name", match=qmodels.MatchValue(value=pdf_name))]
    )
    pts, _ = client.scroll(
        collection_name=collection,
        scroll_filter=flt,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    return len(pts) > 0


def _build_nodes_for_pdf(
    *,
    text: str,
    source_id: str,
    vendor: str,
    doc_name: str,
    drive_file_id: str,
    drive_modified_time: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[TextNode]:
    scraped_date = _utc_now_iso()

    header = (
        f"vendor: {vendor}\n"
        f"doc_name: {doc_name}\n"
        f"drive_file_id: {drive_file_id}\n"
        f"drive_modified_time: {drive_modified_time}\n"
        f"url: {source_id}\n"
        f"scraped_date: {scraped_date}\n"
    )

    chunks = chunk_text(header + "\n" + text, chunk_size=chunk_size, overlap=chunk_overlap)
    nodes: List[TextNode] = []

    for i, ch in enumerate(chunks):
        txt = header + f"chunk_index: {i}\n\n{ch}"
        nodes.append(TextNode(text=txt, metadata={"url": source_id}))

    return nodes


def ingest_drive_folder(
    cfg: WebsearcherConfig,
    *,
    drive_folder_id: Optional[str] = None,
    max_files: Optional[int] = None,
) -> Dict:
    folder_id = drive_folder_id or os.getenv(cfg.drive_folder_id_env)
    if not folder_id:
        raise RuntimeError(f"Missing Drive folder id: env {cfg.drive_folder_id_env} is not set and no override provided.")

    max_files = int(max_files or cfg.max_files)

    qdrant_url = os.environ["QDRANT_URL"]
    qdrant_key = os.environ.get("QDRANT_API_KEY")
    collection = cfg.collection or os.environ.get("QDRANT_COLLECTION", "telco_whitepapers")

    qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    drive = _drive_client()

    files = _list_pdfs(drive, folder_id, max_files=max_files)

    skipped = 0
    processed = 0
    built_nodes = 0
    all_nodes: List[TextNode] = []

    for f in files:
        pdf_name = (f.get("name") or "").strip()
        file_id = (f.get("id") or "").strip()
        modified_time = (f.get("modifiedTime") or "").strip()

        if not pdf_name or not file_id:
            continue

        # ✅ Skip if already ingested by name
        if _exists_in_qdrant_by_name(qdrant, collection, pdf_name):
            skipped += 1
            continue

        pdf_bytes = _download_pdf_bytes(drive, file_id)
        text = _pdf_bytes_to_text(pdf_bytes)
        if not text:
            continue

        # Stable source id (even though dedupe is name-only)
        source_id = f"gdrive:{folder_id}:{pdf_name}"

        nodes = _build_nodes_for_pdf(
            text=text,
            source_id=source_id,
            vendor=cfg.vendor,
            doc_name=pdf_name,
            drive_file_id=file_id,
            drive_modified_time=modified_time,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )

        all_nodes.extend(nodes)
        built_nodes += len(nodes)
        processed += 1

    # Embed + upsert once (faster than per-PDF)
    model = _get_bgem3("BAAI/bge-m3")
    inserted = upsert_nodes_bgem3_hybrid(
        nodes=all_nodes,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_key,
        collection=collection,
        vendor=cfg.vendor,
        model=model,
        client=qdrant,
    )

    return {
        "folder_id": folder_id,
        "collection": collection,
        "files_seen": len(files),
        "skipped": skipped,
        "processed": processed,
        "built_nodes": built_nodes,
        "inserted": int(inserted),
        "ok": True,
    }
