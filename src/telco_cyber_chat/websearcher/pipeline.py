# src/telco_cyber_chat/websearcher/pipeline.py
from __future__ import annotations

import io
import json
import os
from datetime import datetime, timezone
from pathlib import Path
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


def _normalize_doc_name(filename: str) -> str:
    # "My Paper.pdf" -> "My Paper"
    return Path(filename).stem.strip()


def _drive_client(cfg: WebsearcherConfig):
    sa_json = os.environ[cfg.drive_sa_json_env]
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
            fields="nextPageToken, files(id,name)",
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


def _exists_in_qdrant_by_doc_name_and_data_type(
    client: QdrantClient,
    collection: str,
    *,
    doc_name: str,
    data_type: str,
) -> bool:
    flt = qmodels.Filter(
        must=[
            qmodels.FieldCondition(key="data_type", match=qmodels.MatchValue(value=data_type)),
            qmodels.FieldCondition(key="doc_name", match=qmodels.MatchValue(value=doc_name)),
        ]
    )
    pts, _ = client.scroll(
        collection_name=collection,
        scroll_filter=flt,
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return len(pts) > 0


def _build_nodes_for_doc(
    *,
    doc_text: str,
    doc_name: str,
    data_type: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[TextNode]:
    scraped_date = _utc_now_iso()
    chunks = chunk_text(doc_text, chunk_size=chunk_size, overlap=chunk_overlap)

    nodes: List[TextNode] = []
    for i, ch in enumerate(chunks):
        nodes.append(
            TextNode(
                text=ch,  # ✅ clean chunk text only
                metadata={
                    "doc_name": doc_name,
                    "data_type": data_type,
                    "scraped_date": scraped_date,
                    "chunk_index": i,  # ✅ deterministic point IDs downstream
                },
            )
        )
    return nodes


def ingest_drive_folder(
    cfg: WebsearcherConfig,
    *,
    drive_folder_id: Optional[str] = None,
    max_files: Optional[int] = None,
) -> Dict:
    folder_id = drive_folder_id or os.getenv(cfg.drive_folder_id_env)
    if not folder_id:
        raise RuntimeError(
            f"Missing Drive folder id: env {cfg.drive_folder_id_env} is not set and no override provided."
        )

    max_files = int(max_files or cfg.max_files)

    qdrant_url = os.environ[cfg.qdrant_url_env]
    qdrant_key = os.environ.get(cfg.qdrant_api_key_env)
    collection = cfg.collection or os.environ.get(cfg.qdrant_collection_env, "telco_whitepapers")

    qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key)
    drive = _drive_client(cfg)

    files = _list_pdfs(drive, folder_id, max_files=max_files)

    skipped = 0
    processed = 0
    built_nodes = 0
    all_nodes: List[TextNode] = []

    data_type = getattr(cfg, "data_type", None) or getattr(cfg, "doc_type", None) or "unstructured"
    data_type = str(data_type).strip() or "unstructured"

    for f in files:
        pdf_name = (f.get("name") or "").strip()
        file_id = (f.get("id") or "").strip()
        if not pdf_name or not file_id:
            continue

        doc_name = _normalize_doc_name(pdf_name)  # ✅ no ".pdf"

        # ✅ Skip if doc exists at least once
        if _exists_in_qdrant_by_doc_name_and_data_type(
            qdrant, collection, doc_name=doc_name, data_type=data_type
        ):
            skipped += 1
            continue

        pdf_bytes = _download_pdf_bytes(drive, file_id)
        text = _pdf_bytes_to_text(pdf_bytes)
        if not text:
            continue

        nodes = _build_nodes_for_doc(
            doc_text=text,
            doc_name=doc_name,
            data_type=data_type,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )

        all_nodes.extend(nodes)
        built_nodes += len(nodes)
        processed += 1

    if not all_nodes:
        return {
            "folder_id": folder_id,
            "collection": collection,
            "files_seen": len(files),
            "skipped": skipped,
            "processed": processed,
            "built_nodes": 0,
            "inserted": 0,
            "ok": True,
        }

    # Embed + upsert once (faster)
    model = _get_bgem3("BAAI/bge-m3")
    inserted = upsert_nodes_bgem3_hybrid(
        nodes=all_nodes,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_key,
        collection=collection,
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
