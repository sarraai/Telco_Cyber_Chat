from __future__ import annotations

import io
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from llama_index.core.schema import TextNode
from pypdf import PdfReader

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

from .config import WebsearcherConfig
from .chunker import chunk_text
from .websearcher_qdrant import upsert_nodes_bgem3_hybrid


# -------------------- helpers --------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

_UUID_NS = uuid.UUID("5b2f0b2c-7f55-4a3e-9ac2-2e2f3f3f5b4c")

def _stable_point_id(vendor: str, source_id: str, chunk_index: int) -> str:
    return str(uuid.uuid5(_UUID_NS, f"{vendor}|{source_id}|{chunk_index}"))

def _extract_header_value(text: str, key: str) -> Optional[str]:
    prefix = f"{key}:"
    for line in text.splitlines()[:25]:
        if line.startswith(prefix):
            return line.split(":", 1)[1].strip()
    return None


# -------------------- Drive client --------------------

def _drive_client() :
    sa_json = os.environ["GDRIVE_SA_JSON"]
    creds = service_account.Credentials.from_service_account_info(
        __import__("json").loads(sa_json),
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds)

def list_drive_pdfs(folder_id: str, max_files: int = 200) -> List[dict]:
    drive = _drive_client()
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

def download_drive_pdf_bytes(file_id: str) -> bytes:
    drive = _drive_client()
    request = drive.files().get_media(fileId=file_id)

    fh = io.BytesIO()
    downloader = MediaIoBaseDownload(fh, request)
    done = False
    while not done:
        _, done = downloader.next_chunk()
    return fh.getvalue()


# -------------------- PDF -> Text --------------------

def pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts: List[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            parts.append(t)
    return "\n\n".join(parts).strip()


# -------------------- Nodes --------------------

def build_nodes_from_text(
    text: str,
    source_id: str,
    vendor: str,
    doc_name: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[TextNode]:
    scraped_date = _utc_now_iso()

    # Put doc_name into the header so upsert can store it in payload (for name-based dedupe)
    header = (
        f"vendor: {vendor}\n"
        f"doc_name: {doc_name}\n"
        f"url: {source_id}\n"
        f"scraped_date: {scraped_date}\n"
    )

    chunks = chunk_text(header + "\n" + text, chunk_size=chunk_size, overlap=chunk_overlap)

    nodes: List[TextNode] = []
    for i, ch in enumerate(chunks):
        txt = header + f"chunk_index: {i}\n\n{ch}"
        nodes.append(TextNode(text=txt, metadata={"url": source_id}))
    return nodes


# -------------------- Qdrant existence check (by name) --------------------

def exists_in_qdrant_by_name(client: QdrantClient, collection: str, pdf_name: str) -> bool:
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


# -------------------- Main: Drive -> (skip by name) -> nodes -> upsert --------------------

def ingest_drive_folder(cfg: WebsearcherConfig) -> Dict:
    folder_id = os.getenv(cfg.drive_folder_id_env)
    if not folder_id:
        raise RuntimeError(f"Missing env var: {cfg.drive_folder_id_env}")

    qdrant_url = os.environ["QDRANT_URL"]
    qdrant_key = os.environ.get("QDRANT_API_KEY")
    collection = cfg.collection or os.environ.get("QDRANT_COLLECTION", "telco_whitepapers")

    qdrant = QdrantClient(url=qdrant_url, api_key=qdrant_key)

    files = list_drive_pdfs(folder_id, max_files=cfg.max_files)

    skipped = 0
    processed = 0
    built_nodes = 0
    inserted_total = 0

    for f in files:
        name = f.get("name") or ""
        file_id = f.get("id") or ""
        if not name or not file_id:
            continue

        # ✅ skip if name already exists in Qdrant
        if exists_in_qdrant_by_name(qdrant, collection, name):
            skipped += 1
            continue

        pdf_bytes = download_drive_pdf_bytes(file_id)
        text = pdf_bytes_to_text(pdf_bytes)
        if not text:
            continue

        # stable-ish source id (even though dedupe is name-only)
        source_id = f"gdrive:{folder_id}:{name}"

        nodes = build_nodes_from_text(
            text=text,
            source_id=source_id,
            vendor=cfg.vendor,
            doc_name=name,
            chunk_size=cfg.chunk_size,
            chunk_overlap=cfg.chunk_overlap,
        )

        built_nodes += len(nodes)

        inserted = upsert_nodes_bgem3_hybrid(
            nodes=nodes,
            qdrant_url=qdrant_url,
            qdrant_api_key=qdrant_key,
            collection=collection,
            vendor=cfg.vendor,
        )

        inserted_total += int(inserted)
        processed += 1

    return {
        "folder_id": folder_id,
        "collection": collection,
        "files_seen": len(files),
        "skipped": skipped,
        "processed": processed,
        "built_nodes": built_nodes,
        "inserted": inserted_total,
        "ok": True,
    }
