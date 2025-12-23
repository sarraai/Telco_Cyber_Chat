# src/telco_cyber_chat/websearcher/pipeline.py
from __future__ import annotations

import io
import json
import os
import asyncio
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from llama_index.core.schema import TextNode
from pypdf import PdfReader

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from qdrant_client import AsyncQdrantClient  # ✅ Changed to Async
from qdrant_client import models as qmodels

from .config import WebsearcherConfig
from .chunker import chunk_text
from .websearcher_qdrant import upsert_nodes_bgem3_hybrid_async, _get_bgem3


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_doc_name(filename: str) -> str:
    return Path(filename).stem.strip()


def _drive_client(cfg: WebsearcherConfig):
    """Keep sync - Google API client doesn't have good async support"""
    sa_json = os.environ[cfg.drive_sa_json_env]
    creds = service_account.Credentials.from_service_account_info(
        json.loads(sa_json),
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds)


async def _list_pdfs_async(drive, folder_id: str, max_files: int) -> List[dict]:
    """Run sync Drive API calls in thread pool"""
    def _sync_list():
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
    
    # Run in thread pool to avoid blocking event loop
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_list)


async def _download_pdf_bytes_async(drive, file_id: str) -> bytes:
    """Run sync Drive download in thread pool"""
    def _sync_download():
        request = drive.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return fh.getvalue()
    
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, _sync_download)


def _pdf_bytes_to_text(pdf_bytes: bytes) -> str:
    """Keep sync - fast CPU-bound operation"""
    reader = PdfReader(io.BytesIO(pdf_bytes))
    parts: List[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            parts.append(t)
    return "\n\n".join(parts).strip()


async def _exists_in_qdrant_by_doc_name_and_data_type(
    client: AsyncQdrantClient,  # ✅ Async client
    collection: str,
    *,
    doc_name: str,
    data_type: str,
) -> bool:
    """Check if document exists - now async"""
    flt = qmodels.Filter(
        must=[
            qmodels.FieldCondition(key="data_type", match=qmodels.MatchValue(value=data_type)),
            qmodels.FieldCondition(key="doc_name", match=qmodels.MatchValue(value=doc_name)),
        ]
    )
    pts, _ = await client.scroll(  # ✅ await
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
    """Keep sync - fast CPU operation"""
    scraped_date = _utc_now_iso()
    chunks = chunk_text(doc_text, chunk_size=chunk_size, overlap=chunk_overlap)

    nodes: List[TextNode] = []
    for i, ch in enumerate(chunks):
        nodes.append(
            TextNode(
                text=ch,
                metadata={
                    "doc_name": doc_name,
                    "data_type": data_type,
                    "scraped_date": scraped_date,
                    "chunk_index": i,
                },
            )
        )
    return nodes


async def ingest_drive_folder_async(  # ✅ Now async
    cfg: WebsearcherConfig,
    *,
    drive_folder_id: Optional[str] = None,
    max_files: Optional[int] = None,
) -> Dict:
    """Main ingestion pipeline - fully async"""
    folder_id = drive_folder_id or os.getenv(cfg.drive_folder_id_env)
    if not folder_id:
        raise RuntimeError(
            f"Missing Drive folder id: env {cfg.drive_folder_id_env} is not set"
        )

    max_files = int(max_files or cfg.max_files)

    qdrant_url = os.environ[cfg.qdrant_url_env]
    qdrant_key = os.environ.get(cfg.qdrant_api_key_env)
    collection = cfg.collection or os.environ.get(cfg.qdrant_collection_env, "telco_whitepapers")

    # ✅ Use AsyncQdrantClient
    qdrant = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_key)
    
    # Drive client stays sync (Google API limitation)
    drive = _drive_client(cfg)

    # ✅ Await async list
    files = await _list_pdfs_async(drive, folder_id, max_files=max_files)

    skipped = 0
    processed = 0
    built_nodes = 0
    all_nodes: List[TextNode] = []

    data_type = getattr(cfg, "data_type", None) or "unstructured"
    data_type = str(data_type).strip() or "unstructured"

    # Process files concurrently (with semaphore to limit parallel downloads)
    semaphore = asyncio.Semaphore(5)  # Max 5 concurrent downloads
    
    async def process_file(f: dict):
        nonlocal skipped, processed, built_nodes
        
        async with semaphore:
            pdf_name = (f.get("name") or "").strip()
            file_id = (f.get("id") or "").strip()
            if not pdf_name or not file_id:
                return []

            doc_name = _normalize_doc_name(pdf_name)

            # ✅ Await existence check
            if await _exists_in_qdrant_by_doc_name_and_data_type(
                qdrant, collection, doc_name=doc_name, data_type=data_type
            ):
                skipped += 1
                return []

            # ✅ Await download
            pdf_bytes = await _download_pdf_bytes_async(drive, file_id)
            
            # Run CPU-bound PDF parsing in thread pool
            loop = asyncio.get_event_loop()
            text = await loop.run_in_executor(None, _pdf_bytes_to_text, pdf_bytes)
            
            if not text:
                return []

            nodes = _build_nodes_for_doc(
                doc_text=text,
                doc_name=doc_name,
                data_type=data_type,
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
            )

            processed += 1
            return nodes

    # Process all files concurrently
    results = await asyncio.gather(*[process_file(f) for f in files])
    
    # Flatten results
    for nodes in results:
        all_nodes.extend(nodes)
        built_nodes += len(nodes)

    if not all_nodes:
        await qdrant.close()  # ✅ Close async client
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

    # ✅ Async embed + upsert
    model = _get_bgem3("BAAI/bge-m3")
    inserted = await upsert_nodes_bgem3_hybrid_async(
        nodes=all_nodes,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_key,
        collection=collection,
        model=model,
        client=qdrant,
    )

    await qdrant.close()  # ✅ Clean up

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
