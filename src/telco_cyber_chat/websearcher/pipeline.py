# src/telco_cyber_chat/websearcher/pipeline.py
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from llama_index.core.schema import TextNode
from qdrant_client import AsyncQdrantClient
from qdrant_client import models as qmodels

from .chunker import chunk_text
from .config import WebsearcherConfig
from .embed_websearcher import embed_textnodes_hybrid
from .websearcher_qdrant import upsert_nodes_hybrid_from_embeddings_async

logger = logging.getLogger(__name__)


# -------------------- small utils --------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _normalize_doc_name(filename: str) -> str:
    # remove extension + normalize casing/whitespace for matching
    return Path(filename).stem.strip().lower()


# -------------------- Drive (sync client, async wrapper) --------------------

def _drive_client(cfg: WebsearcherConfig):
    """Google Drive client (sync). We'll run calls in threadpool."""
    sa_json = os.environ[cfg.drive_sa_json_env]
    creds = service_account.Credentials.from_service_account_info(
        json.loads(sa_json),
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds)


async def _list_pdfs_async(drive, folder_id: str, max_files: int) -> List[dict]:
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

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_list)


async def _download_pdf_bytes_async(drive, file_id: str) -> bytes:
    def _sync_download():
        request = drive.files().get_media(fileId=file_id)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
        return fh.getvalue()

    loop = asyncio.get_running_loop()
    return await loop.run_in_executor(None, _sync_download)


# -------------------- LlamaParse (agentic) --------------------

def _build_llamaparse_parser():
    """
    Lazy import to keep LangSmith startup fast.
    Uses agentic mode (default parse_mode="parse_page_with_agent")
    """
    # Try both common import paths to be robust across versions
    try:
        from llama_cloud_services import LlamaParse  # type: ignore
    except Exception:  # pragma: no cover
        from llama_parse import LlamaParse  # type: ignore

    api_key = (
        os.environ.get("LLAMA_CLOUD_API_KEY")
        or os.environ.get("LLAMAPARSE_API_KEY")
        or ""
    ).strip()
    if not api_key:
        raise RuntimeError(
            "Missing LlamaParse API key. Set LLAMA_CLOUD_API_KEY (recommended) or LLAMAPARSE_API_KEY."
        )

    parse_mode = os.environ.get("LLAMAPARSE_PARSE_MODE", "parse_page_with_agent").strip()
    result_type = os.environ.get("LLAMAPARSE_RESULT_TYPE", "markdown").strip()

    def _env_bool(name: str, default: bool = False) -> bool:
        v = os.environ.get(name)
        if v is None:
            return default
        return v.strip().lower() in ("1", "true", "yes", "y", "on")

    return LlamaParse(
        api_key=api_key,
        result_type=result_type,
        parse_mode=parse_mode,
        high_res_ocr=_env_bool("LLAMAPARSE_HIGH_RES_OCR", True),
        adaptive_long_table=_env_bool("LLAMAPARSE_ADAPTIVE_LONG_TABLE", True),
        outlined_table_extraction=_env_bool("LLAMAPARSE_OUTLINED_TABLES", True),
        output_tables_as_HTML=_env_bool("LLAMAPARSE_TABLES_AS_HTML", True),
        disable_image_extraction=_env_bool("LLAMAPARSE_DISABLE_IMAGE_EXTRACTION", False),
    )


async def _llamaparse_pdf_bytes_to_markdown(parser, pdf_bytes: bytes, *, filename: str) -> str:
    """
    LlamaParse SDK commonly expects a file path, so we write bytes to a temp file.
    """
    tmp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
            f.write(pdf_bytes)
            tmp_path = f.name

        docs = await parser.aload_data(tmp_path)
        parts: List[str] = []
        for d in docs or []:
            t = getattr(d, "text", None)
            if isinstance(t, str) and t.strip():
                parts.append(t.strip())
                continue
            if hasattr(d, "get_content"):
                try:
                    t2 = d.get_content()
                    if isinstance(t2, str) and t2.strip():
                        parts.append(t2.strip())
                except Exception:
                    pass
        return "\n\n".join(parts).strip()
    finally:
        if tmp_path:
            try:
                os.unlink(tmp_path)
            except Exception:
                pass


# -------------------- Qdrant inventory helpers --------------------

async def _collection_exists(client: AsyncQdrantClient, collection: str) -> bool:
    try:
        await client.get_collection(collection)
        return True
    except Exception:
        return False


async def _fetch_unique_doc_names_from_qdrant(
    client: AsyncQdrantClient,
    collection: str,
    *,
    data_type: str,
    batch_size: int = 512,
) -> Set[str]:
    """
    Scroll and collect UNIQUE payload['doc_name'] for a given data_type.
    """
    if not await _collection_exists(client, collection):
        return set()

    out: Set[str] = set()
    next_offset = None

    flt = qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="data_type",
                match=qmodels.MatchValue(value=data_type),
            )
        ]
    )

    while True:
        points, next_offset = await client.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=batch_size,
            offset=next_offset,
            with_payload=["doc_name"],
            with_vectors=False,
        )

        for p in points:
            payload = getattr(p, "payload", None) or {}
            dn = payload.get("doc_name")
            if isinstance(dn, str) and dn.strip():
                out.add(dn.strip().lower())

        if next_offset is None:
            break

    return out


# -------------------- Node building --------------------

def _build_nodes_for_doc(
    *,
    doc_text: str,
    doc_name: str,
    data_type: str,
    chunk_size: int,
    chunk_overlap: int,
    drive_file_id: str,
    drive_file_name: str,
) -> List[TextNode]:
    scraped_date = _utc_now_iso()
    chunks = chunk_text(doc_text, chunk_size=chunk_size, overlap=chunk_overlap)

    nodes: List[TextNode] = []
    for i, ch in enumerate(chunks):
        nodes.append(
            TextNode(
                text=ch,
                metadata={
                    "doc_name": doc_name,  # normalized doc_name
                    "data_type": data_type,
                    "scraped_date": scraped_date,
                    "chunk_index": i,
                    "drive_file_id": drive_file_id,
                    "drive_file_name": drive_file_name,
                },
            )
        )
    return nodes


# -------------------- MAIN PIPELINE --------------------

async def ingest_drive_folder_async(
    cfg: WebsearcherConfig,
    *,
    drive_folder_id: Optional[str] = None,
    max_files: Optional[int] = None,
) -> Dict:
    folder_id = drive_folder_id or os.getenv(cfg.drive_folder_id_env)
    if not folder_id:
        raise RuntimeError(f"Missing Drive folder id: env {cfg.drive_folder_id_env} is not set")

    max_files = int(max_files or cfg.max_files)

    qdrant_url = os.environ[cfg.qdrant_url_env]
    qdrant_key = os.environ.get(cfg.qdrant_api_key_env)
    collection = cfg.collection or os.environ.get(cfg.qdrant_collection_env, "telco_whitepapers")

    data_type = str(getattr(cfg, "data_type", None) or "unstructured").strip() or "unstructured"

    qdrant = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_key)
    drive = _drive_client(cfg)
    parser = _build_llamaparse_parser()

    try:
        # 1) list PDFs in Drive
        files = await _list_pdfs_async(drive, folder_id, max_files=max_files)

        # 2) normalize Drive doc_names + detect duplicates
        drive_map: Dict[str, dict] = {}
        dup_doc_names: Set[str] = set()

        for f in files:
            name = (f.get("name") or "").strip()
            if not name:
                continue
            dn = _normalize_doc_name(name)
            if dn in drive_map:
                dup_doc_names.add(dn)
                continue
            drive_map[dn] = f

        drive_doc_names = set(drive_map.keys())

        # 3) fetch UNIQUE qdrant doc_names filtered by data_type
        qdrant_doc_names = await _fetch_unique_doc_names_from_qdrant(
            qdrant, collection, data_type=data_type
        )

        # 4) inventory + missing
        missing_doc_names = sorted(drive_doc_names - qdrant_doc_names)
        existing_doc_names = sorted(drive_doc_names & qdrant_doc_names)

        logger.info("📦 Qdrant unique doc_name (data_type=%s): %d", data_type, len(qdrant_doc_names))
        logger.info("🗂️ Drive unique PDFs (by doc_name): %d", len(drive_doc_names))
        logger.info("🆕 Missing docs to ingest: %d", len(missing_doc_names))
        if dup_doc_names:
            logger.warning("⚠️ Duplicate doc_names in Drive folder (same stem): %d", len(dup_doc_names))

        # Only process missing
        to_process = [drive_map[dn] for dn in missing_doc_names]

        if not to_process:
            return {
                "folder_id": folder_id,
                "collection": collection,
                "data_type": data_type,
                "files_seen": len(files),
                "drive_unique": len(drive_doc_names),
                "qdrant_unique": len(qdrant_doc_names),
                "missing": 0,
                "duplicates_on_drive": len(dup_doc_names),
                "skipped_existing": len(existing_doc_names),
                "processed": 0,
                "built_nodes": 0,
                "inserted": 0,
                "ok": True,
            }

        # Concurrency controls
        dl_sem = asyncio.Semaphore(int(os.environ.get("DRIVE_DOWNLOAD_CONCURRENCY", "4")))
        parse_sem = asyncio.Semaphore(int(os.environ.get("LLAMAPARSE_CONCURRENCY", "2")))

        processed_docs = 0
        built_nodes = 0
        all_nodes: List[TextNode] = []

        async def process_one(f: dict) -> List[TextNode]:
            nonlocal processed_docs

            pdf_name = (f.get("name") or "").strip()
            file_id = (f.get("id") or "").strip()
            if not pdf_name or not file_id:
                return []

            norm_dn = _normalize_doc_name(pdf_name)

            try:
                # Download
                async with dl_sem:
                    pdf_bytes = await _download_pdf_bytes_async(drive, file_id)

                # Parse with LlamaParse (agentic)
                async with parse_sem:
                    md_text = await _llamaparse_pdf_bytes_to_markdown(parser, pdf_bytes, filename=pdf_name)

                if not md_text.strip():
                    logger.warning("Empty parse output for %s", pdf_name)
                    return []

                nodes = _build_nodes_for_doc(
                    doc_text=md_text,
                    doc_name=norm_dn,
                    data_type=data_type,
                    chunk_size=int(cfg.chunk_size),
                    chunk_overlap=int(cfg.chunk_overlap),
                    drive_file_id=file_id,
                    drive_file_name=pdf_name,
                )

                processed_docs += 1
                return nodes

            except Exception as e:
                logger.exception("Failed processing %s: %s", pdf_name, e)
                return []

        # Run concurrently but do not fail whole batch if one doc fails
        results = await asyncio.gather(*[process_one(f) for f in to_process], return_exceptions=False)

        for nodes in results:
            all_nodes.extend(nodes)
            built_nodes += len(nodes)

        if not all_nodes:
            return {
                "folder_id": folder_id,
                "collection": collection,
                "data_type": data_type,
                "files_seen": len(files),
                "drive_unique": len(drive_doc_names),
                "qdrant_unique": len(qdrant_doc_names),
                "missing": len(missing_doc_names),
                "duplicates_on_drive": len(dup_doc_names),
                "skipped_existing": len(existing_doc_names),
                "processed": processed_docs,
                "built_nodes": 0,
                "inserted": 0,
                "ok": True,
            }

        # Remote embed
        emb_map = await embed_textnodes_hybrid(
            all_nodes,
            concurrency=int(os.environ.get("EMBED_CONCURRENCY", "2")),
            batch_size=int(os.environ.get("EMBED_BATCH_SIZE", "20")),
            retry_count=int(os.environ.get("EMBED_RETRY_COUNT", "3")),
        )

        # Upsert using the dedicated upsert-only module
        inserted = await upsert_nodes_hybrid_from_embeddings_async(
            nodes=all_nodes,
            emb_map=emb_map,
            client=qdrant,
            collection=collection,
        )

        return {
            "folder_id": folder_id,
            "collection": collection,
            "data_type": data_type,
            "files_seen": len(files),
            "drive_unique": len(drive_doc_names),
            "qdrant_unique": len(qdrant_doc_names),
            "missing": len(missing_doc_names),
            "duplicates_on_drive": len(dup_doc_names),
            "skipped_existing": len(existing_doc_names),
            "processed": processed_docs,
            "built_nodes": built_nodes,
            "inserted": int(inserted),
            "ok": True,
            "missing_preview": missing_doc_names[:30],
        }

    finally:
        await qdrant.close()
