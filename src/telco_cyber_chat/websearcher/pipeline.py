# src/telco_cyber_chat/websearcher/pipeline.py
from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import ssl
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
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

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


def _get_today_start_utc() -> datetime:
    """Get today's date at 00:00:00 UTC"""
    return datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)


# -------------------- Drive (sync client, async wrapper) --------------------

def _drive_client(cfg: WebsearcherConfig):
    sa_json = os.environ[cfg.drive_sa_json_env]
    creds = service_account.Credentials.from_service_account_info(
        json.loads(sa_json),
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    return build("drive", "v3", credentials=creds)


async def _list_pdfs_async(drive, folder_id: str, max_files: int) -> List[dict]:
    """List PDFs with their creation date"""
    def _sync_list():
        q = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
        files: List[dict] = []
        page_token = None

        while True:
            resp = drive.files().list(
                q=q,
                fields="nextPageToken, files(id, name, createdTime, modifiedTime)",  # ✅ Added createdTime
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


# ✅ NEW: Download with retry logic and SSL error handling
@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=10),
    retry=retry_if_exception_type(ssl.SSLError)
)
async def _download_pdf_bytes_async(drive, file_id: str, dl_sem: asyncio.Semaphore) -> bytes:
    """Download PDF with retry logic and concurrency control"""
    async with dl_sem:  # ✅ Limit concurrent downloads
        def _sync_download():
            try:
                request = drive.files().get_media(fileId=file_id)
                fh = io.BytesIO()
                downloader = MediaIoBaseDownload(fh, request, chunksize=1024*1024)
                done = False
                while not done:
                    _, done = downloader.next_chunk()
                return fh.getvalue()
            except ssl.SSLError as e:
                logger.warning(f"SSL Error downloading {file_id}: {e}")
                raise  # Let retry handle it
            except Exception as e:
                logger.error(f"Error downloading {file_id}: {e}")
                raise

        loop = asyncio.get_running_loop()
        # ✅ Add timeout per file
        return await asyncio.wait_for(
            loop.run_in_executor(None, _sync_download),
            timeout=180  # 3 minute timeout per file
        )


# -------------------- LlamaParse (agentic) --------------------

def _build_llamaparse_parser():
    """
    IMPORTANT:
    - ONLY uses llama-cloud-services (no llama_parse fallback).
    - Lazy import to keep LangSmith startup fast.
    """
    try:
        from llama_cloud_services import LlamaParse  # type: ignore
    except Exception as e:
        raise RuntimeError(
            "LlamaParse SDK not installed. Add `llama-cloud-services>=0.5.0` to requirements.txt."
        ) from e

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
    LlamaParse expects a file path, so we write bytes to a temp file.
    More defensive than your earlier version: also checks get_content().
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
    created_time: str,
) -> List[TextNode]:
    scraped_date = _utc_now_iso()
    chunks = chunk_text(doc_text, chunk_size=chunk_size, overlap=chunk_overlap)

    nodes: List[TextNode] = []
    for i, ch in enumerate(chunks):
        nodes.append(
            TextNode(
                text=ch,
                metadata={
                    "doc_name": doc_name,  # normalized stem
                    "data_type": data_type,
                    "scraped_date": scraped_date,
                    "created_date": created_time,  # ✅ Added creation date
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
        raise RuntimeError(
            f"Missing Drive folder id: env {cfg.drive_folder_id_env} is not set"
        )

    max_files = int(max_files or cfg.max_files)

    qdrant_url = os.environ[cfg.qdrant_url_env]
    qdrant_key = os.environ.get(cfg.qdrant_api_key_env)
    collection = cfg.collection or os.environ.get(cfg.qdrant_collection_env, "telco_whitepapers")

    data_type = str(getattr(cfg, "data_type", None) or "unstructured").strip() or "unstructured"

    # Respect your deployment env vars
    dense_name = (os.environ.get("DENSE_FIELD") or "dense").strip()
    sparse_name = (os.environ.get("SPARSE_FIELD") or "sparse").strip()

    qdrant = AsyncQdrantClient(url=qdrant_url, api_key=qdrant_key)

    # NOTE: Drive + LlamaParse are sync-ish / heavy objects; build once per run.
    drive = _drive_client(cfg)
    parser = _build_llamaparse_parser()

    # ✅ Get today's date
    today_start = _get_today_start_utc()
    logger.info(f"🔍 Only processing PDFs created on or after: {today_start.isoformat()}")

    try:
        # 1) list PDFs in Drive with creation dates
        files = await _list_pdfs_async(drive, folder_id, max_files=max_files)
        logger.info(f"📁 Total PDFs in Drive folder: {len(files)}")

        # ✅ 2) Filter for PDFs created TODAY only
        files_created_today = []
        files_created_past = []
        
        for f in files:
            name = (f.get("name") or "").strip()
            if not name:
                continue
                
            created_str = f.get("createdTime")
            if not created_str:
                logger.warning(f"⚠️ No createdTime for {name}, skipping")
                continue
            
            try:
                # Parse ISO format: "2025-12-23T10:30:00.000Z"
                created_time = datetime.fromisoformat(created_str.replace('Z', '+00:00'))
                
                if created_time >= today_start:
                    files_created_today.append(f)
                else:
                    files_created_past.append(f)
            except Exception as e:
                logger.warning(f"⚠️ Could not parse createdTime for {name}: {e}")
                continue

        logger.info(f"🆕 PDFs created TODAY: {len(files_created_today)}")
        logger.info(f"📅 PDFs created in the PAST (skipped): {len(files_created_past)}")

        # ✅ If no new PDFs today, skip everything
        if not files_created_today:
            logger.info("✅ No new PDFs created today. Skipping ingestion.")
            return {
                "folder_id": folder_id,
                "collection": collection,
                "data_type": data_type,
                "files_seen": len(files),
                "files_created_today": 0,
                "files_created_past": len(files_created_past),
                "drive_unique": 0,
                "qdrant_unique": 0,
                "missing": 0,
                "duplicates_on_drive": 0,
                "skipped_existing": 0,
                "processed": 0,
                "built_nodes": 0,
                "inserted": 0,
                "ok": True,
                "message": "No new PDFs created today"
            }

        # 3) normalize Drive doc_names + detect duplicates (same stem) - ONLY for today's files
        drive_map: Dict[str, dict] = {}
        dup_doc_names: Set[str] = set()

        for f in files_created_today:
            name = (f.get("name") or "").strip()
            dn = _normalize_doc_name(name)
            if dn in drive_map:
                dup_doc_names.add(dn)
                continue
            drive_map[dn] = f

        drive_doc_names = set(drive_map.keys())

        # 4) fetch UNIQUE qdrant doc_names filtered by data_type
        qdrant_doc_names = await _fetch_unique_doc_names_from_qdrant(
            qdrant, collection, data_type=data_type
        )

        # 5) inventory + missing
        missing_doc_names = sorted(drive_doc_names - qdrant_doc_names)
        existing_doc_names = sorted(drive_doc_names & qdrant_doc_names)

        logger.info("📦 Qdrant unique doc_name (data_type=%s): %d", data_type, len(qdrant_doc_names))
        logger.info("🗂️ Drive unique PDFs created TODAY (by doc_name): %d", len(drive_doc_names))
        logger.info("🆕 Missing docs to ingest: %d", len(missing_doc_names))
        if dup_doc_names:
            logger.warning("⚠️ Duplicate doc_names in Drive folder (same stem): %d", len(dup_doc_names))

        # Only process missing
        to_process = [drive_map[dn] for dn in missing_doc_names]

        if not to_process:
            logger.info("✅ All PDFs created today are already in Qdrant. Nothing to process.")
            return {
                "folder_id": folder_id,
                "collection": collection,
                "data_type": data_type,
                "files_seen": len(files),
                "files_created_today": len(files_created_today),
                "files_created_past": len(files_created_past),
                "drive_unique": len(drive_doc_names),
                "qdrant_unique": len(qdrant_doc_names),
                "missing": 0,
                "duplicates_on_drive": len(dup_doc_names),
                "skipped_existing": len(existing_doc_names),
                "processed": 0,
                "built_nodes": 0,
                "inserted": 0,
                "ok": True,
                "message": "All new PDFs already processed"
            }

        # ✅ Concurrency controls with stricter limits to prevent SSL overload
        dl_sem = asyncio.Semaphore(int(os.environ.get("DRIVE_DOWNLOAD_CONCURRENCY", "3")))  # Reduced from 4 to 3
        parse_sem = asyncio.Semaphore(int(os.environ.get("LLAMAPARSE_CONCURRENCY", "2")))

        processed_docs = 0
        failed_docs = 0
        built_nodes = 0
        all_nodes: List[TextNode] = []

        async def process_one(f: dict) -> List[TextNode]:
            nonlocal processed_docs, failed_docs

            pdf_name = (f.get("name") or "").strip()
            file_id = (f.get("id") or "").strip()
            created_time = f.get("createdTime", "")
            
            if not pdf_name or not file_id:
                return []

            norm_dn = _normalize_doc_name(pdf_name)

            try:
                logger.info(f"📄 Processing: {pdf_name}")
                
                # Download with retry and timeout
                try:
                    pdf_bytes = await _download_pdf_bytes_async(drive, file_id, dl_sem)
                except asyncio.TimeoutError:
                    logger.error(f"❌ Timeout downloading {pdf_name}")
                    failed_docs += 1
                    return []
                except ssl.SSLError as e:
                    logger.error(f"❌ SSL error downloading {pdf_name}: {e}")
                    failed_docs += 1
                    return []

                # Parse with LlamaParse (agentic)
                async with parse_sem:
                    md_text = await _llamaparse_pdf_bytes_to_markdown(
                        parser, pdf_bytes, filename=pdf_name
                    )

                if not md_text.strip():
                    logger.warning(f"⚠️ Empty parse output for {pdf_name}")
                    return []

                nodes = _build_nodes_for_doc(
                    doc_text=md_text,
                    doc_name=norm_dn,
                    data_type=data_type,
                    chunk_size=int(cfg.chunk_size),
                    chunk_overlap=int(cfg.chunk_overlap),
                    drive_file_id=file_id,
                    drive_file_name=pdf_name,
                    created_time=created_time,
                )

                processed_docs += 1
                logger.info(f"✅ Successfully processed {pdf_name} ({len(nodes)} chunks)")
                return nodes

            except Exception as e:
                logger.exception(f"❌ Failed processing {pdf_name}: {e}")
                failed_docs += 1
                return []

        # Run concurrently, continue even if some docs fail
        logger.info(f"⚡ Starting to process {len(to_process)} PDFs...")
        results = await asyncio.gather(*[process_one(f) for f in to_process], return_exceptions=True)

        for result in results:
            if isinstance(result, Exception):
                logger.error(f"Exception during processing: {result}")
                failed_docs += 1
                continue
            if isinstance(result, list):
                all_nodes.extend(result)
                built_nodes += len(result)

        logger.info(f"📊 Processing complete: {processed_docs} succeeded, {failed_docs} failed, {built_nodes} nodes built")

        if not all_nodes:
            return {
                "folder_id": folder_id,
                "collection": collection,
                "data_type": data_type,
                "files_seen": len(files),
                "files_created_today": len(files_created_today),
                "files_created_past": len(files_created_past),
                "drive_unique": len(drive_doc_names),
                "qdrant_unique": len(qdrant_doc_names),
                "missing": len(missing_doc_names),
                "duplicates_on_drive": len(dup_doc_names),
                "skipped_existing": len(existing_doc_names),
                "processed": processed_docs,
                "failed": failed_docs,
                "built_nodes": 0,
                "inserted": 0,
                "ok": True,
                "message": "No nodes built (all processing failed or empty)"
            }

        # 6) Remote embed (dense + sparse) via embed_websearcher.py
        logger.info(f"🧮 Embedding {built_nodes} nodes...")
        emb_map = await embed_textnodes_hybrid(
            all_nodes,
            concurrency=int(os.environ.get("EMBED_CONCURRENCY", "2")),
            batch_size=int(os.environ.get("EMBED_BATCH_SIZE", "20")),
            retry_count=int(os.environ.get("EMBED_RETRY_COUNT", "3")),
        )

        # 7) Upsert using your upsert-only module (stable IDs + indexes)
        logger.info(f"💾 Upserting to Qdrant collection: {collection}")
        inserted = await upsert_nodes_hybrid_from_embeddings_async(
            nodes=all_nodes,
            emb_map=emb_map,
            client=qdrant,
            collection=collection,
            dense_name=dense_name,
            sparse_name=sparse_name,
        )

        logger.info(f"✅ Ingestion complete! Inserted {inserted} points into {collection}")

        return {
            "folder_id": folder_id,
            "collection": collection,
            "data_type": data_type,
            "files_seen": len(files),
            "files_created_today": len(files_created_today),
            "files_created_past": len(files_created_past),
            "drive_unique": len(drive_doc_names),
            "qdrant_unique": len(qdrant_doc_names),
            "missing": len(missing_doc_names),
            "duplicates_on_drive": len(dup_doc_names),
            "skipped_existing": len(existing_doc_names),
            "processed": processed_docs,
            "failed": failed_docs,
            "built_nodes": built_nodes,
            "inserted": int(inserted),
            "ok": True,
            "missing_preview": missing_doc_names[:30],
        }

    finally:
        await qdrant.close()
