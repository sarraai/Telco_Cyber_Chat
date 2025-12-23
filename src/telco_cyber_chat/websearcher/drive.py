# src/telco_cyber_chat/websearcher/drive.py
from __future__ import annotations

import io
import json
import os
from pathlib import Path
from typing import Dict, List, Optional

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels


def _normalize_doc_name(filename: str) -> str:
    # strip extension (.pdf) and whitespace
    return Path(filename).stem.strip()


def sync_drive_pdfs_skip_if_ingested() -> Dict:
    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    out_dir = Path(os.environ.get("GDRIVE_OUT_DIR", "data/drive_pdfs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # -------- Drive client --------
    creds = service_account.Credentials.from_service_account_info(
        json.loads(os.environ["GDRIVE_SA_JSON"]),
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    drive = build("drive", "v3", credentials=creds)

    # -------- Qdrant client --------
    qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ.get("QDRANT_API_KEY"),
    )
    collection = os.environ.get("QDRANT_COLLECTION", "telco_whitepapers")

    # Indexes expected to exist in Qdrant:
    # - doc_name (keyword)
    # - data_type (keyword)

    def exists_unstructured_by_doc_name(doc_name: str) -> bool:
        flt = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="data_type",
                    match=qmodels.MatchValue(value="unstructured"),
                ),
                qmodels.FieldCondition(
                    key="doc_name",
                    match=qmodels.MatchValue(value=doc_name),
                ),
            ]
        )
        pts, _ = qdrant.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(pts) > 0

    # -------- List PDFs in folder --------
    q = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    files: List[Dict] = []
    page_token: Optional[str] = None

    while True:
        resp = drive.files().list(
            q=q,
            fields="nextPageToken, files(id,name)",
            pageToken=page_token,
            pageSize=200,
        ).execute()
        files += resp.get("files", [])
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    downloaded, skipped = 0, 0
    downloaded_files: List[str] = []
    skipped_docs: List[str] = []

    for f in files:
        name = f["name"]
        file_id = f["id"]

        doc_name = _normalize_doc_name(name)  # IMPORTANT: no .pdf
        if exists_unstructured_by_doc_name(doc_name):
            skipped += 1
            skipped_docs.append(doc_name)
            continue

        out_path = out_dir / name
        request = drive.files().get_media(fileId=file_id)
        with io.FileIO(out_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        downloaded += 1
        downloaded_files.append(str(out_path))

    return {
        "files_seen": len(files),
        "downloaded": downloaded,
        "skipped": skipped,
        "downloaded_files": downloaded_files,
        "skipped_docs": skipped_docs,
        "out_dir": str(out_dir),
        "ok": True,
    }
