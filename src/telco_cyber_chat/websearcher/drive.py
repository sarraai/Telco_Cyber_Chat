# src/telco_cyber_chat/websearcher/drive.py
from __future__ import annotations

import os, io, json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels


def sync_drive_pdfs_skip_if_ingested() -> Dict:
    folder_id = os.environ["GDRIVE_FOLDER_ID"]
    out_dir = Path(os.environ.get("GDRIVE_OUT_DIR", "data/drive_pdfs"))
    out_dir.mkdir(parents=True, exist_ok=True)

    # Drive
    creds = service_account.Credentials.from_service_account_info(
        json.loads(os.environ["GDRIVE_SA_JSON"]),
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    drive = build("drive", "v3", credentials=creds)

    # Qdrant
    qdrant = QdrantClient(
        url=os.environ["QDRANT_URL"],
        api_key=os.environ.get("QDRANT_API_KEY"),
    )
    collection = os.environ.get("QDRANT_COLLECTION", "telco_whitepapers")

    def exists_by_name(pdf_name: str) -> bool:
        flt = qmodels.Filter(
            must=[qmodels.FieldCondition(
                key="doc_name",  # MUST exist in payload of ingested points
                match=qmodels.MatchValue(value=pdf_name)
            )]
        )
        pts, _ = qdrant.scroll(
            collection_name=collection,
            scroll_filter=flt,
            limit=1,
            with_payload=True,
            with_vectors=False,
        )
        return len(pts) > 0

    # List PDFs
    q = f"'{folder_id}' in parents and mimeType='application/pdf' and trashed=false"
    files: List[Dict] = []
    page_token: Optional[str] = None
    while True:
        resp = drive.files().list(
            q=q,
            fields="nextPageToken, files(id,name,modifiedTime)",
            pageToken=page_token,
            pageSize=200,
        ).execute()
        files += resp.get("files", [])
        page_token = resp.get("nextPageToken")
        if not page_token:
            break

    downloaded = 0
    skipped = 0

    for f in files:
        name = f["name"]
        file_id = f["id"]

        if exists_by_name(name):
            skipped += 1
            continue

        out_path = out_dir / name
        request = drive.files().get_media(fileId=file_id)
        with io.FileIO(out_path, "wb") as fh:
            downloader = MediaIoBaseDownload(fh, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()

        downloaded += 1

    return {
        "files_seen": len(files),
        "downloaded": downloaded,
        "skipped": skipped,
        "out_dir": str(out_dir),
        "ok": True,
    }
