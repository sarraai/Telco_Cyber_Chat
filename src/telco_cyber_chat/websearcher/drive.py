import os, io, json
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

# ---------- Drive ----------
FOLDER_ID = os.environ["GDRIVE_FOLDER_ID"]
OUT_DIR = Path(os.environ.get("GDRIVE_OUT_DIR", "data/drive_pdfs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

creds = service_account.Credentials.from_service_account_info(
    json.loads(os.environ["GDRIVE_SA_JSON"]),
    scopes=["https://www.googleapis.com/auth/drive.readonly"],
)
drive = build("drive", "v3", credentials=creds)

# ---------- Qdrant ----------
QDRANT_URL = os.environ["QDRANT_URL"]
QDRANT_API_KEY = os.environ.get("QDRANT_API_KEY")
COLLECTION = os.environ.get("QDRANT_COLLECTION", "telco_whitepapers")

qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)

def exists_in_qdrant_by_name(pdf_name: str) -> bool:
    flt = qmodels.Filter(
        must=[qmodels.FieldCondition(
            key="doc_name",  # you must store this in payload when you upsert!
            match=qmodels.MatchValue(value=pdf_name)
        )]
    )
    pts, _ = qdrant.scroll(
        collection_name=COLLECTION,
        scroll_filter=flt,
        limit=1,
        with_payload=True,
        with_vectors=False,
    )
    return len(pts) > 0

# ---------- List PDFs ----------
q = f"'{FOLDER_ID}' in parents and mimeType='application/pdf' and trashed=false"
files = []
page_token = None
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

    # ✅ CHECK QDRANT BY NAME
    if exists_in_qdrant_by_name(name):
        skipped += 1
        continue

    # ✅ ONLY DOWNLOAD IF NEW
    out_path = OUT_DIR / name
    request = drive.files().get_media(fileId=file_id)
    with io.FileIO(out_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

    downloaded += 1

print(f"✅ Downloaded {downloaded} PDFs to {OUT_DIR}")
print(f"⏭️ Skipped {skipped} PDFs already in Qdrant (by name)")
