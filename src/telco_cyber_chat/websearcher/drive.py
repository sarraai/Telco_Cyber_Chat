import os, io, json
from pathlib import Path

from google.oauth2 import service_account
from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload

FOLDER_ID = os.environ["GDRIVE_FOLDER_ID"]
OUT_DIR = Path(os.environ.get("GDRIVE_OUT_DIR", "data/drive_pdfs"))
OUT_DIR.mkdir(parents=True, exist_ok=True)

creds = service_account.Credentials.from_service_account_info(
    json.loads(os.environ["GDRIVE_SA_JSON"]),
    scopes=["https://www.googleapis.com/auth/drive.readonly"],
)

drive = build("drive", "v3", credentials=creds)

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
for f in files:
    name = f["name"]
    file_id = f["id"]
    out_path = OUT_DIR / name

    request = drive.files().get_media(fileId=file_id)
    with io.FileIO(out_path, "wb") as fh:
        downloader = MediaIoBaseDownload(fh, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()

    downloaded += 1

print(f"✅ Downloaded {downloaded} PDFs to {OUT_DIR}")
