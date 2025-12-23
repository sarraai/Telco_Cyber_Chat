# src/telco_cyber_chat/websearcher/config.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class WebsearcherConfig:
    # ---------------- Google Drive ----------------
    drive_folder_id_env: str = "GDRIVE_FOLDER_ID"
    drive_sa_json_env: str = "GDRIVE_SA_JSON"
    drive_out_dir_env: str = "GDRIVE_OUT_DIR"     # where PDFs get downloaded
    max_files: int = 200                          # safety cap per run

    # ---------------- Chunking ----------------
    chunk_size: int = 2000
    chunk_overlap: int = 200

    # ---------------- Qdrant ----------------
    qdrant_url_env: str = "QDRANT_URL"
    qdrant_api_key_env: str = "QDRANT_API_KEY"
    qdrant_collection_env: str = "QDRANT_COLLECTION"
    collection: Optional[str] = None              # overrides env if set

    # ---------------- Payload tagging ----------------
    doc_type: str = "unstructured"                # matches your payload index doc_type
