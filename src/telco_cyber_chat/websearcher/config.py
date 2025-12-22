from __future__ import annotations

from dataclasses import dataclass
from typing import Optional


@dataclass
class WebsearcherConfig:
    # Google Drive
    drive_folder_id_env: str = "GDRIVE_FOLDER_ID"
    max_files: int = 200

    # Chunking
    chunk_size: int = 2000
    chunk_overlap: int = 200

    # Qdrant / tagging
    vendor: str = "websearcher"
    collection: Optional[str] = None
