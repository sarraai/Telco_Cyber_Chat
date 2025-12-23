from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

from qdrant_client import QdrantClient

logger = logging.getLogger(__name__)


def normalize_doc_name(name: str) -> str:
    return (name or "").strip().lower()


def list_drive_pdf_doc_names(root_dir: str) -> Set[str]:
    """
    Counts PDFs from a local directory (ex: Colab Drive mount or a local folder).
    Returns UNIQUE normalized doc names.
    """
    root = Path(root_dir)
    if not root.exists():
        logger.warning("Drive root does not exist: %s", root_dir)
        return set()

    pdfs = list(root.rglob("*.pdf"))
    names = {normalize_doc_name(p.name) for p in pdfs if p.name}
    return names


def fetch_unique_doc_names_from_qdrant(
    client: QdrantClient,
    collection: str,
    *,
    batch_size: int = 512,
) -> Set[str]:
    """
    Scroll the whole collection, reading only payload['doc_name'].
    Returns UNIQUE normalized doc names.
    """
    doc_names: Set[str] = set()
    next_offset = None

    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            limit=batch_size,
            offset=next_offset,
            with_payload=["doc_name"],
            with_vectors=False,
        )

        for p in points:
            payload = getattr(p, "payload", None) or {}
            dn = payload.get("doc_name")
            if isinstance(dn, str) and dn.strip():
                doc_names.add(normalize_doc_name(dn))

        if next_offset is None:
            break

    return doc_names


def log_doc_inventory(
    *,
    client: QdrantClient,
    collection: str,
    drive_root_dir: Optional[str] = None,
) -> Tuple[int, int, int]:
    """
    Logs:
      - unique doc_name count in Qdrant
      - unique pdf count on drive (if provided)
      - estimated new docs = drive_unique - qdrant_unique
    Returns (qdrant_unique, drive_unique, new_estimated).
    """
    qdrant_docs = fetch_unique_doc_names_from_qdrant(client, collection)
    qdrant_unique = len(qdrant_docs)

    logger.info("📦 Qdrant unique doc_name count: %d", qdrant_unique)

    drive_unique = 0
    new_estimated = 0

    if drive_root_dir:
        drive_docs = list_drive_pdf_doc_names(drive_root_dir)
        drive_unique = len(drive_docs)
        new_estimated = max(0, drive_unique - qdrant_unique)

        logger.info("🗂️ Drive unique PDF count (by filename): %d", drive_unique)
        logger.info("🆕 Estimated new PDFs (drive - qdrant): %d", new_estimated)

    return qdrant_unique, drive_unique, new_estimated


def split_existing_vs_new(
    drive_doc_names: Iterable[str],
    qdrant_doc_names: Set[str],
) -> Tuple[Set[str], Set[str]]:
    """
    Given drive doc names and qdrant doc names, returns (existing, new).
    """
    drive_norm = {normalize_doc_name(x) for x in drive_doc_names if x}
    existing = {dn for dn in drive_norm if dn in qdrant_doc_names}
    new = {dn for dn in drive_norm if dn not in qdrant_doc_names}
    return existing, new
