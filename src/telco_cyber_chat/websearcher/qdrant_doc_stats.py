# src/telco_cyber_chat/websearcher/qdrant_doc_stats.py
from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Optional, Set, Tuple

from qdrant_client import QdrantClient
from qdrant_client import models as qmodels

logger = logging.getLogger(__name__)


# -------------------- Normalization --------------------

def normalize_doc_name(name: str) -> str:
    """Normalize a doc_name already stored/produced (expected: stem, lower)."""
    return (name or "").strip().lower()


def normalize_doc_name_from_filename(filename: str) -> str:
    """Normalize a filename into pipeline doc_name convention: Path(stem).lower()."""
    return Path(filename or "").stem.strip().lower()


# -------------------- Local (Drive mount / folder) --------------------

def list_local_pdf_doc_names(root_dir: str) -> Set[str]:
    """
    Count PDFs from a local directory (e.g., Colab Drive mount or any local folder).
    Returns UNIQUE normalized doc names using PDF *stem* (no .pdf).
    """
    root = Path(root_dir)
    if not root.exists():
        logger.warning("Local root does not exist: %s", root_dir)
        return set()

    pdfs = list(root.rglob("*.pdf"))
    return {normalize_doc_name_from_filename(p.name) for p in pdfs if p.name}


# -------------------- Qdrant --------------------

def fetch_unique_doc_names_from_qdrant(
    client: QdrantClient,
    collection: str,
    *,
    data_type: Optional[str] = "unstructured",
    batch_size: int = 512,
) -> Set[str]:
    """
    Scroll the collection and collect UNIQUE payload['doc_name'].
    If data_type is not None, filter by payload['data_type'] == data_type.
    """
    doc_names: Set[str] = set()
    next_offset = None

    scroll_filter = None
    if data_type is not None:
        scroll_filter = qmodels.Filter(
            must=[
                qmodels.FieldCondition(
                    key="data_type",
                    match=qmodels.MatchValue(value=str(data_type)),
                )
            ]
        )

    while True:
        points, next_offset = client.scroll(
            collection_name=collection,
            scroll_filter=scroll_filter,
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


# -------------------- Inventory / diff --------------------

def split_existing_vs_new(
    drive_doc_names: Iterable[str],
    qdrant_doc_names: Set[str],
) -> Tuple[Set[str], Set[str]]:
    """
    Given drive/local doc names and qdrant doc names, returns (existing, new).
    """
    drive_norm = {normalize_doc_name(x) for x in drive_doc_names if x}
    existing = {dn for dn in drive_norm if dn in qdrant_doc_names}
    new = {dn for dn in drive_norm if dn not in qdrant_doc_names}
    return existing, new


def log_doc_inventory(
    *,
    client: QdrantClient,
    collection: str,
    data_type: Optional[str] = "unstructured",
    drive_root_dir: Optional[str] = None,
) -> Tuple[int, int, int]:
    """
    Logs:
      - unique doc_name count in Qdrant (optionally filtered by data_type)
      - unique pdf count locally (by stem) if drive_root_dir provided
      - missing docs = set difference (not just count subtraction)

    Returns (qdrant_unique, drive_unique, missing_count).
    """
    qdrant_docs = fetch_unique_doc_names_from_qdrant(
        client, collection, data_type=data_type
    )
    qdrant_unique = len(qdrant_docs)

    logger.info(
        "Qdrant unique doc_name count (data_type=%s): %d",
        data_type,
        qdrant_unique,
    )

    drive_unique = 0
    missing_count = 0

    if drive_root_dir:
        drive_docs = list_local_pdf_doc_names(drive_root_dir)
        drive_unique = len(drive_docs)

        _, missing = split_existing_vs_new(drive_docs, qdrant_docs)
        missing_count = len(missing)

        logger.info("🗂️ Local unique PDF count (by stem): %d", drive_unique)
        logger.info("🆕 Missing PDFs to ingest (local - qdrant): %d", missing_count)

    return qdrant_unique, drive_unique, missing_count
