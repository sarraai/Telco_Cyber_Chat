from __future__ import annotations
import os
import uuid
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from llama_index.core.schema import TextNode
from pypdf import PdfReader
from qdrant_client import QdrantClient
from qdrant_client import models as qmodels
from sentence_transformers import SentenceTransformer

from .config import WebsearcherConfig
from .crawler import discover_pdf_urls
from .llamaparse_url import parse_url_to_markdown
from .chunker import chunk_text


# -------------------- helpers --------------------

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

_UUID_NS = uuid.UUID("5b2f0b2c-7f55-4a3e-9ac2-2e2f3f3f5b4c")

def _stable_point_id(vendor: str, source_id: str, chunk_index: int) -> str:
    # deterministic ids => re-running daily won't create duplicates
    return str(uuid.uuid5(_UUID_NS, f"{vendor}|{source_id}|{chunk_index}"))

def build_nodes_from_text(
    text: str,
    source_id: str,
    vendor: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[TextNode]:
    chunks = chunk_text(text, chunk_size=chunk_size, overlap=chunk_overlap)
    scraped_date = _utc_now_iso()

    nodes: List[TextNode] = []
    for i, ch in enumerate(chunks):
        txt = (
            f"vendor: {vendor}\n"
            f"url: {source_id}\n"
            f"scraped_date: {scraped_date}\n"
            f"chunk_index: {i}\n\n"
            f"{ch}"
        )
        nodes.append(TextNode(text=txt, metadata={"url": source_id}))
    return nodes


# -------------------- WEB (crawl + LlamaParse URL) --------------------

def run_websearcher_discover_and_parse(cfg: WebsearcherConfig) -> Tuple[List[str], Dict[str, str]]:
    api_key = os.getenv(cfg.llamacloud_api_key_env)
    if not api_key:
        raise RuntimeError(f"Missing env var: {cfg.llamacloud_api_key_env}")

    pdf_urls = discover_pdf_urls(
        seed_urls=cfg.seed_urls,
        allowed_domains=cfg.allowed_domains,
        max_pages=cfg.max_pages,
        max_depth=cfg.max_depth,
        timeout_s=cfg.request_timeout_s,
    )

    md_by_url: Dict[str, str] = {}
    for u in pdf_urls:
        md_by_url[u] = parse_url_to_markdown(
            source_url=u,
            api_key=api_key,
            tier=cfg.parse_tier,
            version=cfg.parse_version,
            max_pages=cfg.max_pages_per_doc,
        )

    return pdf_urls, md_by_url

def run_websearcher_build_nodes(cfg: WebsearcherConfig) -> List[TextNode]:
    _, md_by_url = run_websearcher_discover_and_parse(cfg)

    all_nodes: List[TextNode] = []
    for url, md in md_by_url.items():
        all_nodes.extend(
            build_nodes_from_text(
                text=md,
                source_id=url,
                vendor=cfg.vendor,
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
            )
        )
    return all_nodes


# -------------------- LOCAL PDFs (Google Drive download output) --------------------

def _pdf_to_text(pdf_path: Path) -> str:
    reader = PdfReader(str(pdf_path))
    parts: List[str] = []
    for page in reader.pages:
        t = page.extract_text() or ""
        if t.strip():
            parts.append(t)
    return "\n\n".join(parts).strip()

def run_websearcher_build_nodes_from_local_pdfs(cfg: WebsearcherConfig, pdf_dir: str) -> List[TextNode]:
    pdf_path = Path(pdf_dir)
    if not pdf_path.exists():
        raise RuntimeError(f"PDF dir not found: {pdf_dir}")

    all_nodes: List[TextNode] = []
    for p in sorted(pdf_path.glob("*.pdf")):
        text = _pdf_to_text(p)
        source_id = f"gdrive:{p.name}"  # stable id for dedupe
        all_nodes.extend(
            build_nodes_from_text(
                text=text,
                source_id=source_id,
                vendor=cfg.vendor,
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
            )
        )
    return all_nodes


# -------------------- EMBED + UPSERT (dense-only minimal) --------------------

def upsert_nodes_to_qdrant(
    nodes: List[TextNode],
    collection: str,
    vendor: str,
    qdrant_url: str,
    qdrant_api_key: Optional[str] = None,
    embed_model: str = "BAAI/bge-small-en-v1.5",
) -> Dict:
    if not nodes:
        return {"inserted": 0, "collection": collection, "ok": True}

    client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    model = SentenceTransformer(embed_model)
    texts = [n.text for n in nodes]
    vectors = model.encode(texts, normalize_embeddings=True, show_progress_bar=False).tolist()
    dim = len(vectors[0])

    # create collection if missing
    try:
        client.get_collection(collection)
    except Exception:
        client.create_collection(
            collection_name=collection,
            vectors_config=qmodels.VectorParams(size=dim, distance=qmodels.Distance.COSINE),
        )

    points: List[qmodels.PointStruct] = []
    for n, v in zip(nodes, vectors):
        url = (n.metadata or {}).get("url", "")
        # chunk index is in text; we can parse it, but simplest is to rely on list order.
        # We'll set id based on url + position in current run.
        # Better: parse chunk_index from text lines:
        chunk_index = 0
        for line in (n.text.splitlines()[:10]):
            if line.startswith("chunk_index:"):
                try:
                    chunk_index = int(line.split(":", 1)[1].strip())
                except Exception:
                    pass
                break

        pid = _stable_point_id(vendor=vendor, source_id=url, chunk_index=chunk_index)

        payload = {
            "vendor": vendor,
            "url": url,
            "text": n.text,
            "scraped_date": _utc_now_iso(),
        }

        points.append(qmodels.PointStruct(id=pid, vector=v, payload=payload))

    client.upsert(collection_name=collection, points=points)
    return {"inserted": len(points), "collection": collection, "ok": True}
