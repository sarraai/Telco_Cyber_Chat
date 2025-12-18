from __future__ import annotations
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

from llama_index.core.schema import TextNode

from .config import WebsearcherConfig
from .crawler import discover_pdf_urls
from .llamaparse_url import parse_url_to_markdown
from .chunker import chunk_text

def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

def build_nodes_from_markdown(
    md: str,
    source_url: str,
    vendor: str,
    chunk_size: int,
    chunk_overlap: int,
) -> List[TextNode]:
    chunks = chunk_text(md, chunk_size=chunk_size, overlap=chunk_overlap)
    scraped_date = _utc_now_iso()

    nodes: List[TextNode] = []
    for i, ch in enumerate(chunks):
        # Keep your existing convention: vendor inside node.text, and only URL in metadata
        txt = (
            f"vendor: {vendor}\n"
            f"url: {source_url}\n"
            f"scraped_date: {scraped_date}\n"
            f"chunk_index: {i}\n\n"
            f"{ch}"
        )
        nodes.append(TextNode(text=txt, metadata={"url": source_url}))

    return nodes

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
            build_nodes_from_markdown(
                md=md,
                source_url=url,
                vendor=cfg.vendor,
                chunk_size=cfg.chunk_size,
                chunk_overlap=cfg.chunk_overlap,
            )
        )

    return all_nodes
