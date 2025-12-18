from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class WebsearcherConfig:
    # crawler
    seed_urls: List[str]
    allowed_domains: Optional[List[str]] = None
    max_pages: int = 50
    max_depth: int = 2
    request_timeout_s: int = 25

    # llamaparse
    llamacloud_api_key_env: str = "LLAMA_CLOUD_API_KEY"
    parse_tier: str = "agentic"          # "cost_effective" | "agentic" | "agentic_plus" | "fast"
    parse_version: str = "latest"        # v2 supports "latest" (v2 is alpha)
    max_pages_per_doc: int = 200

    # chunking
    chunk_size: int = 2000
    chunk_overlap: int = 200

    # qdrant / vendor tagging
    vendor: str = "websearcher"
    collection: Optional[str] = None

    # optional local cache
    cache_markdown_dir: Optional[str] = "src/telco_cyber_chat/websearcher/_cache_md"
