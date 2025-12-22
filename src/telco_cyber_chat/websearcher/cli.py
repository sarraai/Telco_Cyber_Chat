from __future__ import annotations
import argparse
import os

from .config import WebsearcherConfig
from .pipeline import (
    run_websearcher_build_nodes,
    run_websearcher_build_nodes_from_local_pdfs,
    upsert_nodes_to_qdrant,
)

def _common_cfg(args) -> WebsearcherConfig:
    return WebsearcherConfig(
        seed_urls=args.seed or [],
        allowed_domains=args.domain,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        parse_tier=args.tier,
        vendor=args.vendor,
        collection=args.collection,
    )

def main():
    ap = argparse.ArgumentParser(prog="websearcher")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # --- ingest local PDFs (Drive) ---
    p_local = sub.add_parser("ingest-local", help="Ingest PDFs from a local folder (e.g., downloaded from Drive)")
    p_local.add_argument("--pdf-dir", required=True)
    p_local.add_argument("--vendor", default="websearcher")
    p_local.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "telco_whitepapers"))
    p_local.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))
    p_local.add_argument("--max-pages", type=int, default=50)
    p_local.add_argument("--max-depth", type=int, default=2)
    p_local.add_argument("--tier", type=str, default="agentic")
    p_local.add_argument("--domain", action="append", default=None)
    p_local.add_argument("--seed", action="append", default=None)

    # --- ingest from web crawl + LlamaParse URL ---
    p_web = sub.add_parser("ingest-web", help="Crawl web pages, parse PDF URLs with LlamaParse, ingest into Qdrant")
    p_web.add_argument("--seed", action="append", required=True)
    p_web.add_argument("--domain", action="append", default=None)
    p_web.add_argument("--max-pages", type=int, default=50)
    p_web.add_argument("--max-depth", type=int, default=2)
    p_web.add_argument("--tier", type=str, default="agentic")
    p_web.add_argument("--vendor", default="websearcher")
    p_web.add_argument("--collection", default=os.getenv("QDRANT_COLLECTION", "telco_whitepapers"))
    p_web.add_argument("--embed-model", default=os.getenv("EMBED_MODEL", "BAAI/bge-small-en-v1.5"))

    args = ap.parse_args()
    cfg = _common_cfg(args)

    qdrant_url = os.environ["QDRANT_URL"]
    qdrant_key = os.environ.get("QDRANT_API_KEY")

    if args.cmd == "ingest-local":
        nodes = run_websearcher_build_nodes_from_local_pdfs(cfg, pdf_dir=args.pdf_dir)
    else:
        nodes = run_websearcher_build_nodes(cfg)

    print(f"✅ Built {len(nodes)} TextNodes")

    res = upsert_nodes_to_qdrant(
        nodes=nodes,
        collection=args.collection,
        vendor=cfg.vendor,
        qdrant_url=qdrant_url,
        qdrant_api_key=qdrant_key,
        embed_model=args.embed_model,
    )
    print("✅ Upsert:", res)

if __name__ == "__main__":
    main()
