from __future__ import annotations
import argparse

from .config import WebsearcherConfig
from .pipeline import run_websearcher_build_nodes

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", action="append", required=True, help="Seed page URL (repeatable)")
    ap.add_argument("--domain", action="append", default=None, help="Allowed domain (repeatable)")
    ap.add_argument("--max-pages", type=int, default=50)
    ap.add_argument("--max-depth", type=int, default=2)
    ap.add_argument("--tier", type=str, default="agentic")
    args = ap.parse_args()

    cfg = WebsearcherConfig(
        seed_urls=args.seed,
        allowed_domains=args.domain,
        max_pages=args.max_pages,
        max_depth=args.max_depth,
        parse_tier=args.tier,
    )

    nodes = run_websearcher_build_nodes(cfg)
    print(f"✅ Built {len(nodes)} TextNodes")

    # NEXT STEP:
    # Hook into your existing embed + upsert flow, e.g.:
    # - embed nodes with your BGE-M3 service
    # - upsert to Qdrant (vendor/url payload strategy)
    #
    # I’ll align this with your existing functions in telco_cyber_chat when you paste
    # the current embed+upsert imports you want to reuse.

if __name__ == "__main__":
    main()
