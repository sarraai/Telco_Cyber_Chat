# telco_cyber_chat/webscraping/scrape_orchestrator.py

from __future__ import annotations

from typing import Dict, List, Literal, Optional

from .nokia_scraper import scrape_nokia
from .ericsson_scraper import scrape_ericsson
from .huawei_scraper import scrape_huawei
from .cisco_scraper import scrape_cisco
from .variot_scraper import scrape_variot

VendorName = Literal["nokia", "ericsson", "huawei", "cisco", "variot"]


def scrape_vendor(
    vendor: VendorName,
    check_qdrant: bool = True,
) -> List[Dict[str, str]]:
    """
    Run a single vendor scraper and return docs in the common schema:
      { "url": str, "title": str, "description": str }

    Scraper functions themselves already:
      - build `title` and `description` by merging relevant fields
      - check Qdrant (via url_already_ingested) when check_qdrant=True
    """
    if vendor == "nokia":
        return scrape_nokia(check_qdrant=check_qdrant)
    if vendor == "ericsson":
        return scrape_ericsson(check_qdrant=check_qdrant)
    if vendor == "huawei":
        return scrape_huawei(check_qdrant=check_qdrant)
    if vendor == "cisco":
        return scrape_cisco(check_qdrant=check_qdrant)
    if vendor == "variot":
        return scrape_variot(check_qdrant=check_qdrant)

    raise ValueError(f"Unknown vendor: {vendor}")


def scrape_all_vendors(
    vendors: Optional[List[VendorName]] = None,
    check_qdrant: bool = True,
) -> List[Dict[str, str]]:
    """
    High-level orchestrator:

      - Calls each vendor scraper
      - Each scraper:
          * returns {url, title, description}
          * already skips URLs that exist in Qdrant when check_qdrant=True
      - Returns one unified list of documents ready for TextNode creation.
    """
    if vendors is None:
        vendors = ["nokia", "ericsson", "huawei", "cisco", "variot"]

    all_docs: List[Dict[str, str]] = []

    for v in vendors:
        try:
            docs = scrape_vendor(v, check_qdrant=check_qdrant)
            print(f"[SCRAPER] {v}: {len(docs)} new docs")
            all_docs.extend(docs)
        except Exception as e:
            # Don't break the whole pipeline if one vendor fails
            print(f"[WARN] {v} scraper failed: {e}")

    print(f"[SCRAPER] Total new docs from all vendors: {len(all_docs)}")
    return all_docs
