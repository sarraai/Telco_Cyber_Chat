# telco_cyber_chat/webscraping/scrape_orchestrator.py

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from .nokia_scraper import scrape_nokia
from .ericsson_scraper import scrape_ericsson
from .huawei_scraper import scrape_huawei
from .cisco_scraper import scrape_cisco
from .variot_scraper import scrape_variot
from .mitre_attack_scraper import scrape_mitre_mobile  # ✅ add


VendorName = Literal["nokia", "ericsson", "huawei", "cisco", "variot", "mitre_mobile"]


def _ensure_vendor(records: List[Dict[str, Any]], vendor: str) -> List[Dict[str, Any]]:
    """
    Guarantee vendor is present in every record.
    We keep vendor in the record so node_builder can put it into node.text.
    """
    out: List[Dict[str, Any]] = []
    for r in records or []:
        if not isinstance(r, dict):
            continue
        rr = dict(r)
        rr["vendor"] = vendor
        out.append(rr)
    return out


def scrape_vendor(
    vendor: VendorName,
    check_qdrant: bool = True,
) -> List[Dict[str, Any]]:
    """
    Run a single source scraper and return records in the NEW schema:

      - Each record is a dict with arbitrary fields
      - MUST include at least: {"url": "..."} (string) when available
      - Orchestrator injects: {"vendor": "<vendor>"}
    """
    if vendor == "nokia":
        return _ensure_vendor(scrape_nokia(check_qdrant=check_qdrant), "nokia")

    if vendor == "ericsson":
        return _ensure_vendor(scrape_ericsson(check_qdrant=check_qdrant), "ericsson")

    if vendor == "huawei":
        return _ensure_vendor(scrape_huawei(check_qdrant=check_qdrant), "huawei")

    if vendor == "cisco":
        return _ensure_vendor(scrape_cisco(check_qdrant=check_qdrant), "cisco")

    if vendor == "variot":
        return _ensure_vendor(scrape_variot(check_qdrant=check_qdrant), "variot")

    if vendor == "mitre_mobile":
        return _ensure_vendor(scrape_mitre_mobile(check_qdrant=check_qdrant), "mitre_mobile")

    raise ValueError(f"Unknown vendor: {vendor}")


def scrape_all_vendors(
    vendors: Optional[List[VendorName]] = None,
    check_qdrant: bool = True,
) -> List[Dict[str, Any]]:
    """
    High-level orchestrator:

      - Calls each scraper
      - Each scraper returns list[dict] with arbitrary fields + url (when applicable)
      - We add vendor=<vendor> into each record
      - Returns one unified list of records ready for node_builder
    """
    if vendors is None:
        vendors = ["nokia", "ericsson", "huawei", "cisco", "variot", "mitre_mobile"]  # ✅ add

    all_records: List[Dict[str, Any]] = []

    for v in vendors:
        try:
            recs = scrape_vendor(v, check_qdrant=check_qdrant)
            print(f"[SCRAPER] {v}: {len(recs)} new records")
            all_records.extend(recs)
        except Exception as e:
            print(f"[WARN] {v} scraper failed: {e}")

    print(f"[SCRAPER] Total new records from all vendors: {len(all_records)}")
    return all_records
