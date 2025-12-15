import os
import json
import time
import re
import math
import logging
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional
from urllib.parse import urlparse, urlunparse

import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup, Tag

# Keep this import for compatibility (fallback if Qdrant env isn't set)
from telco_cyber_chat.webscraping.scrape_core import url_already_ingested

# Qdrant direct check (vendor + url)
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

# ================== LOGGER ==================
logger = logging.getLogger(__name__)

# ================== CONFIG ==================
VENDOR = "VARIoT"  # filtering value stored in Qdrant

API_KEY = os.getenv("VARIOT_API_KEY")  # no hardcoded fallback
BASE_URL = "https://www.variotdbs.pl/api"
HEADERS = {"Authorization": f"Token {API_KEY}"} if API_KEY else {}
REQ_TIMEOUT = (10, 45)  # (connect, read)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "Telco_CyberChat")

# ================== ROBUST SESSION WITH RETRIES ==================
def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET", "POST"]),
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    s.headers.update({"User-Agent": "VARIoT-scraper/2.0"})
    return s

# ================== URL HELPERS ==================
def canonicalize_url(u: str) -> str:
    """Remove query + fragment for consistent Qdrant filtering."""
    u = (u or "").strip()
    if not u:
        return ""
    p = urlparse(u)
    return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))

def extract_url_from_entry(entry: Dict[str, Any]) -> Optional[str]:
    u = entry.get("url")
    if isinstance(u, str) and u.startswith("http"):
        return canonicalize_url(u)

    vid = entry.get("id")
    if isinstance(vid, str) and vid:
        return canonicalize_url(f"https://www.variotdbs.pl/vuln/{vid}/")

    return None

# ================== PAGINATION ==================
def fetch_all(url: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    next_url = url
    session = make_session()

    while next_url:
        query = params if next_url == url else {}
        r = session.get(next_url, headers=HEADERS, params=query, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        data = r.json()

        if isinstance(data, dict) and "results" in data:
            results.extend(data["results"] or [])
            next_url = data.get("next")
        elif isinstance(data, list):
            results.extend(data)
            next_url = None
        else:
            raise ValueError(f"Unexpected response shape: {type(data)}")

        time.sleep(0.3)

    return results

# ================== HTML SECTION PARSING ==================
def get_section_content(soup: BeautifulSoup, heading_text: str) -> Optional[str]:
    all_divs = soup.find_all("div")
    for i, div in enumerate(all_divs):
        h4 = div.find("h4")
        if h4 and heading_text.upper() in h4.get_text().upper():
            if i + 1 < len(all_divs):
                next_div = all_divs[i + 1]
                p = next_div.find("p", class_="fs-6")
                if p:
                    return p.get_text(strip=True)
    return None

def get_section_table(soup: BeautifulSoup, heading_text: str) -> Optional[Tag]:
    all_divs = soup.find_all("div")
    for i, div in enumerate(all_divs):
        h4 = div.find("h4")
        if h4 and heading_text.upper() in h4.get_text().upper():
            if i + 1 < len(all_divs):
                next_div = all_divs[i + 1]
                table = next_div.find("table")
                if table:
                    return table
    return None

def parse_affected_products(table: Optional[Tag]) -> List[Dict[str, str]]:
    products: List[Dict[str, str]] = []
    if not table:
        return products

    tbody = table.find("tbody")
    rows = tbody.find_all("tr") if tbody else table.find_all("tr")

    for row in rows:
        cells = row.find_all("td")
        if len(cells) >= 8:
            product: Dict[str, str] = {}
            for i in range(0, len(cells), 2):
                if i + 1 < len(cells):
                    label_cell = cells[i]
                    value_cell = cells[i + 1]

                    label = label_cell.get_text(strip=True).rstrip(":")
                    value = value_cell.get_text(strip=True)

                    if "progress" in str(value_cell) or not value:
                        continue

                    product[label] = value

            if product:
                products.append(product)

    return products

def parse_cvss_table(table: Optional[Tag]) -> List[Dict[str, str]]:
    cvss_entries: List[Dict[str, str]] = []
    if not table:
        return cvss_entries

    tbody = table.find("tbody")
    if not tbody:
        return cvss_entries

    main_rows = tbody.find_all("tr", recursive=False)

    for main_row in main_rows:
        main_cells = main_row.find_all("td", recursive=False)
        for main_cell in main_cells:
            nested_table = main_cell.find("table")
            if not nested_table:
                continue

            cvss_entry: Dict[str, str] = {}
            nested_tbody = nested_table.find("tbody")
            if nested_tbody:
                nested_rows = nested_tbody.find_all("tr")
                for row in nested_rows:
                    cells = row.find_all("td")
                    if len(cells) == 2:
                        label = cells[0].get_text(strip=True).rstrip(":")
                        value = cells[1].get_text(strip=True)
                        if value and "Trust:" not in value and label:
                            cvss_entry[label] = value

            if cvss_entry:
                cvss_entries.append(cvss_entry)

    return cvss_entries

def parse_simple_table(table: Optional[Tag]) -> List[Dict[str, str]]:
    items: List[Dict[str, str]] = []
    if not table:
        return items

    tbody = table.find("tbody")
    rows = tbody.find_all("tr") if tbody else table.find_all("tr")

    for row in rows:
        cells = row.find_all("td")
        item: Dict[str, str] = {}
        i = 0
        while i < len(cells):
            cell = cells[i]
            if "progress" in str(cell):
                i += 1
                continue

            small = cell.find("small", class_="text-muted")
            if small:
                label = small.get_text(strip=True).rstrip(":")
                if i + 1 < len(cells):
                    value_cell = cells[i + 1]
                    if "progress" not in str(value_cell):
                        value = value_cell.get_text(strip=True)
                        item[label] = value
                i += 2
            else:
                i += 1

        if item:
            items.append(item)

    return items

# ================== DETAIL PAGE SCRAPE ==================
def scrape_vulnerability_page(url: str, session: requests.Session) -> Dict[str, Any]:
    if not url:
        return {}

    try:
        resp = session.get(url, timeout=REQ_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        logger.warning("[VARIOT] Could not fetch %s: %s", url, e)
        return {}

    soup = BeautifulSoup(resp.text, "html.parser")
    data: Dict[str, Any] = {}

    id_content = get_section_content(soup, "ID")
    if id_content:
        data["id"] = id_content

    cve_content = get_section_content(soup, "CVE")
    if cve_content and "CVE-" in cve_content:
        data["cve"] = cve_content

    desc_content = get_section_content(soup, "DESCRIPTION")
    if desc_content:
        data["description"] = desc_content

    og_title = soup.find("meta", attrs={"property": "og:title"})
    if og_title and og_title.get("content"):
        data["title"] = og_title["content"].strip()
    elif soup.title:
        data["title"] = soup.title.text.strip()

    affected_table = get_section_table(soup, "AFFECTED PRODUCTS")
    products = parse_affected_products(affected_table)
    if products:
        data["affected_products"] = products

    cvss_table = get_section_table(soup, "CVSS")
    cvss_data = parse_cvss_table(cvss_table)
    if cvss_data:
        data["cvss"] = cvss_data

    problem_table = get_section_table(soup, "PROBLEMTYPE DATA")
    problem_items = parse_simple_table(problem_table)
    problem_types = [it.get("problemtype") for it in problem_items if it.get("problemtype")]
    if problem_types:
        data["problem_types"] = problem_types

    external_table = get_section_table(soup, "EXTERNAL IDS")
    external_ids = parse_simple_table(external_table)
    if external_ids:
        data["external_ids"] = external_ids

    ref_table = get_section_table(soup, "REFERENCES")
    ref_items = parse_simple_table(ref_table)
    references = [it.get("url") for it in ref_items if it.get("url")]
    if references:
        data["references"] = references

    sources_table = get_section_table(soup, "SOURCES")
    sources = parse_simple_table(sources_table)
    if sources:
        data["sources"] = sources

    data["url"] = canonicalize_url(url)
    return data

# ================== QDRANT DEDUPE: vendor + url ==================
def build_qdrant_client_from_env() -> Optional[QdrantClient]:
    if not QDRANT_URL:
        return None
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(url=QDRANT_URL)

def vendor_url_already_ingested_qdrant(
    client: QdrantClient,
    collection: str,
    vendor: str,
    url: str,
) -> bool:
    pts, _ = client.scroll(
        collection_name=collection,
        scroll_filter=Filter(
            must=[
                FieldCondition(key="vendor", match=MatchValue(value=vendor)),
                FieldCondition(key="url", match=MatchValue(value=url)),
            ]
        ),
        limit=1,
        with_payload=False,
        with_vectors=False,
    )
    return len(pts) > 0

def already_ingested(url: str) -> bool:
    """
    Preferred: vendor+url Qdrant check (since both are indexed).
    Fallback: url_already_ingested(url) if Qdrant env not present.
    """
    url = canonicalize_url(url)
    qc = build_qdrant_client_from_env()
    if qc is not None:
        return vendor_url_already_ingested_qdrant(qc, QDRANT_COLLECTION, VENDOR, url)
    return url_already_ingested(url)

# ================== MAIN PROCESSING (SAME LOGIC) ==================
def get_all_vulns(
    since_ts: str,
    before_ts: str,
    check_qdrant: bool = True,
) -> List[Dict[str, Any]]:
    if not API_KEY:
        raise RuntimeError("VARIOT_API_KEY is missing. Set it in env/Colab secrets.")

    vulns_url = f"{BASE_URL}/vulns/"
    params = {"jsonld": "false", "limit": 100, "since": since_ts, "before": before_ts}

    logger.info("[VARIOT] Fetching from API since=%s before=%s", since_ts, before_ts)
    items = fetch_all(vulns_url, params)
    logger.info("[VARIOT] Total fetched from API: %d", len(items))

    records: List[Dict[str, Any]] = []
    html_session = make_session()

    seen_urls: set[str] = set()
    n_seen = n_skipped = n_new = 0

    for idx, item in enumerate(items, 1):
        if not isinstance(item, dict):
            continue

        url = extract_url_from_entry(item)
        if not url:
            continue

        if url in seen_urls:
            continue
        seen_urls.add(url)
        n_seen += 1

        # ✅ Qdrant dedupe BEFORE scraping HTML
        if check_qdrant:
            try:
                if already_ingested(url):
                    n_skipped += 1
                    logger.info("[VARIOT] Skip already-ingested: vendor=%s url=%s", VENDOR, url)
                    continue
            except Exception as e:
                logger.error("[VARIOT] Qdrant check failed (%s). Will scrape anyway: %s", e, url)

        n_new += 1
        logger.info("[VARIOT] [%d/%d] Scraping %s", idx, len(items), url)

        vuln_data = scrape_vulnerability_page(url, html_session)
        if vuln_data:
            vuln_data["vendor"] = VENDOR
            vuln_data["scraped_date"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            records.append(vuln_data)

        time.sleep(0.5)  # polite to site

    logger.info("[VARIOT] Seen=%d skipped=%d new=%d output=%d", n_seen, n_skipped, n_new, len(records))
    return records

def scrape_variot(
    check_qdrant: bool = True,
    since_ts: Optional[str] = None,
    before_ts: Optional[str] = None,
) -> List[Dict[str, Any]]:
    if since_ts is None:
        since_ts = datetime(2025, 11, 20, 0, 0, 0, tzinfo=timezone.utc).isoformat(timespec="seconds")
    if before_ts is None:
        before_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    return get_all_vulns(since_ts=since_ts, before_ts=before_ts, check_qdrant=check_qdrant)

# ================== CLI DEBUG ==================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    records = scrape_variot(check_qdrant=True)

    OUT_JSON = "variot_vulnerabilities_complete_2025-11-20onward.json"
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(records)} vulnerabilities → {OUT_JSON}")

    if records:
        print("\nSample vulnerability (first record):")
        print(json.dumps(records[0], ensure_ascii=False, indent=2))
