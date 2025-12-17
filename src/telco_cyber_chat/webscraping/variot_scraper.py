import os
import json
import time
import logging
import hashlib
from datetime import datetime, timezone
from typing import Dict, Any, List, Optional, Set, Tuple
from urllib.parse import urlparse, urlunparse
import sys   # ✅ add this
import requests
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup, Tag

# Fallback (only used if Qdrant env isn't set)
from telco_cyber_chat.webscraping.scrape_core import url_already_ingested

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

from llama_index.core.schema import TextNode

# ================== LOGGER ==================
logger = logging.getLogger("telco_cyber_chat.webscraping.variot")

# (Optional) hard-reset handlers to avoid duplicates / old stderr handlers
# for hh in list(logger.handlers):
#     logger.removeHandler(hh)

if not logger.handlers:
    h = logging.StreamHandler(stream=sys.stdout)  # ✅ stdout (prevents ERROR-looking logs)
    h.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [VARIOT] %(message)s"))
    logger.addHandler(h)

logger.setLevel(logging.INFO)
logger.propagate = False  # ✅ prevents double logging via root logger

# ================== CONFIG ==================
# IMPORTANT: this must match what you store in Qdrant payload for vendor filtering
VENDOR = (os.getenv("VARIOT_VENDOR_VALUE") or "variot").strip()

BASE_URL = "https://www.variotdbs.pl/api"
REQ_TIMEOUT = (10, 45)  # (connect, read)

QDRANT_URL = (os.getenv("QDRANT_URL") or "").strip()
QDRANT_API_KEY = (os.getenv("QDRANT_API_KEY") or "").strip()
QDRANT_COLLECTION = (os.getenv("QDRANT_COLLECTION") or "Telco_CyberChat").strip()

# ================== SECRETS HELPERS ==================
def get_variot_api_key() -> str:
    # 1) Colab secrets
    try:
        from google.colab import userdata
        tok = userdata.get("VARIOT_API_KEY")
        if tok:
            return str(tok).strip()
    except Exception:
        pass
    # 2) Env
    tok = (os.getenv("VARIOT_API_KEY") or "").strip()
    return tok

def make_headers(api_key: str) -> Dict[str, str]:
    if not api_key:
        return {}
    return {"Authorization": f"Token {api_key}"}

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
def fetch_all(url: str, params: Dict[str, Any], headers: Dict[str, str]) -> List[Dict[str, Any]]:
    results: List[Dict[str, Any]] = []
    next_url = url
    session = make_session()

    while next_url:
        query = params if next_url == url else {}
        r = session.get(next_url, headers=headers, params=query, timeout=REQ_TIMEOUT)
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
        logger.warning("Could not fetch %s: %s", url, e)
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

# ================== QDRANT DEDUPE (vendor -> preload urls) ==================
def build_qdrant_client_from_env() -> Optional[QdrantClient]:
    if not QDRANT_URL:
        return None
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(url=QDRANT_URL)

def fetch_existing_vendor_urls(
    client: QdrantClient,
    collection_name: str,
    vendor_value: str,
    page_size: int = 256,
) -> Set[str]:
    existing: Set[str] = set()
    offset = None
    vendor_filter = Filter(
        must=[FieldCondition(key="vendor", match=MatchValue(value=vendor_value))]
    )

    while True:
        points, offset = client.scroll(
            collection_name=collection_name,
            scroll_filter=vendor_filter,
            limit=page_size,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )
        for p in points:
            payload = p.payload or {}
            u = payload.get("url")
            if isinstance(u, str) and u.strip():
                existing.add(u.strip())
        if offset is None:
            break

    return existing

def already_ingested_fallback(url: str) -> bool:
    """Fallback if no Qdrant env is set."""
    return url_already_ingested(canonicalize_url(url))

# ================== TextNode builder (RULE: text=all fields except url, metadata=only url) ==================
def _stable_id(vendor: str, url: str) -> str:
    raw = f"{vendor}|{url}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _dump_json(v: Any) -> str:
    try:
        return json.dumps(v, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        return str(v)

def record_to_text(rec: Dict[str, Any]) -> str:
    lines: List[str] = []
    for k in sorted(rec.keys(), key=lambda s: str(s).lower()):
        ks = str(k).strip()
        if not ks or ks.lower() == "url":
            continue
        v = rec.get(k)
        if v is None:
            continue
        if isinstance(v, str) and not v.strip():
            continue
        if isinstance(v, (list, dict)) and len(v) == 0:
            continue

        if isinstance(v, (dict, list)):
            lines.append(f"{ks}:\n{_dump_json(v)}")
        else:
            lines.append(f"{ks}: {str(v).strip()}")
    return "\n".join(lines).strip()

def create_text_node_from_record(rec: Dict[str, Any]) -> Optional[TextNode]:
    url = rec.get("url")
    if not isinstance(url, str) or not url.strip():
        return None
    url = url.strip()

    # ensure vendor is in TEXT
    rec2 = dict(rec)
    rec2["vendor"] = (rec2.get("vendor") or VENDOR)

    text = record_to_text(rec2)
    if not text:
        return None

    return TextNode(
        id_=_stable_id(str(rec2["vendor"]), url),
        text=text,
        metadata={"url": url},  # ONLY url
    )

# ================== MAIN PROCESSING ==================
def get_all_vulns(
    since_ts: str,
    before_ts: str,
    check_qdrant: bool = True,
) -> Tuple[List[Dict[str, Any]], List[TextNode], Dict[str, int]]:
    api_key = get_variot_api_key()
    if not api_key:
        raise RuntimeError("VARIOT_API_KEY is missing. Set it in env/Colab secrets.")

    headers = make_headers(api_key)
    vulns_url = f"{BASE_URL}/vulns/"
    params = {"jsonld": "false", "limit": 100, "since": since_ts, "before": before_ts}

    logger.info("Fetching API since=%s before=%s", since_ts, before_ts)
    items = fetch_all(vulns_url, params, headers=headers)
    logger.info("Total fetched from API: %d", len(items))

    html_session = make_session()

    # Preload Qdrant existing urls for this vendor (fast)
    existing_urls: Set[str] = set()
    qc = None
    if check_qdrant:
        qc = build_qdrant_client_from_env()
        if qc is not None:
            try:
                existing_urls = fetch_existing_vendor_urls(qc, QDRANT_COLLECTION, VENDOR)
                logger.info("Loaded %d existing URLs from Qdrant for vendor=%s", len(existing_urls), VENDOR)
            except Exception as e:
                logger.warning("Failed to preload existing URLs (%s). Will fallback to url_already_ingested.", e)
                existing_urls = set()
        else:
            logger.warning("check_qdrant=True but QDRANT_URL not set; using url_already_ingested() fallback.")
            check_qdrant = False

    records: List[Dict[str, Any]] = []
    nodes: List[TextNode] = []

    seen_urls: Set[str] = set()
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

        # ✅ Dedup BEFORE scraping HTML
        if check_qdrant and existing_urls:
            if url in existing_urls:
                n_skipped += 1
                continue
        elif check_qdrant and not existing_urls:
            # (shouldn’t happen often; but safe)
            try:
                if already_ingested_fallback(url):
                    n_skipped += 1
                    continue
            except Exception:
                pass
        else:
            # fallback mode only
            if already_ingested_fallback(url):
                n_skipped += 1
                continue

        n_new += 1
        logger.info("[%d/%d] Scraping %s", idx, len(items), url)

        vuln_data = scrape_vulnerability_page(url, html_session)
        if vuln_data:
            vuln_data["vendor"] = VENDOR
            vuln_data["scraped_date"] = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
            records.append(vuln_data)

            node = create_text_node_from_record(vuln_data)
            if node:
                nodes.append(node)

        time.sleep(0.5)

    stats = {"seen": n_seen, "skipped": n_skipped, "new": n_new, "output_records": len(records), "output_nodes": len(nodes)}
    logger.info("Seen=%d skipped=%d new=%d records=%d nodes=%d",
                n_seen, n_skipped, n_new, len(records), len(nodes))

    return records, nodes, stats

def scrape_variot_nodes(
    check_qdrant: bool = True,
    since_ts: Optional[str] = None,
    before_ts: Optional[str] = None,
    return_records: bool = False,
) -> Dict[str, Any]:
    if since_ts is None:
        since_ts = datetime(2025, 11, 20, 0, 0, 0, tzinfo=timezone.utc).isoformat(timespec="seconds")
    if before_ts is None:
        before_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    records, nodes, stats = get_all_vulns(
        since_ts=since_ts,
        before_ts=before_ts,
        check_qdrant=check_qdrant,
    )

    out: Dict[str, Any] = {
        "ok": True,
        "vendor": VENDOR,
        "nodes": nodes,  # ✅ for embedding/upsert module
        "per_source": {"variot": len(nodes)},
        "stats": stats,
    }
    if return_records:
        out["records"] = records
    return out

# ================== CLI DEBUG ==================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    res = scrape_variot_nodes(check_qdrant=True, return_records=True)

    # Save raw dicts (optional)
    OUT_JSON = "variot_vulnerabilities_complete_2025-11-20onward.json"
    recs = res.get("records") or []
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(recs, f, ensure_ascii=False, indent=2)

    print(f"\n✅ Saved {len(recs)} vulnerabilities → {OUT_JSON}")
    print("✅ New TextNodes:", len(res["nodes"]))
    print("Stats:", res["stats"])

    # Preview first node
    if res["nodes"]:
        n0 = res["nodes"][0]
        print("\n--- NODE PREVIEW ---")
        print("id_:", n0.id_)
        print("metadata:", n0.metadata)         # MUST be only {"url": "..."}
        print("text (first 900 chars):\n", n0.text[:900])
