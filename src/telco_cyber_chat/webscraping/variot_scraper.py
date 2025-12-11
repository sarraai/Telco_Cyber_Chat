import os
import json
import time
import logging
import requests
from datetime import datetime, timezone
from requests.adapters import HTTPAdapter, Retry
from bs4 import BeautifulSoup  # <--- HTML parsing

from typing import Dict, Any, List, Optional

from telco_cyber_chat.webscraping.scrape_core import url_already_ingested

# ========= LOGGER =========
logger = logging.getLogger(__name__)

# ========= Config =========
API_KEY = os.getenv("VARIOT_API_KEY", "83eec50617122449a5867b34b24b261df60f86df")
BASE_URL = "https://www.variotdbs.pl/api"
HEADERS = {"Authorization": f"Token {API_KEY}"}
REQ_TIMEOUT = (10, 45)  # (connect, read)

# ========= Robust session with retries =========
def make_session() -> requests.Session:
    s = requests.Session()
    retries = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.5,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=frozenset(["GET"]),
    )
    s.mount("https://", HTTPAdapter(max_retries=retries))
    s.mount("http://", HTTPAdapter(max_retries=retries))
    # Optional: be polite with a basic UA
    s.headers.update({"User-Agent": "VARIoT-scraper/1.0"})
    return s

# ========= Pagination =========
def fetch_all(url: str, params: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    Fetch all pages from a paginated VARIoT endpoint.
    Uses 'next' field until exhausted.
    """
    results: List[Dict[str, Any]] = []
    next_url = url
    session = make_session()
    page_idx = 0

    while next_url:
        page_idx += 1
        # send params only on first page
        query = params if next_url == url else {}
        r = session.get(next_url, headers=HEADERS, params=query, timeout=REQ_TIMEOUT)
        r.raise_for_status()
        data = r.json()

        if isinstance(data, dict) and "results" in data:
            batch = data["results"] or []
            results.extend(batch)
            next_url = data.get("next")
        elif isinstance(data, list):
            batch = data
            results.extend(batch)
            next_url = None
        else:
            raise ValueError(f"Unexpected response shape: {type(data)}")

        logger.info("[VARIOT] Fetched page %d (%d items)", page_idx, len(batch))
        time.sleep(0.3)  # polite throttle to API

    return results

# ========= Helpers: generic text picker =========

def _pick_text(obj: Any, keys=("description", "summary", "details", "text", "content", "body")) -> str:
    """Generic helper in case VARIoT wraps fields in nested dicts."""
    if not isinstance(obj, dict):
        return str(obj or "").strip()
    for k in keys:
        v = obj.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, dict):
            for dk in ("data", "value", "text", "content"):
                dv = v.get(dk)
                if isinstance(dv, str) and dv.strip():
                    return dv.strip()
    return ""

def extract_description(entry: Dict[str, Any]) -> str:
    v = entry.get("description")
    if isinstance(v, str) and v.strip():
        return v.strip()
    if isinstance(v, dict):
        d = _pick_text(v, keys=("description", "text", "content", "data", "value"))
        if d:
            return d
    # Fallback: try to sniff description-like text anywhere
    d = _pick_text(entry, keys=("description", "summary", "details", "text", "content", "body"))
    return d or ""

def extract_url(entry: Dict[str, Any]) -> Optional[str]:
    # 1) If API already provides url
    u = entry.get("url")
    if isinstance(u, str) and u.startswith("http"):
        return u

    # 2) Otherwise build from ID: https://www.variotdbs.pl/vuln/<ID>/
    vid = entry.get("id")
    if isinstance(vid, str) and vid:
        return f"https://www.variotdbs.pl/vuln/{vid}/"

    return None

# ========= HTML title scraper =========

def fetch_html_title(url: str, session: requests.Session) -> Optional[str]:
    """
    Fetch the vulnerability page and extract <meta property="og:title">,
    with a couple of fallbacks.
    """
    if not url:
        return None

    try:
        resp = session.get(url, timeout=REQ_TIMEOUT)
        resp.raise_for_status()
    except Exception as e:
        logger.warning("[VARIOT] Could not fetch HTML for %s: %s", url, e)
        return None

    html = resp.text
    soup = BeautifulSoup(html, "html.parser")

    # 1) Explicit og:title
    og = soup.find("meta", attrs={"property": "og:title"})
    if og and og.get("content"):
        return og["content"].strip()

    # 2) Fallback: <title> tag
    if soup.title and soup.title.text.strip():
        return soup.title.text.strip()

    # 3) Fallback: first <h1>
    h1 = soup.find("h1")
    if h1:
        text = h1.get_text(strip=True)
        if text:
            return text

    return None

def extract_title(
    entry: Dict[str, Any],
    session: Optional[requests.Session] = None,
    url_hint: Optional[str] = None,
) -> Optional[str]:
    # Prefer direct title/name fields from the API if they exist
    for k in ("title", "name"):
        v = entry.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
        if isinstance(v, dict):
            t = _pick_text(v, keys=("title", "name"))
            if t:
                return t

    # If not present in JSON, try HTML meta og:title
    if session is not None:
        url = url_hint or extract_url(entry)
        if url:
            html_title = fetch_html_title(url, session=session)
            if html_title:
                return html_title

    # Last fallback: first line of description
    d = extract_description(entry)
    return (d.splitlines()[0][:200] if d else None)

# ========= VARIoT crawl with Qdrant dedupe =========

def get_all_vulns(
    since_ts: str,
    before_ts: str,
    check_qdrant: bool = True,
) -> List[Dict[str, Any]]:
    """
    Crawl VARIoT vulnerabilities in the given time window and return minimal docs:

      { "title": ..., "url": ..., "description": ... }

    Pattern:
      - Fetch all entries from /vulns/ with since/before
      - For each entry:
          * derive canonical URL
          * dedupe within this run
          * if check_qdrant=True -> call url_already_ingested(url)
              - if True => skip (no HTML)
              - if False => enrich + build document
      - Log Seen / Skipped / New counts (like Huawei scraper).
    """
    logger.info(
        "[VARIOT] Starting vulnerability crawl (check_qdrant=%s, since=%s, before=%s)",
        check_qdrant, since_ts, before_ts,
    )

    vulns_url = f"{BASE_URL}/vulns/"
    params = {
        "jsonld": "false",
        "limit": 100,
        "since": since_ts,
        "before": before_ts,
    }

    items = fetch_all(vulns_url, params)
    logger.info("[VARIOT] Total fetched from API: %d", len(items))

    docs: List[Dict[str, Any]] = []
    html_session = make_session()  # separate session for HTML pages

    seen_urls: set[str] = set()
    n_seen = 0
    n_skipped = 0
    n_new = 0

    for it in items:
        if not isinstance(it, dict):
            continue

        url = extract_url(it)
        if not url:
            continue

        # Deduplicate within this run
        if url in seen_urls:
            continue
        seen_urls.add(url)

        # Count distinct advisory URL as "seen"
        n_seen += 1

        # Qdrant dedupe BEFORE HTML fetch
        if check_qdrant:
            try:
                if url_already_ingested(url):
                    n_skipped += 1
                    logger.info("[VARIOT] Skipping already-ingested URL: %s", url)
                    continue
            except Exception as e:
                logger.error("[VARIOT] Qdrant check failed for %s: %s", url, e)
                # Conservative: fall through and scrape anyway

        # Considered NEW for this run
        n_new += 1

        # Build minimal document (title, description, url)
        title = extract_title(it, session=html_session, url_hint=url)
        desc = extract_description(it)

        if not title:
            title = "VARIoT Vulnerability"
        if not desc:
            desc = title

        docs.append(
            {
                "title": title,
                "description": desc,
                "url": url,
            }
        )

        # Optional tiny pause between HTML hits to be polite
        time.sleep(0.1)

    logger.info(
        "[VARIOT] Seen=%d, skipped=%d (already in Qdrant), new=%d",
        n_seen, n_skipped, n_new,
    )
    logger.info("[VARIOT] Scraped %d new vulnerabilities (documents).", len(docs))
    return docs

# ========= Public entrypoint (Huawei-style pattern) =========

def scrape_variot(
    check_qdrant: bool = True,
    since_ts: Optional[str] = None,
    before_ts: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    High-level scraper used by your cron / scraping graph.

      - Default window: from 2025-11-20T00:00:00Z up to now
      - Calls get_all_vulns(...)
      - Returns list of {title, url, description} documents
    """
    # 🔒 Default: only from 2025-11-20 onward (UTC)
    if since_ts is None:
        since_dt = datetime(2025, 11, 20, 0, 0, 0, tzinfo=timezone.utc)
        since_ts = since_dt.isoformat(timespec="seconds")
    if before_ts is None:
        before_ts = datetime.now(timezone.utc).isoformat(timespec="seconds")

    return get_all_vulns(since_ts=since_ts, before_ts=before_ts, check_qdrant=check_qdrant)

# ========= CLI DEBUG (LOCAL USE ONLY) =========

if __name__ == "__main__":
    # Example: Fetch only items updated since 2025-11-20 (onward)
    since_dt = datetime(2025, 11, 20, 0, 0, 0, tzinfo=timezone.utc)
    before_dt = datetime.now(timezone.utc)

    since_ts = since_dt.isoformat(timespec="seconds")
    before_ts = before_dt.isoformat(timespec="seconds")

    print(f"Fetching VARIoT vulnerabilities updated since {since_ts} …")
    # Local run: don't check Qdrant to avoid needing DB config in a notebook
    records = scrape_variot(check_qdrant=False, since_ts=since_ts, before_ts=before_ts)

    print(f"Total new docs built: {len(records)}")

    OUT_JSON = "variot_vulnerabilities_2025-11-20onward_title_desc_url.json"
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    print(f"Saved → {OUT_JSON}")

    # Print the first JSON object for inspection
    if records:
        print("\nFirst normalized record:")
        print(json.dumps(records[0], ensure_ascii=False, indent=2))
    else:
        print("No vulnerabilities found for this range.")
