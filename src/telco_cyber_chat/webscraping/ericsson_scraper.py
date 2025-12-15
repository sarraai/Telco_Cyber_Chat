from __future__ import annotations

import re
import os
import json
import logging
from typing import Optional, List, Dict, Tuple
from urllib.parse import urljoin, urlparse, urlunparse
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup, Tag, NavigableString

try:
    import httpx  # optional
except Exception:
    httpx = None

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)

# ================== CONFIG ==================
BASE = "https://www.ericsson.com"
START_URL = "https://www.ericsson.com/en/about-us/security/security-bulletins"

OUT_JSON = "ericsson_security_bulletins.json"
VENDOR = "Ericsson"  # ✅ vendor filter value (must match your Qdrant payload exactly)

QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "Telco_CyberChat")

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) "
                   "Chrome/124.0.0.0 Safari/537.36"),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.google.com/",
    "Connection": "keep-alive",
    "sec-ch-ua": '"Chromium";v="124", "Not.A/Brand";v="99", "Google Chrome";v="124"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": '"Windows"',
    "Sec-Fetch-Dest": "document",
    "Sec-Fetch-Mode": "navigate",
    "Sec-Fetch-Site": "none",
    "Upgrade-Insecure-Requests": "1",
}

# ================== REGEXES ==================
CVE_RE          = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I)
CVSS_SCORE_RE   = re.compile(r"\bCVSS(?:\s*Base\s*Score)?\s*[: ]\s*([0-9]+(?:\.[0-9])?)\b", re.I)
CVSS_VECTOR_RE  = re.compile(r"\bCVSS:[0-9]\.[0-9]/[A-Z0-9:/.-]+\b")
SEVERITY_RE     = re.compile(r"\bSeverity\s*:\s*([A-Za-z]+)\b", re.I)

# ================== HELPERS ==================
def _clean(s: str) -> str:
    return " ".join((s or "").replace("\xa0", " ").split())

def _to_iso(d: str) -> str:
    """Convert common date formats to YYYY-MM-DD; else return raw string unchanged."""
    d = _clean(d)
    for fmt in ("%b %d, %Y", "%B %d, %Y", "%d %B %Y", "%d %b %Y"):
        try:
            dt = datetime.strptime(d, fmt)
            return dt.strftime("%Y-%m-%d")
        except Exception:
            continue
    return d

def _strip_noise(html: str) -> str:
    soup = BeautifulSoup(html, "lxml")
    for t in soup(["script", "style", "noscript", "template"]):
        t.decompose()
    return str(soup)

def canonical_url(u: str) -> str:
    """Canonicalize by removing query/fragment (matches your Qdrant 'canonical url' rule)."""
    link = urljoin(BASE, (u or "").strip())
    p = urlparse(link)
    return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))

def _table_headers(table: Tag) -> List[str]:
    ths = [_clean(th.get_text()) for th in table.find_all("th")]
    return [h.lower() for h in ths]

def _dedup_keep_order(seq: List[str]) -> List[str]:
    seen = set()
    out = []
    for x in seq:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def _flatten_updates(rows: List[Dict]) -> Dict[str, List[str]]:
    products, affected, updated, flat = [], [], [], []
    for r in rows or []:
        p = _clean(r.get("product", ""))
        a = _clean(r.get("versions_affected", ""))
        u = _clean(r.get("updated_version", ""))
        if p: products.append(p)
        if a: affected.append(a)
        if u: updated.append(u)
        if p or a or u:
            flat.append(f"Product: {p} | Affected: {a} | Updated: {u}")
    return {
        "updates_product": _dedup_keep_order(products),
        "updates_versions_affected": _dedup_keep_order(affected),
        "updates_updated_version": _dedup_keep_order(updated),
        "updates_flat": _dedup_keep_order(flat),
    }

# ================== QDRANT DEDUPE (vendor + url) ==================
def build_qdrant_client_from_env() -> Optional[QdrantClient]:
    if not QDRANT_URL:
        return None
    if QDRANT_API_KEY:
        return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    return QdrantClient(url=QDRANT_URL)

def vendor_url_already_ingested(
    client: QdrantClient,
    collection_name: str,
    vendor: str,
    url: str,
) -> bool:
    """
    True if a point exists with payload:
      vendor == <vendor> AND url == <url>
    Assumes payload indexes exist for vendor + url (keyword).
    """
    try:
        points, _next = client.scroll(
            collection_name=collection_name,
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
        return len(points) > 0
    except Exception as e:
        logger.error("[ERICSSON] Qdrant check failed for vendor=%s url=%s: %s", vendor, url, e)
        return False

# ================== LISTING TABLE PARSING ==================
def _table_rows_to_list(table: Tag) -> List[Tuple[Dict, str]]:
    out: List[Tuple[Dict, str]] = []
    headers = _table_headers(table)

    def col_idx(name_opts: List[str]) -> Optional[int]:
        for i, h in enumerate(headers):
            for opt in name_opts:
                if opt in h:
                    return i
        return None

    idx_title = col_idx(["title"])
    idx_cve   = col_idx(["cve"])

    for tr in table.select("tr"):
        if tr.find("th"):
            continue
        tds = tr.find_all("td")
        if not tds:
            continue

        title_cell = tds[idx_title] if idx_title is not None and idx_title < len(tds) else tds[0]
        a = title_cell.find("a")
        title = _clean(a.get_text() if a else title_cell.get_text())
        href  = (a.get("href") or "").strip() if a else ""
        link  = canonical_url(href)

        cve_cell = tds[idx_cve] if idx_cve is not None and idx_cve < len(tds) else None
        cve_text = _clean(cve_cell.get_text(" ", strip=True)) if cve_cell else ""
        cves = [x.upper() for x in dict.fromkeys(CVE_RE.findall(cve_text))]

        row = {"title_list": title, "cves_list": cves}
        out.append((row, link))
    return out

def _extract_table(html: str) -> Optional[Tag]:
    soup = BeautifulSoup(html, "lxml")
    table = soup.select_one("table.eds-table")
    if table:
        return table
    for cand in soup.find_all("table"):
        hdrs = " | ".join(_table_headers(cand))
        if ("title" in hdrs) and ("cve" in hdrs):
            return cand
    return None

def _find_next_page(html: str, current_url: str) -> Optional[str]:
    soup = BeautifulSoup(html, "lxml")
    a = soup.select_one('a[rel="next"]')
    if a and a.get("href"):
        return urljoin(current_url, a["href"])
    for sel in ['a[aria-label*="Next"]', 'a[aria-label*="next"]']:
        a = soup.select_one(sel)
        if a and a.get("href"):
            return urljoin(current_url, a["href"])
    for a in soup.find_all("a"):
        txt = _clean(a.get_text()).lower()
        if txt in {"next", "next ›", "›", "»"} and a.get("href"):
            return urljoin(current_url, a["href"])
    active = soup.select_one("li.active a, li.is-active a")
    if active:
        li = active.find_parent("li")
        if li:
            nxt = li.find_next_sibling("li")
            if nxt:
                na = nxt.find("a")
                if na and na.get("href"):
                    return urljoin(current_url, na["href"])
    return None

# ================== FETCH HTML (ROBUST) ==================
def fetch_html_requests(url: str, timeout=30) -> Optional[str]:
    s = requests.Session()
    s.headers.update(HEADERS)
    try:
        s.get(urljoin(BASE, "/en"), timeout=10)
    except Exception:
        pass
    r = s.get(url, timeout=timeout, allow_redirects=True)
    if r.status_code == 200 and r.text:
        return r.text
    return None

def fetch_html_httpx(url: str, timeout=30) -> Optional[str]:
    if httpx is None:
        return None
    try:
        with httpx.Client(http2=True, headers=HEADERS, timeout=timeout, follow_redirects=True) as client:
            r = client.get(url)
            if r.status_code == 200 and r.text:
                return r.text
    except Exception:
        return None
    return None

def fetch_html_cloudscraper(url: str, timeout=30) -> Optional[str]:
    try:
        import cloudscraper
        scraper = cloudscraper.create_scraper(
            browser={"browser": "chrome", "platform": "windows", "mobile": False}
        )
        r = scraper.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
        if r.status_code == 200 and r.text:
            return r.text
    except Exception:
        return None
    return None

def fetch_html_curl_cffi(url: str, timeout=30) -> Optional[str]:
    try:
        from curl_cffi import requests as curlreq
        r = curlreq.get(url, headers=HEADERS, timeout=timeout, impersonate="chrome")
        if r.status_code == 200 and r.text:
            return r.text
    except Exception:
        return None
    return None

def fetch_html(url: str, timeout=30) -> Optional[str]:
    for getter in (fetch_html_requests, fetch_html_httpx, fetch_html_cloudscraper, fetch_html_curl_cffi):
        html = getter(url, timeout=timeout) if getter != fetch_html_requests else getter(url, timeout)
        if html:
            return _strip_noise(html)
    return None

# ================== DETAIL PAGE PARSING ==================
def _find_heading(soup: BeautifulSoup, text_contains: List[str], tags=("h2","h3","h4","h5")) -> Optional[Tag]:
    wants = [t.lower() for t in text_contains]
    for h in soup.find_all(tags):
        ht = _clean(h.get_text(" ", strip=True)).lower()
        if any(w in ht for w in wants):
            return h
    return None

def _collect_section_text(start_h: Tag) -> str:
    if not start_h:
        return ""
    chunks: List[str] = []
    for el in start_h.next_siblings:
        if isinstance(el, Tag) and el.name in {"h2","h3","h4","h5"}:
            break
        if isinstance(el, Tag):
            if el.name in {"p","ul","ol"}:
                txt = _clean(el.get_text(" ", strip=True))
                if txt:
                    chunks.append(txt)
            elif el.name in {"div","section","article"}:
                inner = []
                for node in el.find_all(["p","li"]):
                    t = _clean(node.get_text(" ", strip=True))
                    if t:
                        inner.append(t)
                if inner:
                    chunks.append(" ".join(inner))
        elif isinstance(el, NavigableString):
            t = _clean(str(el))
            if t:
                chunks.append(t)

    out = []
    for t in chunks:
        if not out or t != out[-1]:
            out.append(t)
    return " ".join(out).strip()

def _parse_affected_table_near(soup: BeautifulSoup, anchor: Tag) -> List[Dict]:
    if not anchor:
        return []
    table = None
    for el in anchor.next_siblings:
        if isinstance(el, Tag) and el.name in {"h2","h3","h4","h5"}:
            break
        if isinstance(el, Tag) and el.name == "table":
            table = el
            break
        if isinstance(el, Tag):
            cand = el.find("table")
            if cand:
                table = cand
                break
    if not table:
        return []

    headers = _table_headers(table)

    def find_idx(opts: List[str]) -> Optional[int]:
        for i, h in enumerate(headers):
            for o in opts:
                if o in h:
                    return i
        return None

    idx_prod = find_idx(["product"])
    idx_aff  = find_idx(["affected", "version"])
    idx_upd  = find_idx(["updated version", "update version", "fixed", "fix", "remedy", "release"])

    rows = []
    for tr in table.find_all("tr"):
        if tr.find("th"):
            continue
        tds = tr.find_all("td")
        if not tds:
            continue

        def cell(i):
            return _clean(tds[i].get_text(" ", strip=True)) if i is not None and i < len(tds) else ""

        prod = cell(idx_prod)
        aff  = cell(idx_aff)
        upd  = cell(idx_upd)
        if prod or aff or upd:
            rows.append({"product": prod, "versions_affected": aff, "updated_version": upd})
    return rows

def _parse_revision_history(soup: BeautifulSoup) -> List[Dict]:
    h = _find_heading(soup, ["revision history"])
    if not h:
        return []
    table = None
    for el in h.next_siblings:
        if isinstance(el, Tag) and el.name in {"h2","h3","h4","h5"}:
            break
        if isinstance(el, Tag) and el.name == "table":
            table = el
            break
        if isinstance(el, Tag):
            t = el.find("table")
            if t:
                table = t
                break
    if not table:
        return []

    headers = _table_headers(table)

    def idx(opts: List[str]) -> Optional[int]:
        for i, h in enumerate(headers):
            for o in opts:
                if o in h:
                    return i
        return None

    i_rev  = idx(["revision"])
    i_date = idx(["date"])
    i_desc = idx(["description", "change"])

    out = []
    for tr in table.find_all("tr"):
        if tr.find("th"):
            continue
        tds = tr.find_all("td")
        if not tds:
            continue

        def cell(i):
            return _clean(tds[i].get_text(" ", strip=True)) if i is not None and i < len(tds) else ""

        out.append({
            "revision": cell(i_rev),
            "date_iso": _to_iso(cell(i_date)),
            "description": cell(i_desc),
        })
    return out

def _extract_cvss_bits(text: str):
    if not text:
        return None, None, None

    score = None
    vector = None
    severity = None

    m1 = CVSS_SCORE_RE.search(text)
    if m1:
        try:
            score = float(m1.group(1))
        except Exception:
            score = None

    m2 = CVSS_VECTOR_RE.search(text)
    if m2:
        vector = m2.group(0)

    m3 = SEVERITY_RE.search(text)
    if m3:
        severity = m3.group(1).capitalize()

    return score, vector, severity

def parse_detail_page(detail_url: str) -> Dict:
    html = fetch_html(detail_url)
    if not html:
        return {}
    soup = BeautifulSoup(html, "lxml")

    title_tag = soup.find("title")
    page_title = _clean(title_tag.get_text()) if title_tag else ""

    h_desc = _find_heading(soup, ["vulnerability description", "description"])
    description = _collect_section_text(h_desc)

    h_su = _find_heading(soup, ["security update", "security updates"])
    security_update = _collect_section_text(h_su)

    affected_rows = _parse_affected_table_near(soup, h_su)
    flat_updates = _flatten_updates(affected_rows)

    h_mit = _find_heading(soup, ["mitigations", "mitigation", "workaround"])
    mitigations = _collect_section_text(h_mit)

    revision_history = _parse_revision_history(soup)

    cves_detail = sorted(set(x.upper() for x in CVE_RE.findall(html)))

    cvss_score, cvss_vector, severity = _extract_cvss_bits(description or "")
    if cvss_score is None and cvss_vector is None and severity is None:
        cvss_score, cvss_vector, severity = _extract_cvss_bits(_clean(soup.get_text(" ", strip=True)))

    return {
        "title": page_title or "",
        "description": description or "",
        "security_update": security_update or "",
        **flat_updates,
        "mitigations": mitigations or "",
        "revision_history": revision_history,
        "cves_detail": cves_detail,
        "cvss_score": cvss_score,
        "cvss_vector": cvss_vector,
        "severity": severity,
    }

# ================== MAIN CRAWL ==================
def fetch_all_bulletins(
    start_url: str = START_URL,
    check_qdrant: bool = True,
    qdrant_client: Optional[QdrantClient] = None,
    collection_name: str = QDRANT_COLLECTION,
) -> List[Dict]:
    """
    - Crawl listing pages
    - For each bulletin: canonicalize URL
    - ✅ If (vendor,url) exists in Qdrant: skip BEFORE detail scraping
    - Else: scrape detail page + return structured record
    """
    client = qdrant_client
    if check_qdrant and client is None:
        client = build_qdrant_client_from_env()
        if client is None:
            logger.warning("[ERICSSON] check_qdrant=True but QDRANT_URL not set; running without dedupe.")
            check_qdrant = False

    seen_links = set()
    results: List[Dict] = []

    url = start_url
    while url:
        html = fetch_html(url)
        if not html:
            break

        table = _extract_table(html)
        if not table:
            break

        rows = _table_rows_to_list(table)
        for row, detail_url_raw in rows:
            if not detail_url_raw:
                continue

            detail_url = canonical_url(detail_url_raw)
            if detail_url in seen_links:
                continue
            seen_links.add(detail_url)

            # ✅ Qdrant dedupe BEFORE detail scraping
            if check_qdrant and client is not None:
                if vendor_url_already_ingested(client, collection_name, VENDOR, detail_url):
                    logger.info("[ERICSSON] Skipping already-ingested: vendor=%s url=%s", VENDOR, detail_url)
                    continue

            detail = {}
            try:
                detail = parse_detail_page(detail_url)
            except Exception as e:
                logger.warning("[ERICSSON] Detail parse failed %s: %s", detail_url, e)

            rec = {
                "vendor": VENDOR,
                "url": detail_url,  # ✅ canonical url used for Qdrant + stored
                "scraped_date": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),

                "title": detail.get("title") or row.get("title_list") or "",
                "description": detail.get("description") or "",
                "security_update": detail.get("security_update") or "",

                # flattened updates
                "updates_product": detail.get("updates_product") or [],
                "updates_versions_affected": detail.get("updates_versions_affected") or [],
                "updates_updated_version": detail.get("updates_updated_version") or [],
                "updates_flat": detail.get("updates_flat") or [],

                "mitigations": detail.get("mitigations") or "",
                "revision_history": detail.get("revision_history") or [],
                "cves": sorted(set((row.get("cves_list") or []) + (detail.get("cves_detail") or []))),
                "cvss_score": detail.get("cvss_score"),
                "cvss_vector": detail.get("cvss_vector"),
                "severity": detail.get("severity"),
            }
            results.append(rec)

        next_url = _find_next_page(html, url)
        url = next_url if next_url and next_url != url else None

    return results

def save_json(rows: List[Dict], path: str = OUT_JSON) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(rows, f, ensure_ascii=False, indent=2)

# ===== Public entrypoint for your pipeline =====
def scrape_ericsson(check_qdrant: bool = True) -> List[Dict]:
    return fetch_all_bulletins(check_qdrant=check_qdrant)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    rows = scrape_ericsson(check_qdrant=True)
    if rows:
        save_json(rows, OUT_JSON)
        print(f"✅ Fetched {len(rows)} bulletins → {OUT_JSON}")
        print(json.dumps(rows[:1], ensure_ascii=False, indent=2))
    else:
        print("No bulletins found (blocked, layout changed, or pagination not detected).")
