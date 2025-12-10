import re
import logging
from typing import Optional, List, Dict, Tuple
from urllib.parse import urljoin, urlparse, urlunparse
from datetime import datetime

import requests
from bs4 import BeautifulSoup, Tag, NavigableString

try:
    import httpx  # optional
except Exception:
    httpx = None

from telco_cyber_chat.webscraping.scrape_core import url_already_ingested

# ================== LOGGER ==================

logger = logging.getLogger(__name__)

# ================== CONFIG ==================

BASE = "https://www.ericsson.com"
URL = "https://www.ericsson.com/en/about-us/security/security-bulletins"

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
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

CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I)
CVSS_SCORE_RE = re.compile(
    r"\bCVSS(?:\s*Base\s*Score)?\s*[: ]\s*([0-9]+(?:\.[0-9])?)\b", re.I
)
CVSS_VECTOR_RE = re.compile(r"\bCVSS:[0-9]\.[0-9]/[A-Z0-9:/.-]+\b")
SEVERITY_RE = re.compile(r"\bSeverity\s*:\s*([A-Za-z]+)\b", re.I)

# ================== BASIC HELPERS ==================


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
    return d  # fallback


def _strip_noise(html: str) -> str:
    soup = BeautifulSoup(html, "html.parser")
    for t in soup(["script", "style", "noscript", "template"]):
        t.decompose()
    return str(soup)


def _normalize_link(href: str) -> str:
    link = urljoin(BASE, (href or "").strip())
    u = urlparse(link)
    # remove query/fragment for consistency
    return urlunparse((u.scheme, u.netloc, u.path, "", "", ""))


def _table_headers(table: Tag) -> List[str]:
    ths = [_clean(th.get_text()) for th in table.find_all("th")]
    return [h.lower() for h in ths]


def _table_rows_to_list(table: Tag) -> List[Tuple[Dict, str]]:
    """
    Parse listing table rows.
    Returns a list of (row_dict_without_link, detail_url) tuples.
    """
    out: List[Tuple[Dict, str]] = []
    headers = _table_headers(table)

    def col_idx(name_opts: List[str]) -> Optional[int]:
        for i, h in enumerate(headers):
            for opt in name_opts:
                if opt in h:
                    return i
        return None

    idx_title = col_idx(["title"])
    idx_cve = col_idx(["cve"])
    idx_pub = col_idx(["published"])
    idx_updated = col_idx(["updated", "last updated"])

    for tr in table.select("tr"):
        if tr.find("th"):
            continue
        tds = tr.find_all("td")
        if not tds:
            continue

        # Title & link
        title_cell = (
            tds[idx_title] if idx_title is not None and idx_title < len(tds) else tds[0]
        )
        a = title_cell.find("a")
        title = _clean(a.get_text() if a else title_cell.get_text())
        href = (a.get("href") or "").strip() if a else ""
        link = _normalize_link(href)

        # CVEs (from listing cell)
        cve_cell = tds[idx_cve] if idx_cve is not None and idx_cve < len(tds) else None
        cve_text = _clean(cve_cell.get_text(" ", strip=True)) if cve_cell else ""
        cves = [x.upper() for x in dict.fromkeys(CVE_RE.findall(cve_text))]

        # Dates
        pub_cell = tds[idx_pub] if idx_pub is not None and idx_pub < len(tds) else None
        upd_cell = (
            tds[idx_updated]
            if idx_updated is not None and idx_updated < len(tds)
            else None
        )
        published = _to_iso(pub_cell.get_text(" ", strip=True)) if pub_cell else ""
        updated = _to_iso(upd_cell.get_text(" ", strip=True)) if upd_cell else ""

        row = {
            "title_list": title,  # temporary (detail <title> may override)
            "cves_list": cves,
            "published_date": published,
            "last_updated": updated,
        }
        out.append((row, link))
    return out


def _extract_table(html: str) -> Optional[Tag]:
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("table.eds-table")
    if table:
        return table
    for cand in soup.find_all("table"):
        hdrs = " | ".join(_table_headers(cand))
        if all(k in hdrs for k in ["title", "cve", "published", "updated"]):
            return cand
    return None


def _find_next_page(html: str, current_url: str) -> Optional[str]:
    soup = BeautifulSoup(html, "html.parser")
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
        with httpx.Client(
            http2=True, headers=HEADERS, timeout=timeout, follow_redirects=True
        ) as client:
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
    for getter in (
        fetch_html_requests,
        fetch_html_httpx,
        fetch_html_cloudscraper,
        fetch_html_curl_cffi,
    ):
        html = getter(url, timeout=timeout) if getter != fetch_html_requests else getter(
            url, timeout
        )
        if html:
            return _strip_noise(html)
    return None


# ================== DETAIL PAGE PARSING ==================


def _find_heading(
    soup: BeautifulSoup, text_contains: List[str], tags=("h2", "h3", "h4", "h5")
) -> Optional[Tag]:
    wants = [t.lower() for t in text_contains]
    for h in soup.find_all(tags):
        ht = _clean(h.get_text(" ", strip=True)).lower()
        if any(w in ht for w in wants):
            return h
    return None


def _collect_section_text(start_h: Tag) -> str:
    """Collect paragraphs/lists under heading until the next heading tag."""
    if not start_h:
        return ""
    chunks: List[str] = []
    for el in start_h.next_siblings:
        if isinstance(el, Tag) and el.name in {"h2", "h3", "h4", "h5"}:
            break
        if isinstance(el, Tag):
            if el.name in {"p", "ul", "ol"}:
                txt = _clean(el.get_text(" ", strip=True))
                if txt:
                    chunks.append(txt)
            elif el.name in {"div", "section", "article"}:
                inner = []
                for node in el.find_all(["p", "li"]):
                    t = _clean(node.get_text(" ", strip=True))
                    if t:
                        inner.append(t)
                if inner:
                    chunks.append(" ".join(inner))
        elif isinstance(el, NavigableString):
            t = _clean(str(el))
            if t:
                chunks.append(t)
    out: List[str] = []
    for t in chunks:
        if not out or t != out[-1]:
            out.append(t)
    return " ".join(out).strip()


def _parse_affected_table_near(soup: BeautifulSoup, anchor: Tag) -> List[Dict]:
    """
    Find the next table after Security update section and parse rows.
    """
    if not anchor:
        return []
    table = None
    for el in anchor.next_siblings:
        if isinstance(el, Tag) and el.name in {"h2", "h3", "h4", "h5"}:
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
    idx_aff = find_idx(["affected", "version"])
    idx_upd = find_idx(
        ["updated version", "update version", "fixed", "fix", "remedy", "release"]
    )

    rows: List[Dict] = []
    for tr in table.find_all("tr"):
        if tr.find("th"):
            continue
        tds = tr.find_all("td")
        if not tds:
            continue

        def cell(i: Optional[int]) -> str:
            return (
                _clean(tds[i].get_text(" ", strip=True))
                if i is not None and i < len(tds)
                else ""
            )

        prod = cell(idx_prod)
        aff = cell(idx_aff)
        upd = cell(idx_upd)
        if prod or aff or upd:
            rows.append(
                {
                    "product": prod,
                    "versions_affected": aff,
                    "updated_version": upd,
                }
            )
    return rows


def _parse_revision_history(soup: BeautifulSoup) -> List[Dict]:
    h = _find_heading(soup, ["revision history"])
    if not h:
        return []
    table = None
    for el in h.next_siblings:
        if isinstance(el, Tag) and el.name in {"h2", "h3", "h4", "h5"}:
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

    i_rev = idx(["revision"])
    i_date = idx(["date"])
    i_desc = idx(["description", "change"])

    out: List[Dict] = []
    for tr in table.find_all("tr"):
        if tr.find("th"):
            continue
        tds = tr.find_all("td")
        if not tds:
            continue

        def cell(i: Optional[int]) -> str:
            return (
                _clean(tds[i].get_text(" ", strip=True))
                if i is not None and i < len(tds)
                else ""
            )

        rev = cell(i_rev)
        dat = cell(i_date)
        desc = cell(i_desc)
        out.append(
            {
                "revision": rev,
                "date_iso": _to_iso(dat),
                "description": desc,
            }
        )
    return out


def _extract_cvss_bits(text: str):
    """
    Returns (cvss_score_float_or_None, cvss_vector_str_or_None, severity_text_or_None).
    Severity is taken ONLY from explicit 'Severity: High' style text.
    """
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
        logger.warning("[ERICSSON] Empty HTML for detail page: %s", detail_url)
        return {}
    soup = BeautifulSoup(html, "html.parser")

    # title from <title>
    title_tag = soup.find("title")
    page_title = _clean(title_tag.get_text()) if title_tag else ""

    # <meta name="description"> → Summary (strip leading "Summary:")
    meta_desc = soup.find("meta", attrs={"name": "description"})
    summary = _clean(meta_desc["content"]) if meta_desc and meta_desc.get("content") else ""
    if summary.lower().startswith("summary:"):
        summary = summary[len("summary:") :].strip()

    # Optional banner: <h5>Updated Month DD, YYYY</h5>
    page_updated = ""
    for h5 in soup.find_all("h5"):
        txt = _clean(h5.get_text())
        if txt.lower().startswith("updated"):
            page_updated = _to_iso(txt.replace("Updated", "").strip().strip(":"))
            break

    # Sections
    h_vuln = _find_heading(soup, ["vulnerability description"])
    vulnerability_description = _collect_section_text(h_vuln)

    h_su = _find_heading(soup, ["security update", "security updates"])
    security_update = _collect_section_text(h_su)
    affected_products = _parse_affected_table_near(soup, h_su)

    h_mit = _find_heading(soup, ["mitigations", "mitigation", "workaround"])
    mitigations = _collect_section_text(h_mit)

    revision_history = _parse_revision_history(soup)

    # CVEs anywhere on page
    cves_detail = sorted(set(x.upper() for x in CVE_RE.findall(html)))

    # CVSS bits (prefer vuln section; fallback to full text)
    cvss_score, cvss_vector, severity = _extract_cvss_bits(
        vulnerability_description or ""
    )
    if cvss_score is None and cvss_vector is None and severity is None:
        cvss_score, cvss_vector, severity = _extract_cvss_bits(
            _clean(soup.get_text(" ", strip=True))
        )

    return {
        "title": page_title or "",
        "summary": summary or "",
        "vulnerability_description": vulnerability_description or "",
        "security_update": security_update or "",
        "affected_products": affected_products,
        "mitigations": mitigations or "",
        "revision_history": revision_history,
        "page_updated": page_updated or "",
        "cves_detail": cves_detail,
        "cvss_score": cvss_score,
        "cvss_vector": cvss_vector,
        "severity": severity,
    }


# ================== LIST CRAWL → INTERNAL RECORDS ==================


def fetch_all_bulletins(check_qdrant: bool = True) -> List[Dict[str, object]]:
    """
    Crawl Ericsson security bulletins listing and details.

    Returns internal records with:
      - url
      - title
      - summary, vulnerability_description, security_update
      - affected_products, mitigations, revision_history
      - page_updated, cves, published_date, last_updated
      - cvss_score, cvss_vector, severity

    If check_qdrant=True, we skip URLs that are already stored in Qdrant
    BEFORE fetching the detail page.
    """
    seen_links: set[str] = set()
    results: List[Dict[str, object]] = []

    url = URL
    while url:
        html = fetch_html(url)
        if not html:
            logger.warning("[ERICSSON] Failed to fetch listing page: %s", url)
            break

        table = _extract_table(html)
        if not table:
            logger.warning("[ERICSSON] Could not find bulletins table on: %s", url)
            break

        rows = _table_rows_to_list(table)  # List[(row_dict, detail_url)]

        for row, detail_url in rows:
            if not detail_url or detail_url in seen_links:
                continue
            seen_links.add(detail_url)

            # ✅ Qdrant dedupe BEFORE scraping detail page
            if check_qdrant and detail_url and url_already_ingested(detail_url):
                logger.info("[ERICSSON] Skipping already-ingested URL: %s", detail_url)
                continue

            try:
                detail = parse_detail_page(detail_url)
            except Exception as e:
                logger.warning(
                    "[ERICSSON] Failed to parse detail page %s: %s", detail_url, e
                )
                detail = {}

            title = (
                detail.get("title")
                or row.get("title_list")
                or ""
            )

            cves_combined = sorted(
                set((row.get("cves_list") or []) + (detail.get("cves_detail") or []))
            )

            rec: Dict[str, object] = {
                "url": detail_url,
                "title": title,
                "summary": detail.get("summary") or "",
                "vulnerability_description": detail.get(
                    "vulnerability_description"
                )
                or "",
                "security_update": detail.get("security_update") or "",
                "affected_products": detail.get("affected_products") or [],
                "mitigations": detail.get("mitigations") or "",
                "revision_history": detail.get("revision_history") or [],
                "page_updated": detail.get("page_updated") or "",
                "cves": cves_combined,
                "published_date": row.get("published_date") or "",
                "last_updated": row.get("last_updated") or "",
                "cvss_score": detail.get("cvss_score"),
                "cvss_vector": detail.get("cvss_vector"),
                "severity": detail.get("severity"),
            }

            results.append(rec)

        next_url = _find_next_page(html, url)
        url = next_url if next_url and next_url != url else None

    logger.info("[ERICSSON] Collected %d bulletins (internal records).", len(results))
    return results


# ================== RECORD → {url, title, description} ==================


def ericsson_record_to_document(rec: Dict[str, object]) -> Dict[str, str]:
    """
    Build a minimal document for RAG ingestion:

      - url
      - title
      - description = merged:
          vulnerability_description (as main description),
          security_update,
          affected_products,
          mitigations,
          CVEs,
          CVSS (score, vector, severity)
    """
    url = (rec.get("url") or "").strip()  # type: ignore[union-attr]
    title = (
        (rec.get("title") or "Ericsson Security Bulletin")  # type: ignore[operator]
        .strip()
    )

    parts: List[str] = []

    # 1) Core description: vulnerability_description ONLY
    vdesc = rec.get("vulnerability_description")
    if isinstance(vdesc, str) and vdesc.strip():
        parts.append("Description: " + vdesc.strip())

    # 2) Security update
    s_update = rec.get("security_update")
    if isinstance(s_update, str) and s_update.strip():
        parts.append("Security update: " + s_update.strip())

    # 3) Affected products table
    affected = rec.get("affected_products") or []
    if isinstance(affected, list) and affected:
        lines: List[str] = []
        for row in affected:
            if not isinstance(row, dict):
                continue
            prod = (row.get("product") or "").strip()
            vers_aff = (row.get("versions_affected") or "").strip()
            upd = (row.get("updated_version") or "").strip()
            bits: List[str] = []
            if prod:
                bits.append(prod)
            if vers_aff:
                bits.append(f"affected versions: {vers_aff}")
            if upd:
                bits.append(f"updated version: {upd}")
            if bits:
                lines.append(" – ".join(bits))
        if lines:
            parts.append("Affected products:\n" + "\n".join(lines))

    # 4) Mitigations
    mit = rec.get("mitigations")
    if isinstance(mit, str) and mit.strip():
        parts.append("Mitigations: " + mit.strip())

    # 5) CVEs
    cves = rec.get("cves") or []
    if isinstance(cves, list) and cves:
        parts.append("Related CVEs: " + ", ".join(map(str, cves)))

    # 6) CVSS (score, vector, severity)
    score = rec.get("cvss_score")
    vec = rec.get("cvss_vector")
    sev = rec.get("severity")
    cvss_bits: List[str] = []
    if score is not None:
        cvss_bits.append(f"score {score}")
    if vec:
        cvss_bits.append(f"vector {vec}")
    if sev:
        cvss_bits.append(f"severity {sev}")
    if cvss_bits:
        parts.append("CVSS: " + ", ".join(cvss_bits))

    description = "\n".join(parts).strip()

    return {
        "url": url,
        "title": title,
        "description": description or title,
    }


# ================== PUBLIC ENTRYPOINT ==================


def scrape_ericsson(check_qdrant: bool = True) -> List[Dict[str, str]]:
    """
    High-level scraper used by your cron / scraping graph.

      - Calls fetch_all_bulletins(check_qdrant)
      - Returns list of documents:
            {url, title, description}
    """
    records = fetch_all_bulletins(check_qdrant=check_qdrant)
    docs: List[Dict[str, str]] = []

    for rec in records:
        url = (rec.get("url") or "").strip()  # type: ignore[union-attr]
        if not url:
            continue
        doc = ericsson_record_to_document(rec)
        docs.append(doc)

    logger.info("[ERICSSON] Scraped %d bulletins (documents).", len(docs))
    return docs


if __name__ == "__main__":
    # Manual test (no Qdrant check)
    docs = scrape_ericsson(check_qdrant=False)
    print(f"Sample docs: {len(docs)}")
    if docs:
        import json

        print(json.dumps(docs[0], indent=2, ensure_ascii=False))
