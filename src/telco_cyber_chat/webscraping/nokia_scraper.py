import os
import re
import logging
from typing import List, Dict, Optional, Tuple

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from urllib.parse import urljoin

from telco_cyber_chat.webscraping.scrape_core import url_already_ingested

logger = logging.getLogger(__name__)

# ----------------------------------------
# Global config flags
# ----------------------------------------

# Base + index + sitemap
NOKIA_BASE_URL = os.getenv("NOKIA_BASE_URL", "https://www.nokia.com")
NOKIA_PSA_INDEX_URL = os.getenv(
    "NOKIA_PSA_INDEX_URL",
    "https://www.nokia.com/about-us/security-and-privacy/product-security-advisory/",
)
NOKIA_SITEMAP_URL = os.getenv(
    "NOKIA_SITEMAP_URL",
    "https://www.nokia.com/sitemap.xml",
)

# HTTP config
TIMEOUT = int(os.getenv("NOKIA_HTTP_TIMEOUT", "30"))
MAX_RETRIES = int(os.getenv("NOKIA_MAX_RETRIES", "3"))
BACKOFF = float(os.getenv("NOKIA_BACKOFF", "1.7"))

HTML_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": NOKIA_BASE_URL,
    "Connection": "keep-alive",
}

# Optional: ScrapingAnt proxy (helps with 403 from datacenter IPs)
SCRAPINGANT_API_KEY = os.getenv("SCRAPINGANT_API_KEY", "")

# Use sitemap in addition to index page
USE_SITEMAP = os.getenv("NOKIA_USE_SITEMAP", "1").lower() in ("1", "true", "yes")

# Verbose logging toggle
VERBOSE: bool = os.getenv("NOKIA_VERBOSE", "0").lower() in ("1", "true", "yes")

# Regexes
CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I)
CVSS_VECTOR_RE = re.compile(r"\bCVSS:[0-9.]+/[A-Z:0-9/.-]+\b", re.I)
CVSS_SCORE_RE = re.compile(
    r"\b(?:CVSS(?:\s*base)?\s*score|base\s*score|CVSS)\s*[: ]\s*([0-9]+(?:\.[0-9])?)\b",
    re.I,
)

# ================== TEXT CLEANUP / DEDUP ==================
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
HEADING_PREFIX_RE = re.compile(
    r"^(?:description|summary|overview|mitigation(?:\s+plan)?|resolution|fix)\s*[:\-]?\s+",
    re.I,
)


def _norm_ws(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())


def _strip_heading_prefix(t: str) -> str:
    return HEADING_PREFIX_RE.sub("", t or "").strip()


def dedup_description(text: str) -> str:
    """
    Normalize and deduplicate paragraphs + sentences so we don't repeat content.
    """
    if not text:
        return text
    text = _strip_heading_prefix(text)
    paras_raw = re.split(r"(?:\n{2,}|(?:\s{2,}))", text)
    seen_paras, paras_norm = set(), []
    for p in paras_raw:
        p_norm = _norm_ws(_strip_heading_prefix(p))
        if p_norm and p_norm not in seen_paras:
            paras_norm.append(p_norm)
            seen_paras.add(p_norm)

    joined = " ".join(paras_norm)
    parts = [s for s in _SENT_SPLIT.split(joined) if s.strip()]
    seen_sents, dedup_sents = set(), []
    for s in parts:
        s_cmp = _norm_ws(_strip_heading_prefix(s))
        if s_cmp and s_cmp not in seen_sents:
            dedup_sents.append(s_cmp)
            seen_sents.add(s_cmp)
    return " ".join(dedup_sents)


# ----------------------------------------
# HTTP helper
# ----------------------------------------
def get(url: str, session: Optional[requests.Session] = None) -> requests.Response:
    """
    Thin wrapper around requests.get with headers + timeout + retries.

    - If SCRAPINGANT_API_KEY is set → route through ScrapingAnt
    - Else → direct requests with retries.
    Raises RuntimeError on repeated failure.
    """
    # ---- ScrapingAnt mode ----
    if SCRAPINGANT_API_KEY:
        api_url = "https://api.scrapingant.com/v2/general"
        params = {
            "url": url,
            "x-api-key": SCRAPINGANT_API_KEY,
            "timeout": str(TIMEOUT),
        }
        last = None
        for i in range(MAX_RETRIES):
            try:
                r = requests.get(
                    api_url,
                    params=params,
                    timeout=TIMEOUT,
                    allow_redirects=True,
                )
                if r.status_code == 200:
                    return r
                if VERBOSE:
                    logger.warning(
                        "[NOKIA] (ScrapingAnt) %s -> HTTP %s, retrying...",
                        url,
                        r.status_code,
                    )
                time_sleep = BACKOFF**i
                if time_sleep > 0:
                    import time
                    time.sleep(time_sleep)
                last = r
            except requests.RequestException as e:
                last = e
                if VERBOSE:
                    logger.warning(
                        "[NOKIA] (ScrapingAnt) %s -> %s, retrying...", url, e
                    )
                time_sleep = BACKOFF**i
                if time_sleep > 0:
                    import time
                    time.sleep(time_sleep)
        raise RuntimeError(f"GET via ScrapingAnt failed for {url}: {last!r}")

    # ---- Direct mode ----
    sess = session or requests.Session()
    last = None
    for i in range(MAX_RETRIES):
        try:
            resp = sess.get(
                url,
                headers=HTML_HEADERS,
                timeout=TIMEOUT,
                allow_redirects=True,
            )
            if resp.status_code == 200:
                return resp
            if VERBOSE:
                logger.warning(
                    "[NOKIA] %s -> HTTP %s, retrying...", url, resp.status_code
                )
            time_sleep = BACKOFF**i
            if time_sleep > 0:
                import time
                time.sleep(time_sleep)
            last = resp
        except requests.RequestException as e:
            last = e
            if VERBOSE:
                logger.warning("[NOKIA] %s -> %s, retrying...", url, e)
            time_sleep = BACKOFF**i
            if time_sleep > 0:
                import time
                time.sleep(time_sleep)
    raise RuntimeError(f"GET failed for {url}: {last!r}")


# ----------------------------------------
# HTML discovery helpers
# ----------------------------------------
def extract_links_from_index(
    html: str, base: str = NOKIA_BASE_URL
) -> tuple[List[str], List[str]]:
    """
    From an index page, extract:
      - advisory detail URLs
      - pagination URLs
    """
    soup = BeautifulSoup(html, "lxml")
    links, pages = set(), set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = urljoin(base, href)
        if (
            "/product-security-advisory/" in full
            and full.rstrip("/") != NOKIA_PSA_INDEX_URL.rstrip("/")
        ):
            links.add(full)

    # "next" rel
    for a in soup.select("a[rel='next']"):
        pages.add(urljoin(base, a.get("href", "")))

    # generic pagination patterns
    for a in soup.find_all("a", href=True):
        h = a["href"]
        if any(k in h for k in ("?page=", "/page/")):
            pages.add(urljoin(base, h))

    return sorted(links), sorted(pages)


def harvest_from_sitemap(session: Optional[requests.Session]) -> List[str]:
    """
    Optionally pull advisory URLs from the Nokia sitemap.
    """
    urls: set[str] = set()
    if not USE_SITEMAP:
        return []
    try:
        resp = get(NOKIA_SITEMAP_URL, session=session)
        soup = BeautifulSoup(resp.text, "xml")
        for loc in soup.find_all("loc"):
            u = loc.get_text(strip=True)
            if "/product-security-advisory/" in u:
                urls.add(u)
        logger.info("[NOKIA] Sitemap yielded %d advisory URLs.", len(urls))
    except Exception as e:
        logger.warning("[NOKIA] sitemap harvest failed: %s", e)
    return sorted(urls)


def crawl_all_advisory_links(
    session: Optional[requests.Session] = None,
) -> List[str]:
    """
    Discover Nokia PSA advisory URLs by crawling the index page + pagination.
    """
    sess = session or requests.Session()
    from collections import deque

    q = deque([NOKIA_PSA_INDEX_URL])
    visited, advisory_urls = set(), set()

    while q:
        url = q.popleft()
        if url in visited:
            continue
        visited.add(url)

        try:
            resp = get(url, session=sess)
        except Exception as e:
            if VERBOSE:
                logger.warning("[NOKIA] index fetch failed %s: %s", url, e)
            continue

        links, pages = extract_links_from_index(resp.text, base=NOKIA_BASE_URL)
        advisory_urls.update(links)
        for pg in pages:
            if pg not in visited:
                q.append(pg)

    logger.info("[NOKIA] Index crawl discovered %d advisory URLs.", len(advisory_urls))
    return sorted(advisory_urls)


# ----------------------------------------
# Parsing helpers
# ----------------------------------------
def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower().rstrip(" :")


def parse_label_value_table(soup: BeautifulSoup) -> Dict[str, str]:
    """
    Parse generic label→value tables (or <dl>) into a dict.
    """
    kv: Dict[str, str] = {}

    # table-based
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if len(cells) == 2:
                k = _norm_key(cells[0].get_text(" ", strip=True))
                v = cells[1].get_text(" ", strip=True)
                if k and v:
                    kv[k] = v

    # dl-based
    for dl in soup.find_all("dl"):
        dts = dl.find_all("dt")
        dds = dl.find_all("dd")
        if len(dts) and len(dds) and len(dts) == len(dds):
            for dt, dd in zip(dts, dds):
                k = _norm_key(dt.get_text(" ", strip=True))
                v = dd.get_text(" ", strip=True)
                if k and v:
                    kv[k] = v

    return kv


def _collect_text_blocks(scope: Tag) -> Optional[str]:
    chunks = []
    for el in scope.find_all(["p", "li"], recursive=True):
        t = el.get_text(" ", strip=True)
        if t:
            chunks.append(t)
    dedup = []
    for t in chunks:
        if not dedup or t != dedup[-1]:
            dedup.append(t)
    return " ".join(dedup).strip() or None


def section_text_after_heading(
    soup: BeautifulSoup, heading_text: str, level_tags=("h2", "h3", "h4")
) -> Optional[str]:
    """
    Grab text following a section heading until the next heading of same level.
    """
    h = None
    for tag in soup.find_all(level_tags):
        if heading_text.lower() in tag.get_text(" ", strip=True).lower():
            h = tag
            break
    if not h:
        return None

    chunks, collected_bodies = [], set()
    for el in h.next_elements:
        if isinstance(el, Tag) and el.name in level_tags and el is not h:
            break
        if isinstance(el, Tag):
            if "simple-text__body" in el.get("class", []):
                if el not in collected_bodies:
                    txt = _collect_text_blocks(el)
                    if txt:
                        chunks.append(txt)
                    collected_bodies.add(el)
                continue
            if el.name in ("p", "li"):
                # avoid double-collect from nested simple-text__body
                skip = False
                for parent in el.parents:
                    if parent in collected_bodies:
                        skip = True
                        break
                if skip:
                    continue
                t = el.get_text(" ", strip=True)
                if t:
                    chunks.append(t)
        elif isinstance(el, NavigableString):
            t = str(el).strip()
            if t:
                chunks.append(t)

    text = " ".join(chunks).strip()
    return text or None


def first_simple_text_body(soup: BeautifulSoup) -> Optional[str]:
    body = soup.select_one(".simple-text__body")
    if body:
        return _collect_text_blocks(body)
    return None


def parse_cvss_from_text(text: Optional[str]) -> Tuple[Optional[str], Optional[str]]:
    if not text:
        return None, None
    vec, score = None, None

    m = CVSS_VECTOR_RE.search(text)
    if m:
        vec = m.group(0).strip()

    m2 = CVSS_SCORE_RE.search(text)
    if m2:
        score = m2.group(1).strip()

    if not score:
        m3 = re.search(r"\bCVSS\s*([0-9]+(?:\.[0-9])?)\b", text, flags=re.I)
        if m3:
            score = m3.group(1).strip()

    return vec, score


# -------- Affected Products Parser (list of dicts internally) --------
def parse_affected_products(soup: BeautifulSoup) -> List[Dict[str, List[str]]]:
    def _is_section_heading(tag):
        if not isinstance(tag, Tag):
            return False
        if tag.name not in ("h2", "h3", "h4"):
            return False
        txt = tag.get_text(" ", strip=True).lower()
        return "affected" in txt and "product" in txt

    def _split_versions(raw):
        if not raw:
            return []
        if isinstance(raw, list):
            raw = " ".join([x for x in raw if x])
        parts = re.split(r"(?:,|;|\||\band\b)", raw, flags=re.I)
        return [p.strip() for p in parts if p and p.strip()]

    def _cell_text(cell):
        if not isinstance(cell, Tag):
            return ""
        lis = [li.get_text(" ", strip=True) for li in cell.find_all("li")]
        if lis:
            return lis
        return cell.get_text(" ", strip=True)

    def _parse_table(table):
        headers = []
        thead = table.find("thead")
        if thead:
            hdr_cells = thead.find_all(["th", "td"])
            headers = [c.get_text(" ", strip=True).lower() for c in hdr_cells]
        if not headers:
            first_tr = table.find("tr")
            if first_tr and first_tr.find_all("th"):
                headers = [
                    c.get_text(" ", strip=True).lower()
                    for c in first_tr.find_all(["th", "td"])
                ]

        p_idx = v_idx = None
        if headers:
            for i, htxt in enumerate(headers):
                if re.search(r"\bproduct", htxt):
                    p_idx = i
                if re.search(r"\bversion", htxt):
                    v_idx = i
            if p_idx is None and v_idx is None and len(headers) == 2:
                p_idx, v_idx = 0, 1

        rows_out = []
        last_product = None
        for tr in table.find_all("tr"):
            if tr.find_parent("thead"):
                continue
            if tr.find_all("th") and not thead:
                continue

            cells = tr.find_all(["td", "th"])
            if not cells:
                continue

            if p_idx is None or v_idx is None:
                if len(cells) >= 2:
                    prod_raw = _cell_text(cells[0])
                    vers_raw = _cell_text(cells[1])
                else:
                    prod_raw = _cell_text(cells[0])
                    vers_raw = ""
            else:
                prod_raw = _cell_text(cells[p_idx]) if p_idx < len(cells) else ""
                vers_raw = _cell_text(cells[v_idx]) if v_idx < len(cells) else ""

            prod = (prod_raw if isinstance(prod_raw, str) else " ".join(prod_raw)).strip()
            if not prod and last_product:
                prod = last_product

            versions = _split_versions(vers_raw)
            if prod or versions:
                rows_out.append({"product": prod, "versions": versions})
                if prod:
                    last_product = prod

        dedup, seen = [], set()
        for r in rows_out:
            key = (r["product"], tuple(r["versions"]))
            if key not in seen:
                dedup.append(r)
                seen.add(key)
        return dedup

    def _parse_lists(start_node):
        results = []
        for sib in start_node.next_siblings:
            if isinstance(sib, Tag) and sib.name in {"h2", "h3", "h4"}:
                break
            if isinstance(sib, Tag):
                for lst in sib.find_all(["ul", "ol"], recursive=False):
                    for li in lst.find_all("li", recursive=False):
                        line = li.get_text(" ", strip=True)
                        if not line:
                            continue
                        if ":" in line:
                            prod, vers = line.split(":", 1)
                            results.append(
                                {
                                    "product": prod.strip(),
                                    "versions": _split_versions(vers),
                                }
                            )
                        else:
                            results.append(
                                {"product": line.strip(), "versions": []}
                            )
                if results:
                    return results
        return results

    section = None
    for h in soup.find_all(["h2", "h3", "h4"]):
        if _is_section_heading(h):
            section = h
            break

    if section:
        next_table = section.find_next("table")
        if next_table:
            parsed = _parse_table(next_table)
            if parsed:
                return parsed
        parsed = _parse_lists(section)
        if parsed:
            return parsed

    best = []
    for table in soup.find_all("table"):
        parsed = _parse_table(table)
        if parsed and any(r["product"] or r["versions"] for r in parsed):
            if len(parsed) > len(best):
                best = parsed
    if best:
        return best

    results = []
    for p in soup.find_all("p"):
        txt = p.get_text(" ", strip=True)
        if re.search(r"\bproduct\s*:", txt, flags=re.I):
            prod = re.split(r"\bversions?\s*:", txt, flags=re.I)[0]
            prod = re.sub(r"^.*?\bproduct\s*:\s*", "", prod, flags=re.I).strip()
            vers = ""
            m = re.search(r"\bversions?\s*:\s*(.*)$", txt, flags=re.I)
            if m:
                vers = m.group(1).strip()
            results.append({"product": prod, "versions": _split_versions(vers)})
    return results


def _dedup_adjacent_words(s: str) -> str:
    parts = s.split()
    out = []
    prev = None
    for w in parts:
        if prev is None or w.lower() != prev.lower():
            out.append(w)
            prev = w
    return " ".join(out)


def _combine_product_versions(
    affected_rows: List[Dict[str, List[str]]],
) -> List[str]:
    out = []
    seen = set()
    for item in affected_rows or []:
        prod = (item.get("product") or "").strip()
        vers_list = item.get("versions") or []
        if vers_list:
            for v in vers_list:
                v = (v or "").strip()
                if prod and v.lower().startswith(prod.lower()):
                    s = v
                else:
                    s = f"{prod} {v}".strip()
                s = _dedup_adjacent_words(s)
                if s and s.lower() not in seen:
                    out.append(s)
                    seen.add(s.lower())
        else:
            s = _dedup_adjacent_words(prod)
            if s and s.lower() not in seen:
                out.append(s)
                seen.add(s.lower())
    return out


# ----------------------------------------
# Nokia advisory parsing (single page)
# ----------------------------------------
def extract_one_advisory(html: str) -> Dict:
    """
    Parse a single Nokia Product Security Advisory page into an intermediate dict.

    We only keep fields we actually need, which will then be merged into a
    single "description" string in advisory_dict_to_document().
    """
    soup = BeautifulSoup(html, "lxml")

    # ---- Title ----
    title = None
    for sel in ["h1", "meta[property='og:title']", "meta[name='twitter:title']", "title"]:
        el = soup.select_one(sel)
        if el:
            val = (el.get("content") if el.has_attr("content") else el.get_text()).strip()
            if val:
                title = val
                break

    # ---- Table key/values (for vuln type, CVSS, etc.) ----
    kv = parse_label_value_table(soup)

    def get_k(*labels):
        lbls = [l.lower() for l in labels]
        for lbl in lbls:
            if lbl in kv:
                return kv[lbl]
            for k2, v2 in kv.items():
                if lbl in k2:
                    return v2
        return None

    vulnerability_type = get_k("vulnerability type", "type", "vulnerability")
    cvss_vector = get_k("cvss vector", "cvss")
    cvss_score = (get_k("cvss score") or "").strip() or None

    # Fallback parse CVSS from full text if needed
    if not (cvss_vector and cvss_score):
        page_text = soup.get_text(" ", strip=True)
        vec_f, score_f = parse_cvss_from_text(page_text)
        cvss_vector = cvss_vector or vec_f
        cvss_score = cvss_score or score_f

    # Description
    description = section_text_after_heading(soup, "Description") or first_simple_text_body(
        soup
    )
    if not description:
        meta = soup.select_one("meta[name='description']")
        cand = meta.get("content").strip() if meta and meta.get("content") else None
        if cand and "security vulnerability advisories" not in cand.lower():
            description = cand
    if not description:
        p = soup.find("p")
        if p:
            description = p.get_text(" ", strip=True)
    if description:
        description = dedup_description(description)

    # Affected products and versions
    affected_rows = parse_affected_products(soup)
    affected_products_and_versions = _combine_product_versions(affected_rows)

    # Mitigation plan
    mitigation_plan = (
        section_text_after_heading(soup, "Mitigation plan")
        or section_text_after_heading(soup, "Mitigation")
        or section_text_after_heading(soup, "Resolution")
        or section_text_after_heading(soup, "Fix")
    )
    if mitigation_plan:
        mitigation_plan = dedup_description(mitigation_plan)

    # CVEs
    page_text_all = soup.get_text(" ", strip=True)
    cves = sorted(set(CVE_RE.findall(page_text_all))) or None

    return {
        "title": title,
        "vulnerability_type": vulnerability_type,
        "cvss_vector": cvss_vector,
        "cvss_score": cvss_score,
        "description": description,
        "affected_products_and_versions": affected_products_and_versions,
        "cves": cves,
        "mitigation_plan": mitigation_plan,
    }


# ----------------------------------------
# Merge intermediate dict → ingestable document
# ----------------------------------------
def advisory_dict_to_document(url: str, adv: Dict) -> Dict[str, str]:
    """
    Convert a parsed Nokia advisory dict (from extract_one_advisory)
    into a compact document with:
      - url
      - title
      - description (merged description with vuln details)
    """
    title = adv.get("title") or "Nokia Product Security Advisory"

    parts: List[str] = []

    # Vulnerability type
    if adv.get("vulnerability_type"):
        parts.append(f"Vulnerability type: {adv['vulnerability_type']}")

    # CVSS (vector + score)
    cvss_bits = []
    if adv.get("cvss_score"):
        cvss_bits.append(f"score {adv['cvss_score']}")
    if adv.get("cvss_vector"):
        cvss_bits.append(f"vector {adv['cvss_vector']}")
    if cvss_bits:
        parts.append("CVSS " + ", ".join(cvss_bits))

    # Affected products and versions
    if adv.get("affected_products_and_versions"):
        affected_str = "; ".join(adv["affected_products_and_versions"])
        parts.append("Affected products and versions: " + affected_str)

    # CVEs
    if adv.get("cves"):
        parts.append("Related CVEs: " + ", ".join(adv["cves"]))

    # Mitigation
    if adv.get("mitigation_plan"):
        parts.append("Mitigation plan: " + adv["mitigation_plan"])

    # Original description text
    if adv.get("description"):
        parts.append("Technical description: " + adv["description"])

    merged_description = "\n".join(parts).strip()

    return {
        "url": url,
        "title": title,
        "description": merged_description or title,
    }


# ----------------------------------------
# URL discovery wrapper (NO early Qdrant checking)
# ----------------------------------------
def fetch_nokia_advisory_urls(
    session: Optional[requests.Session] = None,
) -> List[str]:
    """
    Discover all Nokia Product Security Advisory URLs.
    Uses HTML index crawl + (optionally) the global sitemap.
    
    Returns raw URLs without any Qdrant deduplication at this stage.
    Deduplication happens later in scrape_nokia() following Huawei/Cisco pattern.
    """
    sess = session or requests.Session()
    raw_urls: set[str] = set()

    if USE_SITEMAP:
        raw_urls.update(harvest_from_sitemap(sess))

    raw_urls.update(crawl_all_advisory_links(sess))

    logger.info("[NOKIA] Total advisory URLs discovered: %d", len(raw_urls))
    return sorted(raw_urls)


# ----------------------------------------
# Public entrypoint (Huawei-style Qdrant checking)
# ----------------------------------------
def scrape_nokia(check_qdrant: bool = True) -> List[Dict[str, str]]:
    """
    Main scraping entrypoint for Nokia following the Huawei/Cisco scraper pattern:

    1. Discover all advisory URLs
    2. For each *distinct* URL:
       - Count as 'seen'
       - If check_qdrant=True:
           * Call url_already_ingested(url)
           * If True → increment 'skipped' and continue
           * If the check fails → log error, but scrape anyway
       - If not in Qdrant → fetch + parse → 'new' doc
    3. Return list of new documents: {url, title, description}
    """
    logger.info("[NOKIA] Starting Nokia advisory scrape (check_qdrant=%s)", check_qdrant)

    session = requests.Session()
    urls = fetch_nokia_advisory_urls(session=session)

    docs: List[Dict[str, str]] = []
    seen_urls: set[str] = set()

    n_seen = 0
    n_skipped = 0
    n_new = 0

    logger.info("[NOKIA] Total URLs discovered: %d", len(urls))

    for url in urls:
        # Skip duplicates in the discovered URL list itself
        if url in seen_urls:
            continue
        seen_urls.add(url)

        # Count distinct advisory URL as "seen"
        n_seen += 1

        # Huawei-style Qdrant check: skip BEFORE expensive fetch/parse
        if check_qdrant and url:
            try:
                if url_already_ingested(url):
                    n_skipped += 1
                    logger.info("[NOKIA] Skipping already-ingested URL: %s", url)
                    continue
            except Exception as e:
                logger.error("[NOKIA] Qdrant check failed for %s: %s", url, e)
                # conservative: fall through and scrape anyway

        # If we reach here, this URL is considered "new" for this run
        n_new += 1

        try:
            resp = get(url, session=session)
            adv = extract_one_advisory(resp.text)
            doc = advisory_dict_to_document(url, adv)
            docs.append(doc)
        except Exception as e:
            logger.warning("[NOKIA] Error scraping %s: %s", url, e)
            continue

    logger.info(
        "[NOKIA] Seen=%d, skipped=%d (already in Qdrant), new=%d",
        n_seen,
        n_skipped,
        n_new,
    )
    logger.info("[NOKIA] Scraped %d new advisories (documents).", len(docs))
    return docs
