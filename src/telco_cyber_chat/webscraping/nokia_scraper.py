from __future__ import annotations

import os
import re
import time
import json
import logging
from collections import deque
from urllib.parse import urljoin
from typing import List, Dict, Optional, Tuple, Any, Set

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue

logger = logging.getLogger(__name__)

# ================== CONFIG ==================
BASE       = "https://www.nokia.com"
INDEX_URL  = "https://www.nokia.com/about-us/security-and-privacy/product-security-advisory/"
SITEMAP    = "https://www.nokia.com/sitemap.xml"

USE_SITEMAP       = True
REPARSE_EXISTING  = True   # If True: still discover all URLs; Qdrant decides skip/new
VERBOSE           = False

# ✅ Qdrant filter keys must match your payload keys in Qdrant
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "Telco_CyberChat")
VENDOR_KEY = "vendor"
URL_KEY    = "url"

# ✅ MUST match exactly the vendor value you store in Qdrant
VENDOR_VALUE = os.getenv("NOKIA_VENDOR_VALUE", "Nokia")

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/126.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": BASE,
    "Connection": "keep-alive",
}

# Regexes
CVE_RE          = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I)
CVSS_VECTOR_RE  = re.compile(r"\bCVSS:[0-9.]+/[A-Z:0-9/.-]+\b", re.I)
CVSS_SCORE_RE   = re.compile(r"\b(?:CVSS(?:\s*base)?\s*score|base\s*score|CVSS)\s*[: ]\s*([0-9]+(?:\.[0-9])?)\b", re.I)

# ================== TEXT CLEANUP / DEDUP ==================
_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")
HEADING_PREFIX_RE = re.compile(
    r"^(?:description|summary|overview|mitigation(?:\s+plan)?|resolution|fix)\s*[:\-]?\s+",
    re.I
)

def _norm_ws(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())

def _strip_heading_prefix(t: str) -> str:
    return HEADING_PREFIX_RE.sub("", t or "").strip()

def dedup_description(text: str) -> str:
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

# ================== QDRANT CONFIG ==================
def get_qdrant_config() -> Tuple[str, str]:
    url = None
    api_key = None
    try:
        from google.colab import userdata
        url = userdata.get("QDRANT_URL")
        api_key = userdata.get("QDRANT_API_KEY")
    except Exception:
        pass

    url = url or os.getenv("QDRANT_URL")
    api_key = api_key or os.getenv("QDRANT_API_KEY")

    if not url:
        raise RuntimeError("QDRANT_URL not found (Colab secrets or env).")
    if not api_key:
        raise RuntimeError("QDRANT_API_KEY not found (Colab secrets or env).")
    return url, api_key

def make_qdrant_client() -> QdrantClient:
    qurl, qkey = get_qdrant_config()
    return QdrantClient(url=qurl, api_key=qkey)

def url_already_ingested_vendor_url(
    client: QdrantClient,
    collection_name: str,
    vendor_value: str,
    url: str,
) -> bool:
    """
    True if a point exists with payload:
      vendor == vendor_value AND url == url
    (vendor + url must be KEYWORD indexes in Qdrant for best performance)
    """
    try:
        points, _ = client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key=VENDOR_KEY, match=MatchValue(value=vendor_value)),
                    FieldCondition(key=URL_KEY, match=MatchValue(value=url)),
                ]
            ),
            limit=1,
            with_payload=False,
            with_vectors=False,
        )
        return len(points) > 0
    except Exception as e:
        logger.error("[NOKIA] Qdrant check failed for url=%s: %s", url, e)
        return False

# ================== HTTP HELPERS ==================
def get(url, session=None, retries=3, backoff=1.7, timeout=30):
    s = session or requests.Session()
    last = None
    for i in range(retries):
        try:
            r = s.get(url, headers=HEADERS, timeout=timeout, allow_redirects=True)
            if r.status_code == 200:
                return r
            if VERBOSE:
                print(f"[warn] {url} -> HTTP {r.status_code}, retrying...")
            time.sleep(backoff ** i)
            last = r
        except requests.RequestException as e:
            last = e
            if VERBOSE:
                print(f"[warn] {url} -> {e}, retrying...")
            time.sleep(backoff ** i)
    raise RuntimeError(f"GET failed for {url}: {getattr(last,'status_code',last)}")

# ================== DISCOVERY ==================
def extract_links_from_index(html, base=BASE):
    soup = BeautifulSoup(html, "lxml")
    links, pages = set(), set()

    for a in soup.find_all("a", href=True):
        href = a["href"].strip()
        full = urljoin(base, href)
        if "/product-security-advisory/" in full and full.rstrip("/") != INDEX_URL.rstrip("/"):
            links.add(full)

    for a in soup.select("a[rel='next']"):
        pages.add(urljoin(base, a.get("href", "")))
    for a in soup.find_all("a", href=True):
        h = a["href"]
        if any(k in h for k in ("?page=", "/page/")):
            pages.add(urljoin(base, h))

    return sorted(links), sorted(pages)

def harvest_from_sitemap(session):
    urls = set()
    try:
        r = get(SITEMAP, session=session)
        soup = BeautifulSoup(r.text, "xml")
        for loc in soup.find_all("loc"):
            u = loc.get_text(strip=True)
            if "/product-security-advisory/" in u:
                urls.add(u)
    except Exception as e:
        if VERBOSE:
            print(f"[warn] sitemap harvest failed: {e}")
    return sorted(urls)

def crawl_all_advisory_links(session):
    q = deque([INDEX_URL])
    visited, advisory_urls = set(), set()
    while q:
        url = q.popleft()
        if url in visited:
            continue
        visited.add(url)
        try:
            r = get(url, session=session)
        except Exception as e:
            if VERBOSE:
                print(f"[warn] index fetch failed {url}: {e}")
            continue
        links, pages = extract_links_from_index(r.text, base=BASE)
        advisory_urls.update(links)
        for pg in pages:
            if pg not in visited:
                q.append(pg)
    return sorted(advisory_urls)

# ================== NOKIA-SPECIFIC PARSERS ==================
def _canonical_url(soup: BeautifulSoup, fallback: Optional[str] = None) -> Optional[str]:
    link = soup.select_one("link[rel='canonical']")
    if link and link.get("href"):
        return link["href"].strip()
    og = soup.select_one("meta[property='og:url']")
    if og and og.get("content"):
        return og["content"].strip()
    tw = soup.select_one("meta[name='twitter:url']")
    if tw and tw.get("content"):
        return tw["content"].strip()
    return fallback

def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip()).lower().rstrip(" :")

def parse_label_value_table(soup):
    kv = {}
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if len(cells) == 2:
                k = _norm_key(cells[0].get_text(" ", strip=True))
                v = cells[1].get_text(" ", strip=True)
                if k and v:
                    kv[k] = v
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

def _collect_text_blocks(scope: Tag):
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

def section_text_after_heading(soup, heading_text, level_tags=("h2","h3","h4")):
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

def first_simple_text_body(soup):
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

# (your existing parse_affected_products / combine helpers / acknowledgements / references)
# ✅ keep exactly as-is (omitted here for brevity in this message)
# --- IMPORTANT ---
# Paste your existing implementations below without changes:
#   parse_affected_products()
#   _dedup_adjacent_words()
#   _combine_product_versions()
#   parse_acknowledgements()
#   parse_references()

# ================== PER-PAGE EXTRACTION ==================
def extract_one_advisory(html: str, page_url: Optional[str] = None) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")

    # Title
    title = None
    for sel in ["h1", "meta[property='og:title']", "meta[name='twitter:title']", "title"]:
        el = soup.select_one(sel)
        if el:
            val = (el.get("content") if el.has_attr("content") else el.get_text()).strip()
            if val:
                title = val
                break

    canonical = _canonical_url(soup, fallback=page_url)
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
    cvss_vector        = get_k("cvss vector", "cvss")
    cvss_score         = (get_k("cvss score") or "").strip() or None

    if not (cvss_vector and cvss_score):
        page_text = soup.get_text(" ", strip=True)
        vec_f, score_f = parse_cvss_from_text(page_text)
        cvss_vector = cvss_vector or vec_f
        cvss_score  = cvss_score  or score_f

    description = section_text_after_heading(soup, "Description") or first_simple_text_body(soup)
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

    # --- keep your existing parsing calls here (same as your file) ---
    affected_rows = parse_affected_products(soup)  # <- from your existing code
    affected_products_and_versions = _combine_product_versions(affected_rows)

    mitigation_plan = (
        section_text_after_heading(soup, "Mitigation plan")
        or section_text_after_heading(soup, "Mitigation")
        or section_text_after_heading(soup, "Resolution")
        or section_text_after_heading(soup, "Fix")
    )
    if mitigation_plan:
        mitigation_plan = dedup_description(mitigation_plan)

    page_text_all = soup.get_text(" ", strip=True)
    cves = sorted(set(CVE_RE.findall(page_text_all))) or None

    acknowledgements = parse_acknowledgements(soup)
    references       = parse_references(soup)

    return {
        "vendor": VENDOR_VALUE,
        "title": title,
        "url": canonical,
        "vulnerability_type": vulnerability_type,
        "cvss_vector": cvss_vector,
        "cvss_score": cvss_score,
        "description": description,
        "affected_products_and_versions": affected_products_and_versions,
        "cves": cves,
        "mitigation_plan": mitigation_plan,
        "acknowledgements": acknowledgements,
        "references": references,
    }

# ================== PUBLIC ENTRYPOINT (used by your scraper assistant) ==================
def scrape_nokia(check_qdrant: bool = True) -> List[Dict[str, Any]]:
    """
    - Discover Nokia PSA URLs (sitemap + crawl)
    - ✅ BEFORE fetching/parsing each advisory page: check Qdrant existence using (vendor, url)
      - If exists -> skip
      - If not -> fetch + parse + return doc
    """
    session = requests.Session()

    qclient: Optional[QdrantClient] = None
    cache_existing: Set[str] = set()  # in-run cache to avoid duplicate Qdrant calls

    if check_qdrant:
        qclient = make_qdrant_client()
        logger.info("[NOKIA] Qdrant dedupe enabled (vendor=%s, collection=%s)", VENDOR_VALUE, QDRANT_COLLECTION)

    urls = set()
    if USE_SITEMAP:
        urls.update(harvest_from_sitemap(session))
    urls.update(crawl_all_advisory_links(session))
    urls = sorted(urls)

    docs: List[Dict[str, Any]] = []
    skipped = 0
    new_count = 0

    for u in urls:
        # ✅ Check Qdrant first using discovered URL (you said it's already canonical)
        if check_qdrant and qclient is not None:
            if u in cache_existing or url_already_ingested_vendor_url(qclient, QDRANT_COLLECTION, VENDOR_VALUE, u):
                cache_existing.add(u)
                skipped += 1
                continue

        try:
            resp = get(u, session=session)
            doc = extract_one_advisory(resp.text, page_url=u)

            canon = (doc.get("url") or u).strip() if isinstance(doc.get("url"), str) else u

            # ✅ Safety: if canonical differs, re-check canonical in Qdrant before keeping it
            if check_qdrant and qclient is not None and canon != u:
                if canon in cache_existing or url_already_ingested_vendor_url(qclient, QDRANT_COLLECTION, VENDOR_VALUE, canon):
                    cache_existing.add(u)
                    cache_existing.add(canon)
                    skipped += 1
                    continue

            cache_existing.add(u)
            cache_existing.add(canon)

            docs.append(doc)
            new_count += 1
            time.sleep(0.2)

        except Exception as e:
            if VERBOSE:
                print(f"[error] {u}: {e}")
            logger.warning("[NOKIA] Failed on %s: %s", u, e)

    logger.info("[NOKIA] New=%d | Skipped(existing in Qdrant)=%d | Total discovered=%d", new_count, skipped, len(urls))
    return docs

# ================== CLI debug (optional) ==================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docs = scrape_nokia(check_qdrant=True)
    print(f"New Nokia docs: {len(docs)}")
    if docs:
        print(json.dumps(docs[0], indent=2, ensure_ascii=False))
