from __future__ import annotations

import os
import re
import time
import json
import logging
import hashlib
from collections import deque
from urllib.parse import urljoin, urlparse, urlunparse
from typing import List, Dict, Optional, Tuple, Any, Set
from datetime import datetime, timezone

import requests
from bs4 import BeautifulSoup, NavigableString, Tag
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from llama_index.core.schema import TextNode

logger = logging.getLogger("telco_cyber_chat.webscraping.nokia")

# Force Nokia logs to STDOUT (not STDERR) so they don't appear as [ERROR] in server logs
logger.handlers.clear()
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [NOKIA] %(message)s"))
logger.addHandler(handler)

logger.setLevel(logging.INFO)
logger.propagate = False

# ================== CONFIG ==================
BASE       = "https://www.nokia.com"
INDEX_URL  = "https://www.nokia.com/about-us/security-and-privacy/product-security-advisory/"
SITEMAP    = "https://www.nokia.com/sitemap.xml"

USE_SITEMAP       = True
REPARSE_EXISTING  = True   # still discover URLs; Qdrant decides skip/new
VERBOSE           = False

QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "Telco_CyberChat")

# These must match payload keys in Qdrant (your stored points)
VENDOR_KEY = "vendor"
URL_KEY    = "url"

# MUST match what you store in Qdrant payload vendor value for Nokia
VENDOR_VALUE = (os.getenv("NOKIA_VENDOR_VALUE", "Nokia") or "Nokia").strip()

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
CVSS_SCORE_RE   = re.compile(
    r"\b(?:CVSS(?:\s*base)?\s*score|base\s*score|CVSS)\s*[: ]\s*([0-9]+(?:\.[0-9])?)\b",
    re.I
)

# ================== URL NORMALIZATION ==================
def canonicalize_url(u: str) -> str:
    """Strip query+fragment (keep path), to align with canonical storage/dedup."""
    if not u:
        return ""
    u = u.strip()
    p = urlparse(u)
    if not p.scheme:
        return ""
    return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))

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

def fetch_existing_vendor_urls(
    client: QdrantClient,
    collection_name: str,
    vendor_value: str,
    page_size: int = 256,
) -> Set[str]:
    """Scroll Qdrant with filter vendor=<vendor_value> and collect payload[url]."""
    existing: Set[str] = set()
    offset = None

    vendor_filter = Filter(
        must=[FieldCondition(key=VENDOR_KEY, match=MatchValue(value=vendor_value))]
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
            u = payload.get(URL_KEY)
            if isinstance(u, str) and u.strip():
                existing.add(canonicalize_url(u.strip()) or u.strip())
        if offset is None:
            break

    return existing

def url_already_ingested_vendor_url(
    client: QdrantClient,
    collection_name: str,
    vendor_value: str,
    url: str,
) -> bool:
    """Fallback per-url check (used only if preload fails or for canonical re-check)."""
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
        logger.error("Qdrant check failed for url=%s: %s", url, e)
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
            links.add(canonicalize_url(full) or full)

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
                urls.add(canonicalize_url(u) or u)
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

def section_text_after_heading(soup, heading_text, level_tags=("h2", "h3", "h4")):
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

# -------- Minimal implementations so the file is runnable --------
def parse_affected_products(soup: BeautifulSoup) -> List[Dict[str, Any]]:
    """Best-effort parse of an affected products table/list."""
    for heading in ["Affected products", "Affected product", "Products affected"]:
        sec = section_text_after_heading(soup, heading)
        if sec:
            return [{"affected_product": sec}]

    for table in soup.find_all("table"):
        hdr = " ".join([c.get_text(" ", strip=True).lower() for c in table.find_all(["th"])])
        if ("product" in hdr and "version" in hdr) or ("affected" in hdr and "product" in hdr):
            rows = []
            trs = table.find_all("tr")
            if len(trs) < 2:
                continue
            headers = [c.get_text(" ", strip=True).lower() for c in trs[0].find_all(["th", "td"])]
            for tr in trs[1:]:
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                if not cells:
                    continue
                row = {}
                for i, val in enumerate(cells):
                    key = headers[i] if i < len(headers) else f"col{i+1}"
                    row[re.sub(r"[^a-z0-9_]+", "_", key).strip("_") or f"col{i+1}"] = val
                if any(v.strip() for v in row.values() if isinstance(v, str)):
                    rows.append(row)
            if rows:
                return rows
    return []

def _combine_product_versions(rows: List[Dict[str, Any]]) -> Any:
    return rows or None

def parse_acknowledgements(soup: BeautifulSoup) -> Optional[str]:
    txt = (
        section_text_after_heading(soup, "Acknowledgements")
        or section_text_after_heading(soup, "Acknowledgments")
    )
    return dedup_description(txt) if txt else None

def parse_references(soup: BeautifulSoup) -> Optional[List[str]]:
    refs: List[str] = []
    h = None
    for tag in soup.find_all(["h2", "h3", "h4"]):
        if "references" in tag.get_text(" ", strip=True).lower():
            h = tag
            break
    if h:
        for el in h.find_all_next(["a"], href=True):
            parent_heading = el.find_parent(["h2", "h3", "h4"])
            if parent_heading and parent_heading is not h:
                break
            href = el.get("href", "").strip()
            if href:
                refs.append(urljoin(BASE, href))
    refs = list(dict.fromkeys(refs))
    return refs or None
# ----------------------------------------------------------------

# ================== PER-PAGE EXTRACTION ==================
def extract_one_advisory(html: str, page_url: Optional[str] = None) -> Dict[str, Any]:
    soup = BeautifulSoup(html, "lxml")

    title = None
    for sel in ["h1", "meta[property='og:title']", "meta[name='twitter:title']", "title"]:
        el = soup.select_one(sel)
        if el:
            val = (el.get("content") if el.has_attr("content") else el.get_text()).strip()
            if val:
                title = val
                break

    canonical = _canonical_url(soup, fallback=page_url)
    canonical = canonicalize_url(canonical) if canonical else canonicalize_url(page_url or "")
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
        cvss_score  = cvss_score or score_f

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

    affected_rows = parse_affected_products(soup)
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
        "scraped_date": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
    }

# ================== TextNode builders (RULE: text=all fields except url, metadata=only url) ==================
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
        if not ks:
            continue
        if ks.lower() == "url":
            continue  # rule: url NOT in text

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
    url = canonicalize_url(url.strip()) or url.strip()

    rec2 = dict(rec)
    rec2["vendor"] = rec2.get("vendor") or VENDOR_VALUE

    text = record_to_text(rec2)
    if not text:
        return None

    return TextNode(
        id_=_stable_id(str(rec2["vendor"]), url),
        text=text,
        metadata={"url": url},  # ONLY url (as you requested)
    )

# ================== PUBLIC ENTRYPOINT (ONE NAME ONLY) ==================
def scrape_nokia(check_qdrant: bool = True) -> List[TextNode]:
    """
    Returns:
      - List[TextNode] (NEW only)
    """
    session = requests.Session()

    qclient: Optional[QdrantClient] = None
    existing_urls: Set[str] = set()

    if check_qdrant:
        qclient = make_qdrant_client()
        try:
            existing_urls = fetch_existing_vendor_urls(qclient, QDRANT_COLLECTION, VENDOR_VALUE)
            logger.info("Loaded %d existing URLs from Qdrant for vendor=%s", len(existing_urls), VENDOR_VALUE)
        except Exception as e:
            logger.warning("Failed to preload existing URLs (%s). Will fallback to per-url checks.", e)
            existing_urls = set()

    urls = set()
    if USE_SITEMAP:
        urls.update(harvest_from_sitemap(session))
    urls.update(crawl_all_advisory_links(session))
    urls = sorted(urls)

    nodes: List[TextNode] = []
    skipped = 0

    for u in urls:
        u = canonicalize_url(u) or u

        # Qdrant check first
        if check_qdrant and qclient is not None:
            if (existing_urls and u in existing_urls) or (
                not existing_urls and url_already_ingested_vendor_url(qclient, QDRANT_COLLECTION, VENDOR_VALUE, u)
            ):
                skipped += 1
                continue

        try:
            resp = get(u, session=session)
            doc = extract_one_advisory(resp.text, page_url=u)

            canon = doc.get("url")
            canon = canonicalize_url(canon.strip()) if isinstance(canon, str) and canon.strip() else u

            # If canonical differs, re-check canonical too
            if check_qdrant and qclient is not None and canon != u:
                if (existing_urls and canon in existing_urls) or (
                    not existing_urls and url_already_ingested_vendor_url(qclient, QDRANT_COLLECTION, VENDOR_VALUE, canon)
                ):
                    skipped += 1
                    continue

            doc["url"] = canon

            node = create_text_node_from_record(doc)
            if node:
                nodes.append(node)

            time.sleep(0.2)

        except Exception as e:
            if VERBOSE:
                print(f"[error] {u}: {e}")
            logger.warning("Failed on %s: %s", u, e)

    logger.info("New nodes=%d | Skipped(existing)=%d | Discovered=%d", len(nodes), skipped, len(urls))
    return nodes

# ================== OPTIONAL DEBUG WRAPPER ==================
def scrape_nokia_debug(check_qdrant: bool = True) -> Dict[str, Any]:
    nodes = scrape_nokia(check_qdrant=check_qdrant)
    return {
        "ok": True,
        "vendor": VENDOR_VALUE,
        "nodes": nodes,
        "per_source": {"nokia": len(nodes)},
        "stats": {"new": len(nodes)},
    }

# ================== CLI debug (optional) ==================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    nodes = scrape_nokia(check_qdrant=True)

    print("✅ Nokia NEW nodes:", len(nodes))
    if nodes:
        n0 = nodes[0]
        print("\n--- NODE PREVIEW ---")
        print("id_:", n0.id_)
        print("metadata:", n0.metadata)  # must be ONLY {"url": "..."}
        print("text (first 800 chars):\n", n0.text[:800])
