# !pip install -q requests beautifulsoup4 lxml qdrant-client llama-index-core

from __future__ import annotations

import json, re, math, time, os, logging, hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional, List, Dict, Tuple, Set
from urllib.parse import urlparse, urlunparse

import requests
from bs4 import BeautifulSoup, Tag

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from llama_index.core.schema import TextNode

# ================== LOGGER ==================
logger = logging.getLogger("telco_cyber_chat.webscraping.huawei")

# Force Huawei logs to STDOUT (not STDERR) so they don't appear as [ERROR] in server logs
logger.handlers.clear()
handler = logging.StreamHandler(stream=sys.stdout)
handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s [Huawei] %(message)s"))
logger.addHandler(handler)

logger.setLevel(logging.INFO)
logger.propagate = False
# ================== CONFIG ==================
# Internal key used for ids / dict keys (keep stable)
VENDOR_KEY = "huawei"

# Display/value stored in record text (and commonly used for payload vendor)
VENDOR_VALUE = (os.getenv("HUAWEI_VENDOR_VALUE", "Huawei") or "Huawei").strip()

# IMPORTANT: dedupe needs to match existing vendor values already stored in Qdrant
# This loads existing URLs for multiple vendor spellings to avoid duplicates while migrating.
# Example: HUAWEI_VENDOR_ALIASES="Huawei,huawei"
_alias_env = (os.getenv("HUAWEI_VENDOR_ALIASES", "") or "").strip()
VENDOR_ALIASES: List[str] = []
if _alias_env:
    VENDOR_ALIASES = [x.strip() for x in _alias_env.split(",") if x.strip()]
# Always include both common forms at minimum:
for v in [VENDOR_VALUE, VENDOR_KEY]:
    if v and v not in VENDOR_ALIASES:
        VENDOR_ALIASES.append(v)

REFERRER  = "https://www.huawei.com/en/psirt/all-bulletins?page=1"
POST_URL  = "https://www.huawei.com/service/portalapplication/v1/corp/psirt"
OUT_FILE  = Path("huawei_psirt_advisories_all.json")

BASE_PAYLOAD: Dict[str, Any] = {
    "contentId": "aadaee27bbac4341a6d2014c788a2c85",
    "catalogPathList": ["/psirt/"],
    "pageNum": "1",
    "pageSize": 20,
    "time": "",
    "filterLabelList": [[]],
}

# Qdrant env
QDRANT_URL = (os.getenv("QDRANT_URL") or "").strip()
QDRANT_API_KEY = (os.getenv("QDRANT_API_KEY") or "").strip()
QDRANT_COLLECTION = (os.getenv("QDRANT_COLLECTION", "Telco_CyberChat") or "Telco_CyberChat").strip()

S = requests.Session()
S.headers.update({
    "User-Agent": "Mozilla/5.0",
    "Accept": "application/json, text/javascript, */*; q=0.01",
    "Content-Type": "application/json",
    "Referer": REFERRER,
    "Origin": "https://www.huawei.com",
    "X-Requested-With": "XMLHttpRequest",
})

# ================== REGEX (robust/i18n) ==================
CVERE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I)
RE_COLON = r"[:：]\s*"

SCORE_RE_LIST = [
    re.compile(rf"\bBase\s*score{RE_COLON}([0-9.]+)", re.I),
    re.compile(rf"\b基准分{RE_COLON}([0-9.]+)"),
]
TEMP_RE_LIST = [
    re.compile(rf"\bTemporal\s*score{RE_COLON}([0-9.]+)", re.I),
    re.compile(rf"\bTemporary\s*score{RE_COLON}([0-9.]+)", re.I),
    re.compile(rf"\b临时分{RE_COLON}([0-9.]+)"),
]
ENV_RE_LIST = [
    re.compile(rf"\bEnvironmental\s*Score{RE_COLON}(NA|N/A|[0-9.]+)", re.I),
    re.compile(rf"\b环境分{RE_COLON}(NA|N/A|[0-9.]+)"),
]
VEC_RE_LIST = [
    re.compile(
        rf"\bCVSS\s*v?3(?:\.\d+)?\s*Vector{RE_COLON}"
        r"([A-Za-z]{1,3}:[^ \t\r\n]+(?:/[A-Za-z]{1,3}:[^ \t\r\n]+)*)",
        re.I,
    ),
    re.compile(
        rf"\bCVSS\s*v?3(?:\.\d+)?\s*向量{RE_COLON}"
        r"([A-Za-z]{1,3}:[^ \t\r\n]+(?:/[A-Za-z]{1,3}:[^ \t\r\n]+)*)"
    ),
]

# ================== URL HELPERS ==================
def absolutize(u: str) -> str:
    if not u:
        return ""
    u = u.strip()
    if u.startswith(("http://", "https://")):
        return u
    if u.startswith("www."):
        return "https://" + u
    if u.startswith("/"):
        return "https://www.huawei.com" + u
    return "https://www.huawei.com/" + u.lstrip("/")

def canonicalize_url(u: str) -> str:
    """Remove query + fragment to match canonical URL storage in Qdrant."""
    u = absolutize(u)
    p = urlparse(u)
    return urlunparse((p.scheme, p.netloc, p.path, "", "", ""))

def classify_item(url: str, title: Optional[str]) -> Optional[str]:
    u = (url or "").lower()
    if "/psirt/security-advisories/" in u:
        return "advisory"
    if "/psirt/security-notices/" in u:
        return "notice"
    if title:
        t = (title or "").lower()
        if "security advisory" in t:
            return "advisory"
        if "security notice" in t:
            return "notice"
    return None

# ================== TEXT HELPERS ==================
def norm_ws(t: str) -> str:
    return re.sub(r"\s+", " ", (t or "").strip())

def split_sentences(text: str) -> List[str]:
    if not text:
        return []
    parts = [p.strip() for p in re.split(r"(?<=[。！？?!\.])\s+", text) if p.strip()]
    return parts or [text.strip()]

def dedup_sentences(text: str) -> str:
    seen, out = set(), []
    for s in split_sentences(norm_ws(text)):
        if s not in seen:
            seen.add(s)
            out.append(s)
    return " ".join(out)

# ================== HTML PARSING ==================
def find_moreinfo_div_for_title(soup: BeautifulSoup, section_title: str) -> Optional[Tag]:
    a = soup.find("a", attrs={"data-expand": section_title}) or soup.find("a", attrs={"data-collapse": section_title})
    if not a:
        return None
    wrapper = a.find_parent(class_="psirt-set-out")
    if wrapper:
        mi = wrapper.find("div", class_="moreinfo")
        if mi:
            return mi
    cur: Optional[Tag] = a
    for _ in range(10):
        cur = cur.find_next_sibling()
        if not cur:
            break
        if isinstance(cur, Tag) and "moreinfo" in (cur.get("class") or []):
            return cur
    return None

def collect_text_from_container(node: Tag) -> str:
    if not isinstance(node, Tag):
        return ""
    parts: List[str] = []
    for el in node.find_all(["p", "li", "div"], recursive=True):
        txt = norm_ws(el.get_text(" ", strip=True))
        if txt:
            parts.append(txt)
    if not parts:
        txt = norm_ws(node.get_text(" ", strip=True))
        if txt:
            parts.append(txt)
    out: List[str] = []
    for t in parts:
        if not out or t != out[-1]:
            out.append(t)
    return " ".join(out).strip()

def extract_description_from_div(soup: BeautifulSoup) -> str:
    node = soup.find("div", class_="summary")
    if isinstance(node, Tag):
        ps = node.find_all("p")
        if ps:
            texts = [norm_ws(p.get_text(" ", strip=True)) for p in ps]
            return " ".join([t for t in texts if t])
        return norm_ws(node.get_text(" ", strip=True))

    mi = find_moreinfo_div_for_title(soup, "Summary") or find_moreinfo_div_for_title(soup, "摘要")
    if mi:
        inner = mi.find("div", class_="summary")
        if inner:
            txt = norm_ws(inner.get_text(" ", strip=True))
            if txt:
                return txt
        txt = collect_text_from_container(mi)
        if txt:
            return txt

    for p in soup.find_all("p"):
        txt = norm_ws(p.get_text(" ", strip=True))
        if txt and "Vulnerabilities are scored based on the CVSS" not in txt:
            return txt
    return ""

def extract_section_text(soup: BeautifulSoup, title: str, cn_variants: Optional[List[str]] = None) -> str:
    mi = find_moreinfo_div_for_title(soup, title)
    if not mi and cn_variants:
        for t in cn_variants:
            mi = find_moreinfo_div_for_title(soup, t)
            if mi:
                break
    return collect_text_from_container(mi) if mi else ""

def parse_table(table: Tag) -> List[Dict[str, str]]:
    rows = table.find_all("tr")
    if not rows:
        return []

    headers_raw = [norm_ws(c.get_text(" ", strip=True)) for c in rows[0].find_all(["th", "td"])]

    def norm_header(h: str) -> str:
        hl = h.lower()
        if "affected product" in hl or "受影响产品" in hl:
            return "affected_product"
        if "affected version" in hl or "受影响版本" in hl:
            return "affected_version"
        if "repair version" in hl or "fixed version" in hl or "修复版本" in hl:
            return "repair_version"
        return re.sub(r"[^a-z0-9_]+", "_", hl).strip("_") or "col"

    headers = [norm_header(h) for h in headers_raw] or ["col1", "col2", "col3"]

    data: List[Dict[str, str]] = []
    current_product: Optional[str] = None

    for tr in rows[1:]:
        cells = tr.find_all(["td", "th"])
        if not cells:
            continue
        values = [norm_ws(c.get_text(" ", strip=True)) for c in cells]
        if len(values) < len(headers):
            values += [""] * (len(headers) - len(values))
        if len(values) > len(headers):
            values = values[: len(headers)]
        row = dict(zip(headers, values))

        if not row.get("affected_product") and current_product:
            row["affected_product"] = current_product
        if row.get("affected_product"):
            current_product = row["affected_product"]

        if any(v for v in row.values()):
            data.append(row)

    return data

def extract_software_versions_and_fixes(soup: BeautifulSoup) -> List[Dict[str, str]]:
    mi = find_moreinfo_div_for_title(soup, "Software Versions and Fixes")
    if mi:
        t = mi.find("table")
        if t:
            parsed = parse_table(t)
            if parsed:
                return parsed

    for table in soup.find_all("table"):
        first_row = table.find("tr")
        if not first_row:
            continue
        hdrs = [norm_ws(c.get_text(" ", strip=True)).lower() for c in first_row.find_all(["th", "td"])]
        if (
            (("affected product" in hdrs) or ("受影响产品" in "".join(hdrs)))
            and (("affected version" in hdrs) or ("受影响版本" in "".join(hdrs)))
            and (("repair version" in hdrs) or ("fixed version" in hdrs) or ("修复版本" in "".join(hdrs)))
        ):
            parsed = parse_table(table)
            if parsed:
                return parsed
    return []

def _first_match(text: str, patterns: List[re.Pattern]) -> str:
    if not text:
        return ""
    for pat in patterns:
        m = pat.search(text)
        if m:
            return m.group(1)
    return ""

def extract_scoring_details(soup: BeautifulSoup) -> Dict[str, Any]:
    mi = find_moreinfo_div_for_title(soup, "Vulnerability Scoring Details") or find_moreinfo_div_for_title(soup, "漏洞评分详情")
    text = collect_text_from_container(mi) if mi else ""
    if not text:
        text = norm_ws(soup.get_text(" ", strip=True))

    base = _first_match(text, SCORE_RE_LIST)
    temp = _first_match(text, TEMP_RE_LIST)
    env  = _first_match(text, ENV_RE_LIST)
    vec  = _first_match(text, VEC_RE_LIST)

    return {
        "text": dedup_sentences(text),
        "base_score": base,
        "temporary_score": temp,
        "environmental_score": env,
        "cvss_vector": vec or None,
    }

# ================== NORMALIZERS ==================
def _to_num_or_na(s: Any):
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    if s.upper() in {"NA", "N/A"}:
        return "NA"
    try:
        return float(s)
    except Exception:
        return None

def normalize_cvss(vsd: Dict[str, Any]) -> Dict[str, Any]:
    vsd = vsd or {}
    return {
        "text": (vsd.get("text") or "").strip(),
        "base_score": _to_num_or_na(vsd.get("base_score")),
        "temporary_score": _to_num_or_na(vsd.get("temporary_score")),
        "environmental_score": _to_num_or_na(vsd.get("environmental_score")),
        "cvss_vector": (vsd.get("cvss_vector") or None),
    }

def clean_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    if not (rec.get("description") or "").strip():
        imp = (rec.get("impact") or "").strip()
        rec["description"] = split_sentences(imp)[0] if imp else None

    rec["vulnerability_scoring_details"] = normalize_cvss(rec.get("vulnerability_scoring_details") or {})

    svaf = rec.get("software_versions_and_fixes") or []
    good = []
    for r in svaf:
        if not isinstance(r, dict):
            continue
        ap = r.get("affected_product") or None
        av = r.get("affected_version") or None
        rv = r.get("repair_version") or None
        if any([ap, av, rv]):
            good.append({"affected_product": ap, "affected_version": av, "repair_version": rv})
    rec["software_versions_and_fixes"] = good or []

    for k in [
        "temporary_fix", "faqs", "revision_history", "source", "impact",
        "technique_details", "description", "obtaining_fixed_software", "title", "url"
    ]:
        v = rec.get(k)
        if isinstance(v, str):
            v = v.strip()
        rec[k] = v if v not in ("", "None", "无") else None

    return rec

# ================== DETAIL FETCH ==================
def fetch_detail(url: str) -> Optional[Dict[str, Any]]:
    r = S.get(url, timeout=30)
    if not r.ok:
        return None

    soup = BeautifulSoup(r.text, "lxml")

    cves        = sorted(set(CVERE.findall(r.text)))
    description = extract_description_from_div(soup)
    svaf        = extract_software_versions_and_fixes(soup)
    impact      = extract_section_text(soup, "Impact", ["影响"])
    scoring     = extract_scoring_details(soup)
    technique   = extract_section_text(soup, "Technique Details", ["技术细节"])
    temp_fix    = extract_section_text(soup, "Temporary Fix", ["临时修复"])
    ofs         = extract_section_text(soup, "Obtaining Fixed Software", ["获取修复软件"])
    source      = extract_section_text(soup, "Source", ["来源"])
    history     = extract_section_text(soup, "Revision History", ["修订记录"])
    faqs        = extract_section_text(soup, "FAQs", ["常见问题"])

    title_tag  = soup.find("h1") or soup.find("title")
    page_title = norm_ws(title_tag.get_text()) if title_tag else None

    return {
        "page_title": page_title,
        "description": description,
        "impact": impact,
        "vulnerability_scoring_details": scoring,
        "technique_details": dedup_sentences(technique),
        "temporary_fix": temp_fix or None,
        "obtaining_fixed_software": ofs,
        "source": dedup_sentences(source),
        "revision_history": history,
        "faqs": faqs or None,
        "cves": cves,
        "software_versions_and_fixes": svaf,
    }

# ================== QDRANT DEDUPE ==================
def build_qdrant_client_from_env() -> Optional[QdrantClient]:
    if not QDRANT_URL:
        return None
    return QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY or None)

def _vendor_alias_filter(vendor_values: List[str]) -> Filter:
    vals = [v for v in vendor_values if isinstance(v, str) and v.strip()]
    vals = list(dict.fromkeys(vals))
    if not vals:
        # fallback: impossible filter (but we won't call this with empty)
        return Filter(must=[FieldCondition(key="vendor", match=MatchValue(value="__no_vendor__"))])
    if len(vals) == 1:
        return Filter(must=[FieldCondition(key="vendor", match=MatchValue(value=vals[0]))])
    # OR filter (should)
    return Filter(should=[FieldCondition(key="vendor", match=MatchValue(value=v)) for v in vals])

def fetch_existing_vendor_urls(
    client: QdrantClient,
    collection_name: str,
    vendor_values: List[str],
    page_size: int = 256,
) -> Set[str]:
    existing: Set[str] = set()
    offset = None

    vendor_filter = _vendor_alias_filter(vendor_values)

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
                existing.add(canonicalize_url(u.strip()))
        if offset is None:
            break

    return existing

# ================== TextNode helpers ==================
def _stable_id(vendor_key: str, url: str) -> str:
    # Keep stable across casing changes
    raw = f"{vendor_key}|{url}"
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

def create_text_node_from_advisory(advisory_dict: Dict[str, Any]) -> Optional[TextNode]:
    """
    RULE (as you wrote):
    - node.text = ALL fields except 'url'
    - node.metadata = ONLY {"url": "<canonical_url>"}
    """
    url = canonicalize_url(str(advisory_dict.get("url") or "").strip())
    if not url:
        return None

    rec = dict(advisory_dict)
    # Store vendor with desired casing in TEXT
    rec["vendor"] = rec.get("vendor") or VENDOR_VALUE

    text = record_to_text(rec)
    if not text:
        return None

    node_id = _stable_id(VENDOR_KEY, url)

    return TextNode(
        id_=node_id,
        text=text,
        metadata={"url": url},  # ONLY url
    )

# ================== LIST API ==================
def post_page(payload: Dict[str, Any]) -> Dict[str, Any]:
    r = S.post(POST_URL, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()

def get_items_and_total(data: Dict[str, Any]):
    items, total = [], 0
    dd = data.get("data") if isinstance(data, dict) else None
    if isinstance(dd, dict):
        items = dd.get("results") or []
        t = dd.get("total")
        if isinstance(t, int):
            total = t
    return items, total

# ================== MAIN CRAWL ==================
def get_all_advisories(check_qdrant: bool = True) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    Returns:
      - records: list of NEW advisory records (dicts)
      - stats: {"scraped": X, "skipped_existing": Y, "total_out": Z}
    """
    qdrant = None
    existing_urls: Set[str] = set()

    if check_qdrant:
        qdrant = build_qdrant_client_from_env()
        if qdrant is None:
            logger.warning("check_qdrant=True but QDRANT_URL not set; running without dedupe.")
            check_qdrant = False
        else:
            try:
                existing_urls = fetch_existing_vendor_urls(qdrant, QDRANT_COLLECTION, VENDOR_ALIASES)
                logger.info(
                    "Loaded %d existing URLs from Qdrant for vendor aliases=%s",
                    len(existing_urls),
                    VENDOR_ALIASES,
                )
            except Exception as e:
                logger.warning("Failed to preload existing URLs from Qdrant (%s). Will fallback to scraping.", e)
                existing_urls = set()

    payload = dict(BASE_PAYLOAD)
    payload["pageNum"] = "1"

    first = post_page(payload)
    _, total = get_items_and_total(first)
    page_size = int(payload["pageSize"])
    total_pages = math.ceil(total / page_size) if page_size else 1

    all_records: List[Dict[str, Any]] = []
    seen_urls: Set[str] = set()

    skipped = 0
    scraped = 0

    for page in range(1, total_pages + 1):
        payload["pageNum"] = str(page)
        data = post_page(payload)
        items, _ = get_items_and_total(data)
        if not items:
            time.sleep(0.35)
            continue

        for it in items:
            if not isinstance(it, dict):
                continue

            raw_url = it.get("pageUrl") or it.get("linkUrl") or it.get("url") or it.get("detailUrl") or ""
            url = canonicalize_url(raw_url)
            if not url:
                continue

            if classify_item(url, it.get("title")) != "advisory":
                continue

            if url in seen_urls:
                continue
            seen_urls.add(url)

            # fast dedupe BEFORE detail fetch
            if check_qdrant and existing_urls and url in existing_urls:
                skipped += 1
                continue

            detail = None
            for _ in range(2):
                try:
                    detail = fetch_detail(url)
                    if detail:
                        break
                except requests.RequestException:
                    pass
                time.sleep(0.4)

            if not detail:
                continue

            rec = {
                "vendor": VENDOR_VALUE,  # ✅ "Huawei"
                "title": it.get("title") or detail.get("page_title"),
                "url": url,
                "description": detail.get("description"),
                "impact": detail.get("impact"),
                "vulnerability_scoring_details": detail.get("vulnerability_scoring_details"),
                "technique_details": detail.get("technique_details"),
                "temporary_fix": detail.get("temporary_fix"),
                "obtaining_fixed_software": detail.get("obtaining_fixed_software"),
                "source": detail.get("source"),
                "revision_history": detail.get("revision_history"),
                "faqs": detail.get("faqs"),
                "cves": detail.get("cves"),
                "software_versions_and_fixes": detail.get("software_versions_and_fixes"),
                "scraped_date": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
            }
            rec = clean_record(rec)
            all_records.append(rec)
            scraped += 1

        time.sleep(0.35)

    logger.info("Done. scraped=%d skipped_existing=%d total_out=%d", scraped, skipped, len(all_records))
    return all_records, {"scraped": scraped, "skipped_existing": skipped, "total_out": len(all_records)}

# ================== PUBLIC ENTRYPOINT ==================
def scrape_huawei_nodes(
    *,
    check_qdrant: bool = True,
) -> List[TextNode]:
    """
    Returns List[TextNode] (NEW only)
    """
    records, stats = get_all_advisories(check_qdrant=check_qdrant)

    nodes: List[TextNode] = []
    for rec in records:
        node = create_text_node_from_advisory(rec)
        if node:
            nodes.append(node)

    logger.info(
        "TextNodes created: %d | scraped=%d skipped_existing=%d total_out=%d | vendor_value=%s | aliases=%s",
        len(nodes),
        stats.get("scraped", 0),
        stats.get("skipped_existing", 0),
        stats.get("total_out", len(records)),
        VENDOR_VALUE,
        VENDOR_ALIASES,
    )
    return nodes

def scrape_huawei_debug(
    *,
    check_qdrant: bool = True,
) -> Dict[str, Any]:
    records, stats = get_all_advisories(check_qdrant=check_qdrant)

    nodes: List[TextNode] = []
    for rec in records:
        node = create_text_node_from_advisory(rec)
        if node:
            nodes.append(node)

    out: Dict[str, Any] = {
        "ok": True,
        "vendor": VENDOR_VALUE,              # "Huawei"
        "vendor_key": VENDOR_KEY,            # "huawei"
        "vendor_aliases": VENDOR_ALIASES,    # e.g. ["Huawei","huawei"]
        "nodes": nodes,
        "per_source": {VENDOR_KEY: len(nodes)},
        "stats": stats,
        "records": records,
    }
    logger.info("TextNodes created: %d", len(nodes))
    return out

# ================== CLI DEBUG ==================
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    res = scrape_huawei_debug(check_qdrant=True)

    # Save records (optional)
    records = res.get("records") or []
    OUT_FILE.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"✅ Saved {len(records)} NEW Huawei advisories → {OUT_FILE.resolve()}")

    # Preview first node
    nodes = res["nodes"]
    if nodes:
        n0 = nodes[0]
        print("\n--- NODE PREVIEW ---")
        print("id_:", n0.id_)
        print("metadata:", n0.metadata)  # ONLY {"url": "..."}
        print("text (first 800 chars):\n", n0.text[:800])
