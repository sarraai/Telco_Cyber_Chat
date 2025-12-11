import json
import logging
import math
import re
import time
from pathlib import Path
from typing import List, Dict, Any, Optional

import requests
from bs4 import BeautifulSoup, Tag

from telco_cyber_chat.webscraping.scrape_core import url_already_ingested

# ================== LOGGER ==================

logger = logging.getLogger(__name__)

# ================== CONFIG & SESSION ==================

REFERRER = "https://www.huawei.com/en/psirt/all-bulletins?page=1"
POST_URL = "https://www.huawei.com/service/portalapplication/v1/corp/psirt"

BASE_PAYLOAD: Dict[str, object] = {
    "contentId": "aadaee27bbac4341a6d2014c788a2c85",
    "catalogPathList": ["/psirt/"],
    "pageNum": "1",
    "pageSize": 20,
    "time": "",
    "filterLabelList": [[]],
}

S = requests.Session()
S.headers.update(
    {
        "User-Agent": "Mozilla/5.0",
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "Content-Type": "application/json",
        "Referer": REFERRER,
        "Origin": "https://www.huawei.com",
        "X-Requested-With": "XMLHttpRequest",
    }
)

# Optional local dump when run as a script
OUT_FILE = Path("huawei_psirt_advisories_all.json")

# ================== REGEX HELPERS ==================

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

# ================== BASIC UTILITIES ==================


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
    return "https://www.huawei.com/" + u


def classify_item(url: str, title: Optional[str]) -> Optional[str]:
    """
    Classify list entry as 'advisory' / 'notice' / None based on URL + title.
    """
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
    return " ".join(out).strip()


# ================== HTML PARSING HELPERS ==================


def find_moreinfo_div_for_title(
    soup: BeautifulSoup, section_title: str
) -> Optional[Tag]:
    """
    Huawei PSIRT pages use <a data-expand="Section"> + a sibling .moreinfo div.
    """
    a = soup.find("a", attrs={"data-expand": section_title}) or soup.find(
        "a", attrs={"data-collapse": section_title}
    )
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


def extract_summary_from_div(soup: BeautifulSoup) -> str:
    """
    Try to extract the 'Summary' section (or Chinese 摘要).
    Fallback to the first reasonable <p>.
    """
    node = soup.find("div", class_="summary")
    if isinstance(node, Tag):
        ps = node.find_all("p")
        if ps:
            texts = [norm_ws(p.get_text(" ", strip=True)) for p in ps]
            return " ".join([t for t in texts if t])
        return norm_ws(node.get_text(" ", strip=True))

    mi = find_moreinfo_div_for_title(soup, "Summary") or find_moreinfo_div_for_title(
        soup, "摘要"
    )
    if mi:
        inner = mi.find("div", class_="summary")
        if inner:
            txt = norm_ws(inner.get_text(" ", strip=True))
            if txt:
                return txt
        txt = collect_text_from_container(mi)
        if txt:
            return txt

    # fallback: first decent paragraph
    for p in soup.find_all("p"):
        txt = norm_ws(p.get_text(" ", strip=True))
        if txt and "Vulnerabilities are scored based on the CVSS" not in txt:
            return txt
    return ""


def extract_section_text(
    soup: BeautifulSoup, title: str, cn_variants: Optional[List[str]] = None
) -> str:
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

    headers_raw = [
        norm_ws(c.get_text(" ", strip=True)) for c in rows[0].find_all(["th", "td"])
    ]

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

    # fallback: detect by header text
    for table in soup.find_all("table"):
        first_row = table.find("tr")
        if not first_row:
            continue
        hdrs = [
            norm_ws(c.get_text(" ", strip=True)).lower()
            for c in first_row.find_all(["th", "td"])
        ]
        if (
            ("affected product" in hdrs or "受影响产品" in "".join(hdrs))
            and ("affected version" in hdrs or "受影响版本" in "".join(hdrs))
            and (
                "repair version" in hdrs
                or "fixed version" in hdrs
                or "修复版本" in "".join(hdrs)
            )
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
    """
    Extract CVSS text + base/temporal/environmental scores + vector.
    """
    mi = find_moreinfo_div_for_title(
        soup, "Vulnerability Scoring Details"
    ) or find_moreinfo_div_for_title(soup, "漏洞评分详情")
    text = collect_text_from_container(mi) if mi else ""
    if not text:
        text = norm_ws(soup.get_text(" ", strip=True))

    base = _first_match(text, SCORE_RE_LIST)
    temp = _first_match(text, TEMP_RE_LIST)
    env = _first_match(text, ENV_RE_LIST)
    vec = _first_match(text, VEC_RE_LIST)

    return {
        "text": dedup_sentences(text),
        "base_score": base,
        "temporary_score": temp,
        "environmental_score": env,
        "cvss_vector": vec or None,
    }


# ================== DETAIL PAGE SCRAPING ==================


def fetch_detail(url: str) -> Optional[Dict[str, Any]]:
    """
    Fetch a single advisory detail page and return the fields we still care about.
    """
    try:
        r = S.get(url, timeout=30)
        r.raise_for_status()
    except requests.RequestException as e:
        logger.warning("[HUAWEI] HTTP error fetching detail %s: %s", url, e)
        return None

    # Use built-in HTML parser (no lxml dependency)
    soup = BeautifulSoup(r.text, "html.parser")

    cves = sorted(set(CVERE.findall(r.text)))
    summary = extract_summary_from_div(soup)
    svaf = extract_software_versions_and_fixes(soup)
    impact = extract_section_text(soup, "Impact", ["影响"])
    scoring = extract_scoring_details(soup)
    technique = extract_section_text(soup, "Technique Details", ["技术细节"])
    temp_fix = extract_section_text(soup, "Temporary Fix", ["临时修复"])
    ofs = extract_section_text(soup, "Obtaining Fixed Software", ["获取修复软件"])

    title_tag = soup.find("h1") or soup.find("title")
    page_title = norm_ws(title_tag.get_text()) if title_tag else None

    return {
        "page_title": page_title,
        "summary": summary,
        "impact": impact,
        "vulnerability_scoring_details": scoring,
        "technique_details": dedup_sentences(technique),
        "temporary_fix": temp_fix,
        "obtaining_fixed_software": ofs,
        "cves": cves,
        "software_versions_and_fixes": svaf,
    }


# ================== LIST API HELPERS ==================


def post_page(payload: Dict[str, object]) -> Dict[str, object]:
    r = S.post(POST_URL, json=payload, timeout=30)
    r.raise_for_status()
    return r.json()


def get_items_and_total(data: Dict[str, object]):
    items, total = [], 0
    dd = data.get("data") if isinstance(data, dict) else None
    if isinstance(dd, dict):
        items = dd.get("results") or []
        t = dd.get("total")
        if isinstance(t, int):
            total = t
    return items, total


# ================== CVSS NORMALIZATION ==================


def _to_num_or_na(s: Any):
    """
    Returns float if numeric string like '7.3',
    returns 'NA' if explicitly NA/N/A,
    returns None if empty/unknown.
    """
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
    """
    Minimal cleaning for the fields we still care about.
    No id / published / extra Huawei boilerplate.
    """
    # Summary fallback from impact
    if not (rec.get("summary") or "").strip():
        imp = (rec.get("impact") or "").strip()
        rec["summary"] = split_sentences(imp)[0] if imp else None

    # Normalize CVSS block
    rec["vulnerability_scoring_details"] = normalize_cvss(
        rec.get("vulnerability_scoring_details")  # type: ignore[arg-type]
    )

    # SV&F cleanup with guaranteed keys
    svaf = rec.get("software_versions_and_fixes") or []
    good: List[Dict[str, Optional[str]]] = []
    for r in svaf:
        if not isinstance(r, dict):
            continue
        ap = r.get("affected_product") or None
        av = r.get("affected_version") or None
        rv = r.get("repair_version") or None
        if any([ap, av, rv]):
            good.append(
                {
                    "affected_product": ap,
                    "affected_version": av,
                    "repair_version": rv,
                }
            )
    rec["software_versions_and_fixes"] = good or []

    # Normalize empties to None for the used text fields
    for k in [
        "temporary_fix",
        "impact",
        "technique_details",
        "obtaining_fixed_software",
        "summary",
    ]:
        v = rec.get(k)
        if isinstance(v, str):
            v = v.strip()
        rec[k] = v if v not in ("", "None", "无") else None

    return rec


# ================== RECORD → DOCUMENT (title + merged description) ==================


def build_document_from_record(rec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create the final minimal document:

      {
        "title": "...",
        "url": "...",
        "description": "Summary + impact + CVSS + technique + fixes + SV&F + CVEs"
      }
    """
    url = (rec.get("url") or "").strip()
    title = (rec.get("title") or rec.get("page_title") or "Huawei Security Advisory").strip()

    parts: List[str] = []

    # 1) Summary → base description
    summary = rec.get("summary")
    if isinstance(summary, str) and summary.strip():
        parts.append("Summary: " + summary.strip())

    # 2) Impact
    impact = rec.get("impact")
    if isinstance(impact, str) and impact.strip():
        parts.append("Impact: " + impact.strip())

    # 3) Vulnerability scoring details
    vsd = rec.get("vulnerability_scoring_details") or {}
    vs_text = (vsd.get("text") or "").strip()
    base = vsd.get("base_score")
    temp = vsd.get("temporary_score")
    env = vsd.get("environmental_score")
    vec = vsd.get("cvss_vector")

    cvss_bits: List[str] = []
    if vs_text:
        cvss_bits.append(vs_text)
    if base is not None:
        cvss_bits.append(f"Base score: {base}")
    if temp is not None:
        cvss_bits.append(f"Temporary score: {temp}")
    if env is not None:
        cvss_bits.append(f"Environmental score: {env}")
    if vec:
        cvss_bits.append(f"CVSS vector: {vec}")
    if cvss_bits:
        parts.append("Vulnerability scoring details: " + " ".join(map(str, cvss_bits)))

    # 4) Technique details
    technique = rec.get("technique_details")
    if isinstance(technique, str) and technique.strip():
        parts.append("Technique details: " + technique.strip())

    # 5) Temporary fix
    temp_fix = rec.get("temporary_fix")
    if isinstance(temp_fix, str) and temp_fix.strip():
        parts.append("Temporary fix: " + temp_fix.strip())

    # 6) Obtaining fixed software
    ofs = rec.get("obtaining_fixed_software")
    if isinstance(ofs, str) and ofs.strip():
        parts.append("Obtaining fixed software: " + ofs.strip())

    # 7) Software versions and fixes (as text)
    svaf = rec.get("software_versions_and_fixes") or []
    if isinstance(svaf, list) and svaf:
        lines: List[str] = []
        for row in svaf:
            if not isinstance(row, dict):
                continue
            ap = (row.get("affected_product") or "").strip()
            av = (row.get("affected_version") or "").strip()
            rv = (row.get("repair_version") or "").strip()
            bits: List[str] = []
            if ap:
                bits.append(ap)
            if av:
                bits.append(f"affected version: {av}")
            if rv:
                bits.append(f"repair version: {rv}")
            if bits:
                lines.append(" – ".join(bits))
        if lines:
            parts.append("Software versions and fixes:\n" + "\n".join(lines))

    # 8) CVEs (merged into description)
    cves = rec.get("cves") or []
    if isinstance(cves, list) and cves:
        parts.append("Related CVEs: " + ", ".join(map(str, cves)))

    description = "\n".join(parts).strip()

    return {
        "title": title,
        "url": url,
        "description": description or title,
    }


# ================== FULL CRAWL → MINIMAL DOCS (Cisco-style pattern) ==================


def get_all_advisories(check_qdrant: bool = True) -> List[Dict[str, Any]]:
    """
    Crawl Huawei PSIRT advisories following Cisco scraper pattern:

    1. Fetch all advisory metadata from API (paginated)
    2. For each advisory:
       - Check if URL already ingested in Qdrant (if check_qdrant=True)
       - Skip if already exists (BEFORE fetching detail page)
       - Otherwise fetch + parse the detail page
    3. Return list of minimal documents: {title, url, description}

    This follows the same strategy as cisco_scraper.py for efficiency.
    """
    logger.info("[HUAWEI] Starting advisory crawl (check_qdrant=%s)", check_qdrant)

    all_docs: List[Dict[str, Any]] = []
    seen: set[str] = set()

    payload = dict(BASE_PAYLOAD)
    payload["pageNum"] = "1"

    try:
        data = post_page(payload)
    except Exception as e:
        logger.error("[HUAWEI] Initial POST failed: %s", e)
        return all_docs

    items, total = get_items_and_total(data)
    page_size = int(payload["pageSize"])  # type: ignore[arg-type]
    total_pages = math.ceil(total / page_size) if page_size else 1

    for page in range(1, total_pages + 1):
        if page > 1:
            payload["pageNum"] = str(page)
            try:
                data = post_page(payload)
            except Exception as e:
                logger.warning("[HUAWEI] POST failed for page %s: %s", page, e)
                time.sleep(0.35)
                continue
            items, _ = get_items_and_total(data)
            if not items:
                time.sleep(0.35)
                continue

        for it in items:
            if not isinstance(it, dict):
                continue

            # Get URL from API response
            url = absolutize(
                it.get("pageUrl")
                or it.get("linkUrl")
                or it.get("url")
                or it.get("detailUrl")
                or ""
            )
            if not url:
                continue
            
            # Only process advisories (not notices)
            if classify_item(url, it.get("title")) != "advisory":
                continue
            
            # Skip duplicates in this run
            if url in seen:
                continue
            seen.add(url)

            # Cisco-style Qdrant check: skip BEFORE expensive detail page fetch
            if check_qdrant and url and url_already_ingested(url):
                logger.info("[HUAWEI] Skipping already-ingested URL: %s", url)
                continue

            # If we reach here, it's a new URL - fetch detail page
            detail = None
            for _ in range(2):
                try:
                    detail = fetch_detail(url)
                    if detail:
                        break
                except requests.RequestException as e:
                    logger.warning(
                        "[HUAWEI] Error fetching detail for %s: %s", url, e
                    )
                time.sleep(0.4)

            if not detail:
                logger.warning("[HUAWEI] No detail parsed for URL: %s", url)
                continue

            # Build record from API metadata + detail page
            rec: Dict[str, Any] = {
                "id": it.get("id"),
                "title": it.get("title") or detail.get("page_title"),
                "url": url,
                "summary": detail["summary"],
                "impact": detail["impact"],
                "vulnerability_scoring_details": detail[
                    "vulnerability_scoring_details"
                ],
                "technique_details": detail["technique_details"],
                "temporary_fix": detail["temporary_fix"],
                "obtaining_fixed_software": detail["obtaining_fixed_software"],
                "cves": detail["cves"],
                "software_versions_and_fixes": detail[
                    "software_versions_and_fixes"
                ],
            }

            rec = clean_record(rec)
            doc = build_document_from_record(rec)
            all_docs.append(doc)

        time.sleep(0.35)  # polite delay between pages

    logger.info("[HUAWEI] Scraped %d new advisories (documents).", len(all_docs))
    return all_docs


# ================== OPTIONAL: CLEAN EXISTING RICH JSON ==================


def clean_existing_json(in_path: str, out_path: str) -> None:
    """
    If you already scraped with the old structure,
    this will convert it into the minimal {title,url,description} format.
    """
    data = json.loads(Path(in_path).read_text(encoding="utf-8"))
    docs: List[Dict[str, Any]] = []
    for rec in data:
        if not isinstance(rec, dict):
            continue
        rec = clean_record(rec)
        doc = build_document_from_record(rec)
        docs.append(doc)
    Path(out_path).write_text(
        json.dumps(docs, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    logger.info("✅ Converted %d records → %s", len(docs), out_path)


# ================== PUBLIC ENTRYPOINT ==================


def scrape_huawei(check_qdrant: bool = True) -> List[Dict[str, Any]]:
    """
    High-level scraper used by your cron / scraping graph.

      - Calls get_all_advisories(check_qdrant)
      - Returns list of documents:
            {title, url, description}
    """
    return get_all_advisories(check_qdrant=check_qdrant)


# ================== CLI DEBUG (LOCAL USE ONLY) ==================


if __name__ == "__main__":
    # Local run: don't check Qdrant by default
    records = scrape_huawei(check_qdrant=False)
    OUT_FILE.write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"✅ Saved {len(records)} advisories → {OUT_FILE.resolve()}")

    if records:
        print("\n📄 Example advisory (index 0):")
        print(json.dumps(records[0], indent=2, ensure_ascii=False))
    else:
        print("No advisories scraped.")
