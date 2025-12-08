import re
import math
import time
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup, Tag

from telco_cyber_chat.webscraping.scrape_core import url_already_ingested

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
        rf"\bCVSS\s*v?3(?:\.\d+)?\s*Vector{RE_COLON}([A-Za-z]{1,3}:[^ \t\r\n]+(?:/[A-Za-z]{1,3}:[^ \t\r\n]+)*)",
        re.I,
    ),
    re.compile(
        rf"\bCVSS\s*v?3(?:\.\d+)?\s*向量{RE_COLON}([A-Za-z]{1,3}:[^ \t\r\n]+(?:/[A-Za-z]{1,3}:[^ \t\r\n]+)*)"
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
    """Classify list entry as 'advisory' / 'notice' / None based on URL + title."""
    u = (url or "").lower()
    if "/psirt/security-advisories/" in u:
        return "advisory"
    if "/psirt/security-notices/" in u:
        return "notice"
    if title:
        t = title.lower()
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
    return " ".join(out)


# ================== HTML PARSING HELPERS ==================


def find_moreinfo_div_for_title(
    soup: BeautifulSoup, section_title: str
) -> Optional[Tag]:
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
    parts = []
    for el in node.find_all(["p", "li", "div"], recursive=True):
        txt = norm_ws(el.get_text(" ", strip=True))
        if txt:
            parts.append(txt)
    if not parts:
        txt = norm_ws(node.get_text(" ", strip=True))
        if txt:
            parts.append(txt)
    out = []
    for t in parts:
        if not out or t != out[-1]:
            out.append(t)
    return " ".join(out).strip()


def extract_summary_from_div(soup: BeautifulSoup) -> str:
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


def extract_scoring_details(soup: BeautifulSoup) -> Dict[str, Optional[str]]:
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


def extract_declaration(soup: BeautifulSoup) -> str:
    mi = find_moreinfo_div_for_title(soup, "Declaration") or find_moreinfo_div_for_title(
        soup, "声明"
    )
    if mi:
        return collect_text_from_container(mi)

    a = soup.find("a", string=lambda s: bool(s and s.strip().lower() == "declaration"))
    if a:
        wrapper = a.find_parent(class_="psirt-set-out")
        if wrapper:
            mi = wrapper.find("div", class_="moreinfo")
            if mi:
                return collect_text_from_container(mi)

    for a in soup.find_all("a"):
        for attr in ("data-expand", "data-collapse"):
            val = a.get(attr)
            if val and ("declaration" in val.strip().lower() or "声明" in val):
                wrapper = a.find_parent(class_="psirt-set-out")
                if wrapper:
                    mi = wrapper.find("div", class_="moreinfo")
                    if mi:
                        return collect_text_from_container(mi)

    return ""


# ================== DETAIL PAGE SCRAPING ==================


def fetch_detail(url: str) -> Optional[Dict[str, object]]:
    r = S.get(url, timeout=30)
    if not r.ok:
        return None

    soup = BeautifulSoup(r.text, "lxml")

    cves = sorted(set(CVERE.findall(r.text)))
    summary = extract_summary_from_div(soup)
    svaf = extract_software_versions_and_fixes(soup)
    impact = extract_section_text(soup, "Impact", ["影响"])
    scoring = extract_scoring_details(soup)
    technique = extract_section_text(soup, "Technique Details", ["技术细节"])
    temp_fix = extract_section_text(soup, "Temporary Fix", ["临时修复"])
    ofs = extract_section_text(soup, "Obtaining Fixed Software", ["获取修复软件"])
    source = extract_section_text(soup, "Source", ["来源"])
    history = extract_section_text(soup, "Revision History", ["修订记录"])
    faqs = extract_section_text(soup, "FAQs", ["常见问题"])
    decl = extract_declaration(soup)

    title_tag = soup.find("h1") or soup.find("title")
    page_title = norm_ws(title_tag.get_text()) if title_tag else None

    return {
        "page_title": page_title,
        "summary": summary,
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
        "declaration": decl,
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


def _to_num_or_na(s):
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


def normalize_cvss(vsd: Dict[str, object]) -> Dict[str, object]:
    vsd = vsd or {}
    return {
        "text": (vsd.get("text") or "").strip(),
        "base_score": _to_num_or_na(vsd.get("base_score")),
        "temporary_score": _to_num_or_na(vsd.get("temporary_score")),
        "environmental_score": _to_num_or_na(vsd.get("environmental_score")),
        "cvss_vector": (vsd.get("cvss_vector") or None),
    }


def clean_record(rec: Dict[str, object]) -> Dict[str, object]:
    """Normalize CVSS + versions + empty fields; no id/published kept."""
    # Fallback: summary from first sentence of impact
    if not (rec.get("summary") or ""):
        imp = (rec.get("impact") or "").strip()
        rec["summary"] = split_sentences(imp)[0] if imp else None

    # Keep CVSS only under vulnerability_scoring_details
    rec["vulnerability_scoring_details"] = normalize_cvss(
        rec.get("vulnerability_scoring_details")  # type: ignore[arg-type]
    )
    for k in [
        "cvss_base_score",
        "cvss_temporary_score",
        "cvss_environmental_score",
        "cvss_vector",
    ]:
        rec.pop(k, None)

    # Normalize software_versions_and_fixes
    svaf = rec.get("software_versions_and_fixes") or []
    good = []
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

    # Normalize empty strings to None
    for k in [
        "temporary_fix",
        "faqs",
        "declaration",
        "revision_history",
        "source",
        "impact",
        "technique_details",
        "summary",
    ]:
        v = rec.get(k)
        if isinstance(v, str):
            v = v.strip()
        rec[k] = v if v not in ("", "None", "无") else None

    return rec


# ================== FULL CRAWL (NO id/published/huawei_security_procedures) ==================


def get_all_advisories(check_qdrant: bool = True) -> List[Dict[str, object]]:
    """
    Fetch and normalize all Huawei security advisories.

    Returns records with:
      - url
      - title
      - summary, impact
      - vulnerability_scoring_details
      - technique_details
      - temporary_fix
      - obtaining_fixed_software
      - source
      - revision_history
      - faqs
      - cves
      - software_versions_and_fixes
      - declaration

    No id / published / huawei_security_procedures.
    If check_qdrant=True, skip URLs already stored in Qdrant
    BEFORE fetching the detail page.
    """
    all_records: List[Dict[str, object]] = []
    seen: set[str] = set()

    payload = dict(BASE_PAYLOAD)
    payload["pageNum"] = "1"
    data = post_page(payload)
    items, total = get_items_and_total(data)
    page_size = int(payload["pageSize"])  # type: ignore[arg-type]
    total_pages = math.ceil(total / page_size) if page_size else 1

    for page in range(1, total_pages + 1):
        if page > 1:
            payload["pageNum"] = str(page)
            data = post_page(payload)
            items, _ = get_items_and_total(data)
            if not items:
                time.sleep(0.35)
                continue

        for it in items:
            if not isinstance(it, dict):
                continue

            url = absolutize(
                it.get("pageUrl")
                or it.get("linkUrl")
                or it.get("url")
                or it.get("detailUrl")
                or ""
            )
            if not url:
                continue
            if classify_item(url, it.get("title")) != "advisory":
                continue
            if url in seen:
                continue
            seen.add(url)

            # ✅ Qdrant check BEFORE scraping detail page
            if check_qdrant and url_already_ingested(url):
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
                # skip broken detail pages
                continue

            rec: Dict[str, object] = {
                "url": url,
                "title": it.get("title") or detail.get("page_title"),
                "summary": detail["summary"],
                "impact": detail["impact"],
                "vulnerability_scoring_details": detail[
                    "vulnerability_scoring_details"
                ],
                "technique_details": detail["technique_details"],
                "temporary_fix": detail["temporary_fix"],
                "obtaining_fixed_software": detail["obtaining_fixed_software"],
                "source": detail["source"],
                "revision_history": detail["revision_history"],
                "faqs": detail["faqs"],
                "cves": detail["cves"],
                "software_versions_and_fixes": detail[
                    "software_versions_and_fixes"
                ],
                "declaration": detail["declaration"],
            }

            rec = clean_record(rec)
            all_records.append(rec)

        time.sleep(0.35)  # polite

    return all_records


# ================== RECORD → DOCUMENT (title + merged description) ==================


def huawei_record_to_document(rec: Dict[str, object]) -> Dict[str, object]:
    """
    Build a minimal document for RAG ingestion:

      - url
      - title
      - description: summary (renamed) + merged with
         impact, CVSS, technique_details, temporary_fix,
         obtaining_fixed_software, source, revision_history,
         faqs, cves, software_versions_and_fixes, declaration
    """
    url = (rec.get("url") or "").strip()  # type: ignore[union-attr]
    title = (
        (rec.get("title") or "Huawei Security Advisory")  # type: ignore[operator]
        .strip()
    )

    parts: List[str] = []

    # Summary → Description
    summary = rec.get("summary")
    if isinstance(summary, str) and summary.strip():
        parts.append("Description: " + summary.strip())

    # Impact
    impact = rec.get("impact")
    if isinstance(impact, str) and impact.strip():
        parts.append("Impact: " + impact.strip())

    # CVSS details
    vsd = rec.get("vulnerability_scoring_details") or {}
    if isinstance(vsd, dict):
        base = vsd.get("base_score")
        temp = vsd.get("temporary_score")
        env = vsd.get("environmental_score")
        vec = vsd.get("cvss_vector")
        cvss_bits: List[str] = []
        if base not in (None, "", "NA"):
            cvss_bits.append(f"base score {base}")
        if temp not in (None, "", "NA"):
            cvss_bits.append(f"temporary score {temp}")
        if env not in (None, "", "NA"):
            cvss_bits.append(f"environmental score {env}")
        if vec:
            cvss_bits.append(f"vector {vec}")
        if cvss_bits:
            parts.append("CVSS details: " + ", ".join(map(str, cvss_bits)))

    # Affected products and versions
    svaf = rec.get("software_versions_and_fixes") or []
    if isinstance(svaf, list) and svaf:
        lines: List[str] = []
        for r in svaf:
            if not isinstance(r, dict):
                continue
            ap = (r.get("affected_product") or "").strip()
            av = (r.get("affected_version") or "").strip()
            rv = (r.get("repair_version") or "").strip()
            bits: List[str] = []
            if ap:
                bits.append(ap)
            if av:
                bits.append(f"affected version: {av}")
            if rv:
                bits.append(f"fixed in: {rv}")
            if bits:
                lines.append(" – ".join(bits))
        if lines:
            parts.append("Affected products and versions:\n" + "\n".join(lines))

    # CVEs
    cves = rec.get("cves") or []
    if isinstance(cves, list) and cves:
        parts.append("Related CVEs: " + ", ".join(map(str, cves)))

    # Mitigation / fixes
    temp_fix = rec.get("temporary_fix")
    if isinstance(temp_fix, str) and temp_fix.strip():
        parts.append("Temporary fix: " + temp_fix.strip())

    ofs = rec.get("obtaining_fixed_software")
    if isinstance(ofs, str) and ofs.strip():
        parts.append("Obtaining fixed software: " + ofs.strip())

    # Technical details
    tech = rec.get("technique_details")
    if isinstance(tech, str) and tech.strip():
        parts.append("Technical details: " + tech.strip())

    # Source
    source = rec.get("source")
    if isinstance(source, str) and source.strip():
        parts.append("Source: " + source.strip())

    # Revision history
    history = rec.get("revision_history")
    if isinstance(history, str) and history.strip():
        parts.append("Revision history: " + history.strip())

    # FAQs
    faqs = rec.get("faqs")
    if isinstance(faqs, str) and faqs.strip():
        parts.append("FAQs: " + faqs.strip())

    # Declaration
    decl = rec.get("declaration")
    if isinstance(decl, str) and decl.strip():
        parts.append("Declaration: " + decl.strip())

    description = "\n".join(parts).strip()

    return {
        "url": url,
        "title": title,
        "description": description or title,
    }


# ================== PUBLIC ENTRYPOINT ==================


def scrape_huawei(check_qdrant: bool = True) -> List[Dict[str, object]]:
    """
    High-level scraper used by your cron / scraping graph.

      - Calls get_all_advisories(check_qdrant)
      - Returns list of documents:
            {url, title, description}
    """
    records = get_all_advisories(check_qdrant=check_qdrant)
    docs: List[Dict[str, object]] = []

    for rec in records:
        url = (rec.get("url") or "").strip()  # type: ignore[union-attr]
        if not url:
            continue

        doc = huawei_record_to_document(rec)
        docs.append(doc)

    print(f"[HUAWEI] Scraped {len(docs)} advisories (documents).")
    return docs
