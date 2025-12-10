import os, re, json
import logging
from typing import Any, Dict, List, Optional
from datetime import datetime

from telco_cyber_chat.webscraping.scrape_core import url_already_ingested

# ====== LOGGING ======
logger = logging.getLogger(__name__)

# ====== CONFIG (INPUT ONLY, NO OUTPUT FILES) ======
# New: single configurable path for the cleaned VARIoT JSON
VARIOT_JSON_PATH = (
    os.getenv("VARIOT_JSON_PATH")
    or os.getenv("VARIOT_VULN_JSON")  # backward compatible with old env var
    or "/content/drive/MyDrive/VarIoT/VarIoTvuln_cleaned.json"
)

# ====== HELPERS ======
CVE_RE = re.compile(r"\bCVE-\d{4}-\d{4,7}\b", re.I)
CWE_RE = re.compile(r"\bCWE-\d{1,4}\b", re.I)

SECTION_HEADERS = {
    "vendor:",
    "product:",
    "download:",
    "vulnerability type:",
    "cve reference:",
    "security issue:",
    "exploit:",
    "network access:",
    "severity:",
    "disclosure timeline:",
    "references:",
    "tags:",
    "credits:",
    "sources:",
    "description:",
}


def norm_text(x: Any) -> str:
    if x is None:
        return ""
    if not isinstance(x, str):
        x = str(x)
    x = re.sub(r"\r\n?", "\n", x)
    x = re.sub(r"[ \t]+$", "", x, flags=re.MULTILINE)
    x = re.sub(r"(?:\n\s*){2,}", "\n", x)
    return x.strip()


def get_data(obj: dict, key: str, default=""):
    val = obj.get(key, default)
    if isinstance(val, dict) and "data" in val:
        return val.get("data", default)
    return val


def coalesce(*vals):
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v
        if v not in (None, "", [], {}):
            return v
    return ""


def parse_iso_date(s: Any) -> Optional[str]:
    """Return YYYY-MM-DD if parseable; else None."""
    if not s:
        return None
    s = str(s).strip()
    # quick cut if it's already YYYY-MM-DD*
    if re.match(r"^\d{4}-\d{2}-\d{2}", s):
        return s[:10]
    try:
        # handle Z/offsets
        s2 = s.replace("Z", "+00:00")
        dt = datetime.fromisoformat(s2)
        return dt.date().isoformat()
    except Exception:
        try:
            from dateutil import parser as du  # optional fallback
            return du.parse(s, fuzzy=True).date().isoformat()
        except Exception:
            return None


def parse_section(text: str, header: str) -> str:
    if not text:
        return ""
    lines = text.splitlines()
    header_lc = header.lower().strip()
    start = -1
    for i, line in enumerate(lines):
        if line.lower().strip().startswith(header_lc):
            start = i + 1
            break
    if start == -1:
        return ""
    end = len(lines)
    for j in range(start, len(lines)):
        l = lines[j].strip().lower()
        if l.endswith(":") and (l in SECTION_HEADERS):
            end = j
            break
    return "\n".join(lines[start:end]).strip()


def extract_description(v: dict) -> str:
    # prefer common keys
    for k in ("description", "summary", "details", "content"):
        val = v.get(k)
        if isinstance(val, dict):
            data = val.get("data")
            if isinstance(data, dict) and "data" in data:
                data = data["data"]
            if not data:
                for kk in ("description", "text", "value"):
                    if isinstance(val.get(kk), str):
                        data = val.get(kk)
                        break
            if data:
                return norm_text(data)
        elif isinstance(val, list) and val:
            for item in val:
                if isinstance(item, dict):
                    data = (
                        item.get("data")
                        or item.get("description")
                        or item.get("text")
                        or item.get("value")
                    )
                    if isinstance(data, dict) and "data" in data:
                        data = data["data"]
                    data = norm_text(data or "")
                    if data:
                        return data
                elif isinstance(item, str) and item.strip():
                    return norm_text(item)
        elif isinstance(val, str) and val.strip():
            return norm_text(val)
    return ""


def extract_all_cves(v: dict, full_text: str) -> List[str]:
    cves = set()
    # structured
    for key in ("cve", "cve_id", "cves"):
        val = v.get(key)
        if isinstance(val, str) and CVE_RE.search(val):
            cves.update(CVE_RE.findall(val))
        elif isinstance(val, dict):
            for subk in ("id", "cve_id", "value", "data"):
                s = val.get(subk)
                if isinstance(s, str) and CVE_RE.search(s):
                    cves.update(CVE_RE.findall(s))
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str) and CVE_RE.search(item):
                    cves.update(CVE_RE.findall(item))
                elif isinstance(item, dict):
                    s = coalesce(
                        item.get("id"), item.get("cve_id"), item.get("data")
                    )
                    if isinstance(s, str) and CVE_RE.search(s):
                        cves.update(CVE_RE.findall(s))
    # from refs & description
    for field in ("references", "refs", "external_references"):
        rv = v.get(field)
        if isinstance(rv, list):
            for x in rv:
                if isinstance(x, str):
                    cves.update(CVE_RE.findall(x))
                elif isinstance(x, dict):
                    for sub in (
                        x.get("url"),
                        x.get("link"),
                        x.get("title"),
                        x.get("name"),
                    ):
                        if isinstance(sub, str):
                            cves.update(CVE_RE.findall(sub))
        elif isinstance(rv, dict):
            for sub in rv.values():
                if isinstance(sub, str):
                    cves.update(CVE_RE.findall(sub))
    if full_text:
        cves.update(CVE_RE.findall(full_text))
    # normalize
    return sorted({c.upper() for c in cves})


def extract_cwes(v: dict, full_text: str) -> List[str]:
    cwes = set()
    for key in ("cwe", "cwes", "weaknesses", "weakness"):
        val = v.get(key)
        if isinstance(val, str):
            cwes.update(CWE_RE.findall(val))
        elif isinstance(val, list):
            for item in val:
                if isinstance(item, str):
                    cwes.update(CWE_RE.findall(item))
                elif isinstance(item, dict):
                    s = coalesce(
                        item.get("id"),
                        item.get("name"),
                        item.get("cwe"),
                        item.get("value"),
                    )
                    if isinstance(s, str):
                        cwes.update(CWE_RE.findall(s))
        elif isinstance(val, dict):
            for sub in val.values():
                if isinstance(sub, str):
                    cwes.update(CWE_RE.findall(sub))
    if full_text:
        cwes.update(CWE_RE.findall(full_text))
    return sorted({c.upper() for c in cwes})


def extract_references(v: dict) -> List[str]:
    urls = []
    for key in ("references", "refs", "external_references", "links"):
        val = v.get(key)
        if isinstance(val, list):
            for x in val:
                if isinstance(x, str) and x.startswith("http"):
                    urls.append(x)
                elif isinstance(x, dict):
                    u = x.get("url") or x.get("link")
                    if isinstance(u, str) and u.startswith("http"):
                        urls.append(u)
        elif isinstance(val, dict):
            for u in val.values():
                if isinstance(u, str) and u.startswith("http"):
                    urls.append(u)
    # also scan description for urls
    desc = extract_description(v)
    urls += re.findall(r"https?://\S+", desc)
    # dedupe preserve order
    seen = set()
    out = []
    for u in urls:
        if u not in seen:
            out.append(u)
            seen.add(u)
    return out


def extract_products(v: dict) -> List[Dict[str, str]]:
    raw = get_data(v, "affected_products", get_data(v, "products", []))
    out: List[Dict[str, str]] = []

    def push(vendor, model, version):
        if not (vendor or model or version):
            return
        out.append(
            {
                "vendor": str(vendor or "").strip(),
                "model": str(model or "").strip(),
                "version": str(version or "").strip(),
            }
        )

    if isinstance(raw, list):
        for item in raw:
            if isinstance(item, dict):
                push(
                    item.get("vendor"),
                    item.get("model") or item.get("product"),
                    item.get("version"),
                )
            elif isinstance(item, str):
                s = item.strip()
                push("", s, "")
    elif isinstance(raw, dict):
        push(
            raw.get("vendor"),
            raw.get("model") or raw.get("product"),
            raw.get("version"),
        )
    return out


def extract_cvss(v: dict) -> Dict[str, Dict[str, Any]]:
    cvss = {"v2": {"base": None, "vector": ""}, "v3": {"base": None, "vector": ""}}
    candidates = [
        ("v2", ("cvss2", "cvss_v2", "cvss_v2_score")),
        ("v3", ("cvss3", "cvss_v3", "cvss_v3_score")),
    ]
    for ver, keys in candidates:
        for k in keys:
            val = v.get(k)
            if isinstance(val, (int, float, str)):
                try:
                    cvss[ver]["base"] = float(str(val).strip())
                except Exception:
                    pass
        # vectors
        for k in (f"{keys[0]}_vector", "cvss_vector", "vector", "cvssVector"):
            val = v.get(k)
            if isinstance(val, str) and "/" in val:
                cvss[ver]["vector"] = val.strip()
    # sometimes nested
    for k in ("cvss", "scoring"):
        obj = v.get(k)
        if isinstance(obj, dict):
            for ver_key in ("v2", "v3", "cvss_v2", "cvss_v3"):
                sub = obj.get(ver_key)
                if isinstance(sub, dict):
                    base = sub.get("baseScore") or sub.get("score")
                    vector = sub.get("vectorString") or sub.get("vector")
                    tgt = "v3" if "3" in ver_key else "v2"
                    if base is not None:
                        try:
                            cvss[tgt]["base"] = float(base)
                        except Exception:
                            pass
                    if isinstance(vector, str):
                        cvss[tgt]["vector"] = vector
    return cvss


def guess_severity(v: dict, desc: str) -> str:
    # direct
    for key in ("severity", "sev", "risk"):
        val = get_data(v, key, "")
        if isinstance(val, str) and val.strip():
            return val.strip().title()
    # heuristics
    m = re.search(r"\b(Critical|High|Medium|Low)\b", desc, flags=re.I)
    return m.group(1).title() if m else ""


def first_nonempty(*vals) -> Optional[str]:
    for v in vals:
        if isinstance(v, str) and v.strip():
            return v.strip()
    return None


# ====== NORMALIZATION ======
def normalize_variot_vuln(v: dict) -> Dict[str, Any]:
    title = norm_text(first_nonempty(get_data(v, "title", ""), v.get("name", "")) or "")
    desc = extract_description(v)
    full_text = "\n".join([x for x in (title, desc) if x])

    cves = extract_all_cves(v, full_text)
    cwes = extract_cwes(v, full_text)
    refs = extract_references(v)
    prods = extract_products(v)
    cvss = extract_cvss(v)
    severity = guess_severity(v, full_text)

    published = first_nonempty(
        v.get("published"), v.get("created"), v.get("disclosed")
    )
    updated = first_nonempty(
        v.get("last_update_date"),
        v.get("last_update"),
        v.get("modified"),
        v.get("updated"),
    )
    published_date = parse_iso_date(published)
    last_update_date = parse_iso_date(updated)

    # derive vendor if uniform in products
    vendor = ""
    if prods and all((p.get("vendor") == prods[0].get("vendor")) for p in prods):
        vendor = prods[0].get("vendor", "")

    normalized = {
        "source": "VARIoT",
        "record_type": "vulnerability",
        "id": v.get("id", ""),
        "title": title,
        "description": desc,
        "cves": cves,
        "cwes": cwes,
        "severity": severity,
        "cvss": cvss,  # {"v2": {"base": float|None, "vector": str}, "v3": {...}}
        "affected_products": prods,  # list of {vendor, model, version}
        "vendor": vendor,
        "references": refs,
        "external_ids": v.get("external_ids")
        or v.get("externalIds")
        or {},
        "published_date": published_date,
        "last_update_date": last_update_date,
        "tags": v.get("tags") or v.get("labels") or v.get("keywords") or [],
        "sources": get_data(v, "sources", []),
    }
    return normalized


# ====== CANONICAL URL + DOC BUILDER FOR RAG ======
def _pick_canonical_url(n: Dict[str, Any]) -> str:
    """
    Prefer:
      1) First HTTP(S) reference URL (vendor / advisory / NVD, etc.).
      2) If missing, derive from first CVE → NVD URL.
      3) Fallback to synthetic variot://<id>.
    """
    refs = n.get("references") or []
    url = ""
    if isinstance(refs, list):
        for r in refs:
            if isinstance(r, str) and r.startswith(("http://", "https://")):
                url = r.strip()
                break

    if not url:
        cves = n.get("cves") or []
        if isinstance(cves, list) and cves:
            first_cve = str(cves[0]).upper().strip()
            if first_cve:
                url = f"https://nvd.nist.gov/vuln/detail/{first_cve}"

    if not url:
        url = f"variot://{n.get('id','')}".strip()

    return url


def variot_record_to_document(
    n: Dict[str, Any], url: Optional[str] = None
) -> Dict[str, str]:
    """
    Build minimal RAG document:

      - url
      - title
      - description:
          * Description (main vulnerability text)
          * Severity
          * CVEs / CWEs
          * CVSS (v3 / v2)
          * Affected products
          * Published / last update dates
          * References / tags / sources
    """
    if not url:
        url = _pick_canonical_url(n)

    title = (n.get("title") or "VARIoT Vulnerability").strip()

    parts: List[str] = []

    # 1) Description
    desc = n.get("description")
    if isinstance(desc, str) and desc.strip():
        parts.append("Description: " + desc.strip())

    # 2) Severity
    sev = n.get("severity")
    if isinstance(sev, str) and sev.strip():
        parts.append(f"Severity: {sev.strip()}")

    # 3) CVEs
    cves = n.get("cves") or []
    if isinstance(cves, list) and cves:
        parts.append("CVEs: " + ", ".join(map(str, cves)))

    # 4) CWEs
    cwes = n.get("cwes") or []
    if isinstance(cwes, list) and cwes:
        parts.append("CWEs: " + ", ".join(map(str, cwes)))

    # 5) CVSS
    cvss = n.get("cvss") or {}
    if isinstance(cvss, dict):
        v3 = cvss.get("v3") or {}
        v2 = cvss.get("v2") or {}

        if isinstance(v3, dict):
            base = v3.get("base")
            vector = v3.get("vector")
            line = []
            if base is not None:
                line.append(f"base score {base}")
            if isinstance(vector, str) and vector.strip():
                line.append(vector.strip())
            if line:
                parts.append("CVSS v3: " + " – ".join(line))

        if isinstance(v2, dict):
            base = v2.get("base")
            vector = v2.get("vector")
            line = []
            if base is not None:
                line.append(f"base score {base}")
            if isinstance(vector, str) and vector.strip():
                line.append(vector.strip())
            if line:
                parts.append("CVSS v2: " + " – ".join(line))

    # 6) Affected products
    prods = n.get("affected_products") or []
    if isinstance(prods, list) and prods:
        lines = []
        for p in prods:
            if not isinstance(p, dict):
                continue
            vendor = p.get("vendor", "") or ""
            model = p.get("model", "") or ""
            version = p.get("version", "") or ""
            label = " ".join(x for x in [vendor, model, version] if x).strip()
            if label:
                lines.append(label)
        if lines:
            parts.append("Affected products:\n" + "\n".join(lines))

    # 7) Dates
    pub = n.get("published_date")
    mod = n.get("last_update_date")
    if pub or mod:
        date_bits = []
        if pub:
            date_bits.append(f"published: {pub}")
        if mod:
            date_bits.append(f"last updated: {mod}")
        parts.append("Timeline: " + " | ".join(date_bits))

    # 8) References
    refs = n.get("references") or []
    if isinstance(refs, list) and refs:
        parts.append("References:\n" + "\n".join(map(str, refs)))

    # 9) Tags / sources
    tags = n.get("tags") or []
    if isinstance(tags, list) and tags:
        parts.append("Tags: " + ", ".join(map(str, tags)))

    sources = n.get("sources") or []
    if isinstance(sources, list) and sources:
        parts.append("Sources: " + ", ".join(map(str, sources)))

    description = "\n".join(parts).strip()

    return {
        "url": url,
        "title": title,
        "description": description or title,
    }


# ====== NORMALIZE FILE (NO DISK OUTPUT) ======
def normalize_variot_file(
    in_json: Optional[str] = None,
) -> List[Dict[str, Any]]:
    """
    Load raw VARIoT JSON and normalize each vulnerability.

    - Uses VARIOT_JSON_PATH by default.
    - Does NOT write JSON/CSV to disk (pure normalization).
    - If input file is missing, log and return [] so scraper is skipped.
    """
    path = in_json or VARIOT_JSON_PATH
    logger.info("[VARIOT] Normalizing VARIoT file from %s", path)

    if not os.path.exists(path):
        logger.warning(
            "[VARIOT] Input not found at %s. Skipping VARIoT scraper.", path
        )
        return []

    with open(path, "r", encoding="utf-8") as f:
        raw = json.load(f)

    if not isinstance(raw, list):
        logger.error(
            "[VARIOT] Expected a JSON array of vulnerabilities in %s", path
        )
        return []

    normalized: List[Dict[str, Any]] = []
    for rec in raw:
        if isinstance(rec, dict):
            n = normalize_variot_vuln(rec)
            normalized.append(n)

    logger.info("[VARIOT] Normalized %d VARIoT vulnerabilities.", len(normalized))
    return normalized


# ====== PUBLIC ENTRYPOINT FOR YOUR RAG PIPELINE ======
def scrape_variot(
    check_qdrant: bool = True,
    in_json: Optional[str] = None,
) -> List[Dict[str, str]]:
    """
    High-level VARIoT ingestion used by your cron / scraping graph.

      - Loads and normalizes VARIoT vulnerabilities from VARIOT_JSON_PATH (or `in_json`).
      - Picks a canonical URL per record (references → CVE → synthetic).
      - If check_qdrant=True: skips records whose canonical URL is already
        present in Qdrant via url_already_ingested(url).
      - Returns list of minimal RAG docs: {url, title, description}.

    ⚠️ This module ONLY reads + normalizes; it does not write CSV/JSON or
    insert into Qdrant (just uses url_already_ingested for dedupe).
    """
    normalized = normalize_variot_file(in_json=in_json)
    if not normalized:
        logger.info("[VARIOT] No normalized records found; returning empty docs list.")
        return []

    docs: List[Dict[str, str]] = []
    for n in normalized:
        url = _pick_canonical_url(n)

        # ✅ Qdrant check BEFORE adding this record as a doc
        if check_qdrant and url and url_already_ingested(url):
            logger.info("[VARIOT] Skipping already-ingested URL: %s", url)
            continue

        doc = variot_record_to_document(n, url=url)
        docs.append(doc)

    logger.info("[VARIOT] Built %d vulnerability documents.", len(docs))
    return docs


# ====== CLI / LOCAL TEST ENTRYPOINT ======
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    normalized = normalize_variot_file()
    print(f"Normalized records: {len(normalized)}")

    docs = scrape_variot(check_qdrant=False)
    print(f"Sample docs: {len(docs)}")
    if docs:
        print("\n--- SAMPLE DOC ---")
        print(json.dumps(docs[min(20, len(docs) - 1)], indent=2, ensure_ascii=False))
