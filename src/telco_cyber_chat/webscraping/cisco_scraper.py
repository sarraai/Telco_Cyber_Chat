# pip install requests beautifulsoup4 lxml

import sys
import time
import json
import re
import logging
import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timezone
from urllib.parse import urlparse
from bs4 import BeautifulSoup, Tag
from pathlib import Path

from telco_cyber_chat.webscraping.scrape_core import url_already_ingested

logger = logging.getLogger(__name__)

# ===== OAuth & API =====
CLIENT_ID = "ucprtmc76uqkhda36q26z8f9"      # <- your Cisco client_id
CLIENT_SECRET = "gNjqQMquNyejxerJ2SYFPdqX"  # <- your Cisco client_secret
TOKEN_URL = "https://id.cisco.com/oauth2/default/v1/token"
BASE_URL  = "https://apix.cisco.com/security/advisories/v2"

# ===== Range / defaults =====
START_YEAR = 2024
END_YEAR   = datetime.now(timezone.utc).year

# ===== Output (debug / exports) =====
OUT_JSON = f"cisco_advisories_{START_YEAR}_{END_YEAR}.json"

# ===== Networking =====
TIMEOUT   = 60
PAUSE_SEC = 0.25
HTML_HEADERS = {
    "User-Agent": "Mozilla/5.0",
    "Accept-Language": "en-US,en;q=0.9",
}

# Canonical pages
MIRROR_BASE = "https://www.cisco.com/c/en/us/support/docs/csa/"
CANON_BASE  = "https://sec.cloudapps.cisco.com/security/center/content/CiscoSecurityAdvisory/"

# ===== Utils =====
def _clean(s: Optional[str]) -> str:
    return " ".join((s or "").replace("\xa0", " ").split())


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        s2 = s.strip()
        if "T" in s2:
            # ISO with time (possibly with 'Z')
            return datetime.fromisoformat(s2.replace("Z", "+00:00"))
        # Date-only → assume midnight UTC
        return datetime.fromisoformat(s2 + "T00:00:00+00:00")
    except Exception:
        return None


def _is_valid_http_url(u: Optional[str]) -> bool:
    if not u:
        return False
    up = u.strip().upper()
    if up in {"NA", "N/A", "-", "NONE"}:
        return False
    try:
        p = urlparse(u)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False

# ===== HTTP helpers =====
def get_token(client_id: str, client_secret: str) -> str:
    r = requests.post(
        TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=TIMEOUT,
    )
    if not r.ok:
        raise SystemExit(f"Token request failed: HTTP {r.status_code}\n{r.text}")
    tok = r.json().get("access_token")
    if not tok:
        raise SystemExit("No access_token in token response")
    return tok


def get_json(endpoint: str, token: str, params: Optional[Dict] = None) -> Dict:
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    r = requests.get(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        },
        params=params or {},
        timeout=TIMEOUT,
    )
    if not r.ok:
        raise RuntimeError(f"GET {url} failed: HTTP {r.status_code}\n{r.text}")
    return r.json()


def get_text(url: str, token: str) -> Optional[str]:
    try:
        r = requests.get(
            url,
            headers={"Accept": "*/*", "Authorization": f"Bearer {token}"},
            timeout=TIMEOUT,
        )
        return r.text if r.ok else None
    except Exception:
        return None

# ===== Fetch advisory =====
def fetch_advisory_by_id(token: str, advisory_id: str) -> Optional[Dict]:
    """
    Try direct /advisory/{id}, then fall back to scanning year endpoints.
    """
    try:
        data = get_json(f"advisory/{advisory_id}", token)
        if isinstance(data, dict) and "advisory" in data:
            return data["advisory"]
        if isinstance(data, dict) and (data.get("advisoryId") or data.get("id")):
            return data
    except Exception:
        pass

    # Fallback: scan recent years
    for y in range(END_YEAR, START_YEAR - 1, -1):
        try:
            payload = get_json(f"year/{y}", token)
            for adv in payload.get("advisories", []) or []:
                aid = adv.get("advisoryId") or adv.get("id") or ""
                if aid == advisory_id:
                    return adv
        except Exception:
            continue
        time.sleep(PAUSE_SEC)
    return None


def fetch_latest_advisory(token: str, start_year: int, end_year: int) -> Optional[Dict]:
    """
    Find the most recently updated advisory between start_year and end_year.
    """
    best, best_dt = None, None
    for y in range(end_year, start_year - 1, -1):
        try:
            items = get_json(f"year/{y}", token).get("advisories", []) or []
        except Exception:
            continue
        for adv in items:
            fp = adv.get("firstPublished")
            lu = adv.get("lastUpdated") or adv.get("lastUpdatedDate")
            dt = _parse_dt(lu) or _parse_dt(fp)
            if dt and (best_dt is None or dt > best_dt):
                best_dt, best = dt, adv
        time.sleep(PAUSE_SEC)
    return best

# ===== CVRF parsing (ONLY summary + not-vulnerable products) =====
def _lname(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _iter_by_name(node: ET.Element, name: str):
    for el in node.iter():
        if _lname(el.tag) == name:
            yield el


def _get_all_text(el: Optional[ET.Element]) -> str:
    return "" if el is None else _clean("".join(el.itertext()))


def parse_cvrf(xml_text: str) -> Dict[str, Any]:
    """
    Parse Cisco CVRF XML and return ONLY:
      - summary: str
      - products_confirmed_not_vulnerable: List[str]
    """
    out: Dict[str, Any] = {
        "summary": "",
        "products_confirmed_not_vulnerable": [],
    }
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return out

    # ---- Notes → Summary ----
    notes_by_title: Dict[str, str] = {}
    for note in _iter_by_name(root, "Note"):
        title = _clean(note.attrib.get("Title", ""))
        if title:
            notes_by_title[title.lower()] = _get_all_text(note)

    def first_note(*contains: str) -> str:
        for k, v in notes_by_title.items():
            if all(x.lower() in k for x in contains):
                return v
        return ""

    out["summary"] = first_note("summary")

    # ---- Products Confirmed Not Vulnerable ----
    prod_map: Dict[str, str] = {}
    for fp in _iter_by_name(root, "FullProductName"):
        pid = (
            fp.attrib.get("ProductID")
            or fp.attrib.get("ProductIDRef")
            or fp.attrib.get("ProductIDref")
            or ""
        )
        if pid:
            prod_map[pid] = _get_all_text(fp)

    not_vuln_ids = set()
    for pstatus in _iter_by_name(root, "ProductStatuses"):
        for status in _iter_by_name(pstatus, "Status"):
            stype = _clean(status.attrib.get("Type", ""))
            ids = [
                _clean(el.attrib.get("ProductID", "")) or _clean(el.text or "")
                for el in _iter_by_name(status, "ProductID")
            ]
            if "known not affected" in stype.lower() or "not vulnerable" in stype.lower():
                not_vuln_ids.update(ids)

    not_vuln_list = sorted(
        {prod_map.get(pid, pid) for pid in not_vuln_ids if pid}
    )
    out["products_confirmed_not_vulnerable"] = not_vuln_list

    return out

# ===== HTML helpers (fallback for affected / not-vulnerable) =====
def _txt(node: Tag) -> str:
    return _clean(node.get_text(" ", strip=True)) if isinstance(node, Tag) else ""


def html_affected_products_fallback(
    advisory_id: Optional[str],
    publication_url: Optional[str],
) -> Dict[str, str]:
    """
    HTML fallback to extract:
      - affected_products: free text
      - vulnerable_products: free text (unused here)
      - products_confirmed_not_vulnerable: free text (we'll split it into a list)
    """
    out = {
        "affected_products": "",
        "vulnerable_products": "",
        "products_confirmed_not_vulnerable": "",
    }

    candidates = [
        f"{MIRROR_BASE}{advisory_id}.html" if advisory_id else None,
        f"{CANON_BASE}{advisory_id}" if advisory_id else None,
        publication_url or None,
    ]

    def collect(container: Tag) -> str:
        parts = [_txt(tag) for tag in container.find_all(["p", "li"]) if _txt(tag)]
        # de-dup while preserving order
        return "\n\n".join(dict.fromkeys(parts))

    for u in filter(None, candidates):
        try:
            r = requests.get(u, headers=HTML_HEADERS, timeout=40)
            if not r.ok:
                continue
            soup = BeautifulSoup(r.text, "html.parser")

            # Direct fields by id
            aff = soup.find(id="affectfield")
            if aff:
                out["affected_products"] = collect(aff)
            vp = soup.find(id="vulnerableproducts")
            if vp:
                out["vulnerable_products"] = collect(vp)
            nv = soup.find(id="productsconfirmednotvulnerable")
            if nv:
                out["products_confirmed_not_vulnerable"] = collect(nv)

            # Heuristic heading-based "Affected Products"
            if not out["affected_products"]:
                heading = None
                for h in soup.find_all(["h2", "h3", "h4"]):
                    if "affected products" in _txt(h).lower():
                        heading = h
                        break
                if heading:
                    buf = soup.new_tag("div")
                    for sib in heading.next_siblings:
                        if isinstance(sib, Tag) and sib.name in {"h2", "h3"}:
                            break
                        if isinstance(sib, Tag):
                            buf.append(sib)
                    out["affected_products"] = collect(buf)

            if any(out.values()):
                return out
        except Exception:
            continue

    return out

# ===== Transform one advisory → minimal RAG doc =====
def transform_advisory(adv: Dict, token: str) -> Dict[str, str]:
    """
    Transform 1 Cisco advisory into minimal RAG document:

      {
        "title": "...",
        "url": "...",
        "description": "Summary + Affected Products + Products Confirmed Not Vulnerable"
      }
    """
    aid = adv.get("advisoryId") or adv.get("id") or ""
    raw_title = adv.get("title") or adv.get("advisoryTitle") or ""
    title = _clean(raw_title) or (f"Cisco Security Advisory {aid}" if aid else "Cisco Security Advisory")

    # Base URL (canonical)
    url = f"{CANON_BASE}{aid}" if aid else (adv.get("publicationUrl") or "")

    # ---- CVRF: summary + not-vulnerable list ----
    summary = ""
    not_vuln_list: List[str] = []

    cvrf_url = adv.get("cvrfUrl") or adv.get("cvrfURL")
    if _is_valid_http_url(cvrf_url):
        xml = get_text(cvrf_url, token)
        if xml:
            cvrf = parse_cvrf(xml)
            summary = (cvrf.get("summary") or "").strip()
            not_vuln_list = cvrf.get("products_confirmed_not_vulnerable", []) or []
            time.sleep(PAUSE_SEC)

    # ---- HTML fallback: affected products + not-vulnerable (if missing) ----
    affected_products_text = ""
    html_aff = html_affected_products_fallback(aid, adv.get("publicationUrl"))

    if html_aff.get("affected_products"):
        affected_products_text = html_aff["affected_products"]

    if not not_vuln_list and html_aff.get("products_confirmed_not_vulnerable"):
        raw_nv = html_aff["products_confirmed_not_vulnerable"]
        not_vuln_list = [line for line in raw_nv.split("\n\n") if line.strip()]

    # ---- Build description string ----
    parts: List[str] = []

    if summary:
        parts.append("Summary: " + summary)

    if affected_products_text:
        parts.append("Affected products:\n" + affected_products_text)

    if not_vuln_list:
        nv_lines = "\n".join(f"- {p}" for p in not_vuln_list)
        parts.append("Products confirmed not vulnerable:\n" + nv_lines)

    description = "\n\n".join(parts).strip() or title

    return {
        "title": title,
        "url": url,
        "description": description,
    }

# ====== Bulk fetch from 2024 onward → minimal docs ======
def fetch_all_advisories(
    start_year: int,
    end_year: int,
    check_qdrant: bool = True,
) -> List[Dict[str, str]]:
    """
    Iterates year endpoints and transforms each advisory with MINIMAL enrichment.

    Returns list of documents (sorted newest → oldest):
      {
        "title": "...",
        "url": "...",
        "description": "Summary + Affected Products + Products Confirmed Not Vulnerable"
      }

    If check_qdrant=True, skip advisories whose canonical URL
    is already present in Qdrant (via url_already_ingested).
    """
    token = get_token(CLIENT_ID, CLIENT_SECRET)
    seen_ids = set()
    items_with_dates: List[Tuple[datetime, Dict[str, str]]] = []

    for y in range(end_year, start_year - 1, -1):
        try:
            payload = get_json(f"year/{y}", token)
            year_list = payload.get("advisories", []) or []
        except Exception as e:
            logger.warning("[CISCO] Fetch year %s failed: %s", y, e)
            continue

        for adv in year_list:
            aid = adv.get("advisoryId") or adv.get("id")
            if not aid or aid in seen_ids:
                continue

            # Canonical URL used for dedupe
            url = f"{CANON_BASE}{aid}" if aid else (adv.get("publicationUrl") or "")

            if check_qdrant and url and url_already_ingested(url):
                logger.info("[CISCO] Skipping already-ingested URL: %s", url)
                continue

            # Choose sort date using lastUpdated / firstPublished
            dt = _parse_dt(adv.get("lastUpdated") or adv.get("lastUpdatedDate")) \
                 or _parse_dt(adv.get("firstPublished"))
            if dt is None:
                dt = datetime.fromtimestamp(0, tz=timezone.utc)

            try:
                doc = transform_advisory(adv, token)
                items_with_dates.append((dt, doc))
                seen_ids.add(aid)
            except Exception as e:
                logger.warning("[CISCO] Transform failed for %s: %s", aid, e)

            time.sleep(PAUSE_SEC)

        time.sleep(PAUSE_SEC)

    # sort newest first
    items_with_dates.sort(key=lambda t: t[0], reverse=True)
    docs = [doc for _, doc in items_with_dates]
    return docs


def save_json(path: str, data: List[Dict[str, str]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved {len(data)} advisories to {path}")

# ===== COLAB / debug helpers =====
def fetch_one_advisory_colab(advisory_id: Optional[str] = None) -> Dict[str, str]:
    """
    Fetch a single advisory (by ID or latest) and print the minimal document.
    """
    token = get_token(CLIENT_ID, CLIENT_SECRET)
    adv = (
        fetch_advisory_by_id(token, advisory_id)
        if advisory_id
        else fetch_latest_advisory(token, START_YEAR, END_YEAR)
    )
    if not adv:
        raise RuntimeError("Advisory not found")

    doc = transform_advisory(adv, token)
    print(json.dumps(doc, indent=2, ensure_ascii=False))
    return doc


def fetch_all_advisories_colab():
    """
    Fetch ALL advisories from START_YEAR → END_YEAR and save minimal docs to JSON.
    (Used for manual debugging / snapshots; no Qdrant dedupe.)
    """
    docs = fetch_all_advisories(START_YEAR, END_YEAR, check_qdrant=False)
    save_json(OUT_JSON, docs)

    if docs:
        print("\n[Preview of first item]")
        print(json.dumps(docs[0], indent=2, ensure_ascii=False))

# ===== PUBLIC ENTRYPOINT FOR YOUR PIPELINE =====
def scrape_cisco(
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    check_qdrant: bool = True,
) -> List[Dict[str, str]]:
    """
    High-level scraper used by your cron / scraping graph.

      - Fetches Cisco advisories from Cisco API (CVRF + HTML).
      - If check_qdrant=True: skips canonical URLs already in Qdrant
        BEFORE heavy CVRF/HTML calls, via url_already_ingested().
      - Returns list of minimal RAG documents: {url, title, description}
    """
    docs = fetch_all_advisories(
        start_year=start_year,
        end_year=end_year,
        check_qdrant=check_qdrant,
    )
    logger.info("[CISCO] Scraped %d advisories (documents).", len(docs))
    return docs

# === Standalone CLI / local test ===
if __name__ == "__main__":
    # For RAG testing: minimal docs, no Qdrant dedupe in local tests
    docs = scrape_cisco(check_qdrant=False)
    print(f"Sample docs: {len(docs)}")
    if docs:
        print(json.dumps(docs[0], indent=2, ensure_ascii=False))
