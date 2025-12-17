# pip install requests beautifulsoup4 lxml qdrant-client llama-index-core

import os
import time
import json
import re
import logging
import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional, Any, Tuple, Set
from datetime import datetime, timezone
from urllib.parse import urlparse
from bs4 import BeautifulSoup, Tag
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue
from llama_index.core.schema import TextNode

logger = logging.getLogger(__name__)

# ===== OAuth & API =====
TOKEN_URL = "https://id.cisco.com/oauth2/default/v1/token"
BASE_URL  = "https://apix.cisco.com/security/advisories/v2"

# ===== Range =====
START_YEAR = 2024
END_YEAR   = datetime.now(timezone.utc).year

# ===== Output =====
OUT_JSON = f"cisco_advisories_{START_YEAR}_{END_YEAR}.json"

# ===== Networking =====
TIMEOUT   = 60
PAUSE_SEC = 0.25
HTML_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}

# Canonical pages
MIRROR_BASE = "https://www.cisco.com/c/en/us/support/docs/csa/"
CANON_BASE  = "https://sec.cloudapps.cisco.com/security/center/content/CiscoSecurityAdvisory/"

# ✅ Qdrant payload keys/values (must match what you store in Qdrant)
VENDOR_VALUE = "cisco"   # <-- vendor filter value in Qdrant
VENDOR_KEY   = "vendor"
URL_KEY      = "url"


# =====================
# Config loaders
# =====================
def get_qdrant_client() -> Tuple[QdrantClient, str]:
    qurl = (os.getenv("QDRANT_URL") or "").strip()
    qkey = (os.getenv("QDRANT_API_KEY") or "").strip()
    collection = (os.getenv("QDRANT_COLLECTION") or "Telco_CyberChat").strip()
    if not qurl or not qkey:
        raise RuntimeError("Missing QDRANT_URL / QDRANT_API_KEY in env.")
    return QdrantClient(url=qurl, api_key=qkey), collection


def get_cisco_creds() -> Tuple[str, str]:
    cid = (os.getenv("CISCO_CLIENT_ID") or "").strip()
    csec = (os.getenv("CISCO_CLIENT_SECRET") or "").strip()
    if not cid or not csec:
        raise RuntimeError("Missing CISCO_CLIENT_ID / CISCO_CLIENT_SECRET in env.")
    return cid, csec


# =====================
# Qdrant: load existing URLs once (vendor=cisco)
# =====================
def fetch_existing_vendor_urls(
    client: QdrantClient,
    collection_name: str,
    vendor_value: str,
    page_size: int = 256,
) -> Set[str]:
    """
    Scroll Qdrant with filter vendor=<vendor_value> and collect payload[url].
    This is the "filter-first" optimization to avoid per-advisory Qdrant calls.
    """
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
            with_payload=True,   # we need url from payload
            with_vectors=False,
        )

        for p in points:
            payload = p.payload or {}
            u = payload.get(URL_KEY)
            if isinstance(u, str) and u.strip():
                existing.add(u.strip())

        if offset is None:
            break

    return existing


# =====================
# Utils
# =====================
def _clean(s: Optional[str]) -> str:
    return " ".join((s or "").replace("\xa0", " ").split())


def _iso_date_only(s: Optional[str]) -> str:
    if not s:
        return ""
    try:
        s2 = s.strip()
        if "T" in s2:
            return datetime.fromisoformat(s2.replace("Z", "+00:00")).date().isoformat()
        if len(s2) >= 10 and s2[4] == "-" and s2[7] == "-":
            return s2[:10]
    except Exception:
        pass
    return s or ""


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        if "T" in s:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        return datetime.fromisoformat(s + "T00:00:00+00:00")
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


def _sort_dt_from_adv(adv: Dict) -> datetime:
    lu = adv.get("lastUpdated") or adv.get("lastUpdatedDate")
    fp = adv.get("firstPublished")
    dt = _parse_dt(lu) or _parse_dt(fp)
    return dt if dt is not None else datetime.fromtimestamp(0, tz=timezone.utc)


# =====================
# HTTP helpers
# =====================
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
        raise RuntimeError(f"Token request failed: HTTP {r.status_code}\n{r.text}")
    tok = r.json().get("access_token")
    if not tok:
        raise RuntimeError("No access_token in token response")
    return tok


def get_json(endpoint: str, token: str, params: Optional[Dict] = None) -> Dict:
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    r = requests.get(
        url,
        headers={"Accept": "application/json", "Authorization": f"Bearer {token}"},
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


# =====================
# CVRF parsing (your "new" fields)
# =====================
def _lname(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _iter_by_name(node: ET.Element, name: str):
    for el in node.iter():
        if _lname(el.tag) == name:
            yield el


def _get_all_text(el: Optional[ET.Element]) -> str:
    return "" if el is None else _clean("".join(el.itertext()))


def parse_cvrf(xml_text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "description": "",
        "source": "",
        "workarounds": "",
        "revision_history": [],
        "affected_products": "",
        "vulnerable_products": [],
        "products_confirmed_not_vulnerable": [],
        "fixed_software": [],
        "cvss_max_base": None,
    }
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return out

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

    out["description"] = first_note("summary")
    out["source"] = first_note("source", "credit", "acknowledg")

    wa = [v for k, v in notes_by_title.items() if "workaround" in k or "mitigation" in k]
    for rem in _iter_by_name(root, "Remediation"):
        rtype = _clean(rem.attrib.get("Type", ""))
        if "workaround" in rtype.lower() or "mitigation" in rtype.lower():
            wa.append(_get_all_text(rem))
    if wa:
        out["workarounds"] = "\n\n".join(dict.fromkeys([w.strip() for w in wa if w.strip()]))

    for rh in _iter_by_name(root, "Revision"):
        num = _get_all_text(next((c for c in rh if _lname(c.tag) == "Number"), None))
        date = _get_all_text(next((c for c in rh if _lname(c.tag) == "Date"), None))
        desc = _get_all_text(next((c for c in rh if _lname(c.tag) == "Description"), None))
        out["revision_history"].append({"version": num, "date": _iso_date_only(date), "description": desc})

    prod_map: Dict[str, str] = {}
    for fp in _iter_by_name(root, "FullProductName"):
        pid = fp.attrib.get("ProductID") or fp.attrib.get("ProductIDRef") or fp.attrib.get("ProductIDref") or ""
        if pid:
            prod_map[pid] = _get_all_text(fp)

    vuln_ids, not_vuln_ids = set(), set()
    for pstatus in _iter_by_name(root, "ProductStatuses"):
        for status in _iter_by_name(pstatus, "Status"):
            stype = _clean(status.attrib.get("Type", ""))
            ids = [
                _clean(el.attrib.get("ProductID", "")) or _clean(el.text or "")
                for el in _iter_by_name(status, "ProductID")
            ]
            if "known affected" in stype.lower() or "vulnerable" in stype.lower():
                vuln_ids.update(ids)
            if "known not affected" in stype.lower() or "not vulnerable" in stype.lower():
                not_vuln_ids.update(ids)

    out["vulnerable_products"] = sorted({prod_map.get(pid, pid) for pid in vuln_ids if pid})
    out["products_confirmed_not_vulnerable"] = sorted({prod_map.get(pid, pid) for pid in not_vuln_ids if pid})

    cvss_vals = []
    for b in _iter_by_name(root, "BaseScore"):
        try:
            cvss_vals.append(float(_get_all_text(b)))
        except Exception:
            pass
    if cvss_vals:
        out["cvss_max_base"] = max(cvss_vals)

    return out


# =====================
# HTML fallbacks (minimal set; keep expanding if you want)
# =====================
def _txt(node: Tag) -> str:
    return _clean(node.get_text(" ", strip=True)) if isinstance(node, Tag) else ""


def html_workarounds_fallback(advisory_id: str, publication_url: Optional[str]) -> str:
    candidates = [
        f"{MIRROR_BASE}{advisory_id}.html" if advisory_id else None,
        f"{CANON_BASE}{advisory_id}" if advisory_id else None,
        publication_url or None,
    ]
    for u in filter(None, candidates):
        try:
            r = requests.get(u, headers=HTML_HEADERS, timeout=40)
            if not r.ok:
                continue
            soup = BeautifulSoup(r.text, "lxml")

            for anchor_id in ("workaroundsfield", "workaroundfield", "mitigationfield"):
                cont = soup.find(id=anchor_id)
                if cont:
                    parts = [_txt(tag) for tag in cont.find_all(["p", "li"]) if _txt(tag)]
                    if parts:
                        return "\n\n".join(dict.fromkeys(parts))
        except Exception:
            continue
    return ""


def html_affected_products_fallback(advisory_id: str, publication_url: Optional[str]) -> Dict[str, str]:
    out = {"affected_products": "", "vulnerable_products": "", "products_confirmed_not_vulnerable": ""}
    candidates = [
        f"{MIRROR_BASE}{advisory_id}.html" if advisory_id else None,
        f"{CANON_BASE}{advisory_id}" if advisory_id else None,
        publication_url or None,
    ]

    def collect(container: Tag) -> str:
        parts = [_txt(tag) for tag in container.find_all(["p", "li"]) if _txt(tag)]
        return "\n\n".join(dict.fromkeys(parts))

    for u in filter(None, candidates):
        try:
            r = requests.get(u, headers=HTML_HEADERS, timeout=40)
            if not r.ok:
                continue
            soup = BeautifulSoup(r.text, "lxml")

            aff = soup.find(id="affectfield")
            if aff:
                out["affected_products"] = collect(aff)
            vp = soup.find(id="vulnerableproducts")
            if vp:
                out["vulnerable_products"] = collect(vp)
            nv = soup.find(id="productsconfirmednotvulnerable")
            if nv:
                out["products_confirmed_not_vulnerable"] = collect(nv)

            if any(out.values()):
                return out
        except Exception:
            continue
    return out


# =====================
# ✅ NEW: Convert dict to TextNode with formatted text
# =====================
def create_text_node_from_advisory(adv_dict: Dict[str, Any]) -> TextNode:
    """
    Creates a TextNode with all fields formatted in the text part,
    and only the URL in metadata.
    """
    # Extract fields
    vendor = adv_dict.get("vendor", "")
    url = adv_dict.get("url", "")
    title = adv_dict.get("title", "")
    advisory_id = adv_dict.get("advisory_id", "")
    description = adv_dict.get("description", "")
    source = adv_dict.get("source", "")
    workarounds = adv_dict.get("workarounds", "")
    affected_products = adv_dict.get("affected_products", "")
    vulnerable_products = adv_dict.get("vulnerable_products", [])
    products_not_vulnerable = adv_dict.get("products_confirmed_not_vulnerable", [])
    revision_history = adv_dict.get("revision_history", [])
    
    # Build formatted text content
    text_parts = []
    
    # Header
    text_parts.append(f"VENDOR: {vendor}")
    text_parts.append(f"ADVISORY ID: {advisory_id}")
    text_parts.append(f"TITLE: {title}")
    text_parts.append(f"URL: {url}")
    text_parts.append("")
    
    # Description
    if description:
        text_parts.append("DESCRIPTION:")
        text_parts.append(description)
        text_parts.append("")
    
    # Source/Credits
    if source:
        text_parts.append("SOURCE/CREDITS:")
        text_parts.append(source)
        text_parts.append("")
    
    # Workarounds
    if workarounds:
        text_parts.append("WORKAROUNDS/MITIGATIONS:")
        text_parts.append(workarounds)
        text_parts.append("")
    
    # Affected Products
    if affected_products:
        text_parts.append("AFFECTED PRODUCTS:")
        text_parts.append(affected_products)
        text_parts.append("")
    
    # Vulnerable Products
    if vulnerable_products:
        text_parts.append("VULNERABLE PRODUCTS:")
        for i, prod in enumerate(vulnerable_products, 1):
            text_parts.append(f"  {i}. {prod}")
        text_parts.append("")
    
    # Products Not Vulnerable
    if products_not_vulnerable:
        text_parts.append("PRODUCTS CONFIRMED NOT VULNERABLE:")
        for i, prod in enumerate(products_not_vulnerable, 1):
            text_parts.append(f"  {i}. {prod}")
        text_parts.append("")
    
    # Revision History
    if revision_history:
        text_parts.append("REVISION HISTORY:")
        for rev in revision_history:
            version = rev.get("version", "")
            date = rev.get("date", "")
            desc = rev.get("description", "")
            text_parts.append(f"  Version {version} ({date}): {desc}")
        text_parts.append("")
    
    # Combine all parts
    text_content = "\n".join(text_parts).strip()
    
    # Create TextNode with only URL in metadata
    node = TextNode(
        text=text_content,
        metadata={
            "url": url,
            "vendor": vendor  # Keep vendor for filtering
        }
    )
    
    return node


# =====================
# Heavy transform (ONLY called if url NOT in Qdrant)
# =====================
def transform_advisory(adv: Dict[str, Any], token: str, canonical_url: str) -> Dict[str, Any]:
    aid = adv.get("advisoryId") or adv.get("id")
    title = _clean(adv.get("title") or adv.get("advisoryTitle") or "") or f"Cisco Security Advisory {aid}"

    cvrf_url = adv.get("cvrfUrl") or adv.get("cvrfURL")
    cvrf: Dict[str, Any] = {}
    if _is_valid_http_url(cvrf_url):
        xml = get_text(cvrf_url, token)
        if xml:
            cvrf = parse_cvrf(xml)
            time.sleep(PAUSE_SEC)

    workarounds = (cvrf.get("workarounds") or "").strip()
    if not workarounds:
        workarounds = html_workarounds_fallback(str(aid), adv.get("publicationUrl"))

    affected_products_text = (cvrf.get("affected_products") or "").strip()
    vulnerable_list = cvrf.get("vulnerable_products", []) or []
    not_vuln_list = cvrf.get("products_confirmed_not_vulnerable", []) or []
    if not affected_products_text or (not vulnerable_list and not not_vuln_list):
        html_aff = html_affected_products_fallback(str(aid), adv.get("publicationUrl"))
        if html_aff.get("affected_products") and not affected_products_text:
            affected_products_text = html_aff["affected_products"]
        if not vulnerable_list and html_aff.get("vulnerable_products"):
            vulnerable_list = [line for line in html_aff["vulnerable_products"].split("\n\n") if line.strip()]
        raw_nv = html_aff.get("products_confirmed_not_vulnerable") or ""
        if not not_vuln_list and raw_nv:
            not_vuln_list = [line for line in raw_nv.split("\n\n") if line.strip()]

    # ✅ output keys aligned with your Qdrant filtering
    return {
        "vendor": VENDOR_VALUE,
        "url": canonical_url,

        "title": title,
        "advisory_id": aid,
        "description": cvrf.get("description", ""),
        "source": cvrf.get("source", ""),
        "workarounds": workarounds,
        "affected_products": affected_products_text,
        "vulnerable_products": vulnerable_list,
        "products_confirmed_not_vulnerable": not_vuln_list,
        "revision_history": cvrf.get("revision_history", []),
    }


# =====================
# Bulk fetch: CHECK QDRANT (vendor+url) BEFORE SCRAPING
# =====================
def fetch_all_advisories(
    start_year: int,
    end_year: int,
    check_qdrant: bool = True,
) -> List[Dict[str, Any]]:
    qclient, collection = get_qdrant_client()

    existing_urls: Set[str] = set()
    if check_qdrant:
        existing_urls = fetch_existing_vendor_urls(qclient, collection, VENDOR_VALUE)
        logger.info("[CISCO] Loaded %d existing URLs from Qdrant for vendor=%s", len(existing_urls), VENDOR_VALUE)

    cid, csec = get_cisco_creds()
    token = get_token(cid, csec)

    seen_ids = set()
    items_with_dt: List[Tuple[datetime, Dict[str, Any]]] = []

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

            # ✅ canonical URL computed BEFORE scraping
            canonical_url = f"{CANON_BASE}{aid}"

            # ✅ Qdrant check (vendor=cisco + url=canonical_url) BEFORE heavy work
            if check_qdrant and canonical_url in existing_urls:
                seen_ids.add(aid)
                continue

            try:
                item = transform_advisory(adv, token, canonical_url)
                sort_dt = _sort_dt_from_adv(adv)
                items_with_dt.append((sort_dt, item))
                seen_ids.add(aid)
            except Exception as e:
                logger.warning("[CISCO] Transform failed for %s: %s", aid, e)

            time.sleep(PAUSE_SEC)

        time.sleep(PAUSE_SEC)

    items_with_dt.sort(key=lambda x: x[0], reverse=True)
    return [it for _, it in items_with_dt]


def scrape_cisco(
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    check_qdrant: bool = True,
) -> List[TextNode]:
    """
    ✅ UPDATED: Returns List[TextNode] instead of List[Dict]
    """
    docs = fetch_all_advisories(start_year, end_year, check_qdrant=check_qdrant)
    logger.info("[CISCO] New advisories fetched: %d", len(docs))
    
    # Convert dicts to TextNodes
    text_nodes = [create_text_node_from_advisory(doc) for doc in docs]
    logger.info("[CISCO] TextNodes created: %d", len(text_nodes))
    
    return text_nodes


def save_json(path: str, data: List[Dict[str, Any]]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved {len(data)} advisories to {path}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    # Get TextNodes
    text_nodes = scrape_cisco(check_qdrant=True)
    
    # Save the raw dicts for reference
    raw_docs = fetch_all_advisories(START_YEAR, END_YEAR, check_qdrant=True)
    save_json(OUT_JSON, raw_docs)
    
    # Print example of first TextNode
    if text_nodes:
        print("\n" + "="*80)
        print("EXAMPLE TEXTNODE OUTPUT:")
        print("="*80)
        print(f"\nMetadata: {text_nodes[0].metadata}")
        print(f"\nText Content (first 1000 chars):\n{text_nodes[0].text[:1000]}...")
        print("\n" + "="*80)
