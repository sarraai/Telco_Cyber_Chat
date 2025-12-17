# telco_cyber_chat/webscraping/scrape_orchestrator.py

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

from llama_index.core.schema import TextNode

from .scrape_core import canonical_url, normalize_vendor

# ✅ UPDATED imports (node-returning scrapers)
from .nokia_scraper import scrape_nokia_nodes
from .huawei_scraper import scrape_huawei_nodes

# Keep these imports if they still return records (list[dict]) in your repo
from .ericsson_scraper import scrape_ericsson
from .cisco_scraper import scrape_cisco
from .variot_scraper import scrape_variot
from .mitre_attack_scraper import scrape_mitre_mobile


VendorName = Literal["nokia", "ericsson", "huawei", "cisco", "variot", "mitre_mobile"]


# -----------------------------------------------------------------------------
# Generic Record -> TextNode (RULE: text=all fields except url, metadata=only url)
# -----------------------------------------------------------------------------
def _dump_json(v: Any) -> str:
    try:
        return json.dumps(v, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        return str(v)

def _record_to_text(rec: Dict[str, Any]) -> str:
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

def _stable_id(vendor: str, url: str, text: str) -> str:
    """
    Deterministic id so your pipeline doesn't *require* the node to already have one.
    - If url exists: hash(vendor|url)
    - Else: hash(vendor|sha256(text))
    """
    vendor = normalize_vendor(vendor or "unknown")
    url = canonical_url(url or "")
    if url:
        raw = f"{vendor}|{url}"
    else:
        th = hashlib.sha256((text or "").encode("utf-8")).hexdigest()
        raw = f"{vendor}|text|{th}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()

def _records_to_nodes(records: List[Dict[str, Any]], vendor: str) -> List[TextNode]:
    out: List[TextNode] = []
    v = normalize_vendor(vendor)

    for r in records or []:
        if not isinstance(r, dict):
            continue

        rr = dict(r)
        rr["vendor"] = rr.get("vendor") or v  # ensure vendor in TEXT

        url_raw = str(rr.get("url") or "").strip()
        url = canonical_url(url_raw) if url_raw else ""

        text = _record_to_text(rr)
        if not text:
            continue

        nid = _stable_id(v, url, text)
        meta = {"url": url} if url else {}  # ONLY url

        out.append(TextNode(id_=nid, text=text, metadata=meta))

    return out


# -----------------------------------------------------------------------------
# Output normalizer: accept dict-with-nodes, list-of-nodes, list-of-records
# -----------------------------------------------------------------------------
def _as_nodes(result: Any, vendor: str) -> List[Any]:
    # Case A: scraper returns {"nodes": [...]}
    if isinstance(result, dict) and "nodes" in result:
        nodes = result.get("nodes") or []
        return list(nodes) if isinstance(nodes, list) else []

    # Case B: scraper returns list
    if isinstance(result, list):
        if not result:
            return []

        first = result[0]

        # list[TextNode] or node-like
        if hasattr(first, "text"):
            return result

        # list[dict] records -> convert
        if isinstance(first, dict):
            return _records_to_nodes(result, vendor)

    # Unknown shape
    return []


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def scrape_vendor_nodes(vendor: VendorName, check_qdrant: bool = True) -> List[Any]:
    """
    Run a single vendor scraper and ALWAYS return a List of nodes (TextNode or node-like).

    - Nokia/Huawei: already return nodes via scrape_*_nodes()
    - Others: if they still return list[dict], we convert to TextNodes here
    """
    if vendor == "nokia":
        res = scrape_nokia_nodes(check_qdrant=check_qdrant, return_records=False)
        return _as_nodes(res, "nokia")

    if vendor == "huawei":
        res = scrape_huawei_nodes(check_qdrant=check_qdrant, return_records=False)
        return _as_nodes(res, "huawei")

    # These may still return list[dict] in your repo; orchestrator converts them.
    if vendor == "ericsson":
        return _as_nodes(scrape_ericsson(check_qdrant=check_qdrant), "ericsson")

    if vendor == "cisco":
        return _as_nodes(scrape_cisco(check_qdrant=check_qdrant), "cisco")

    if vendor == "variot":
        return _as_nodes(scrape_variot(check_qdrant=check_qdrant), "variot")

    if vendor == "mitre_mobile":
        return _as_nodes(scrape_mitre_mobile(check_qdrant=check_qdrant), "mitre_mobile")

    raise ValueError(f"Unknown vendor: {vendor}")


def scrape_all_vendors_nodes(
    vendors: Optional[List[VendorName]] = None,
    check_qdrant: bool = True,
) -> List[Any]:
    """
    High-level orchestrator:
      - Calls each scraper
      - Normalizes into ONE flat list of nodes
      - Works even if some scrapers still return list[dict]
    """
    if vendors is None:
        vendors = ["nokia", "ericsson", "huawei", "cisco", "variot", "mitre_mobile"]

    all_nodes: List[Any] = []

    for v in vendors:
        try:
            nodes = scrape_vendor_nodes(v, check_qdrant=check_qdrant)
            print(f"[SCRAPER] {v}: {len(nodes)} new nodes")
            all_nodes.extend(nodes)
        except Exception as e:
            print(f"[WARN] {v} scraper failed: {e}")

    print(f"[SCRAPER] Total new nodes from all vendors: {len(all_nodes)}")
    return all_nodes
