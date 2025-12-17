# telco_cyber_chat/webscraping/scrape_orchestrator.py

from __future__ import annotations

import hashlib
import json
from typing import Any, Dict, List, Literal, Optional

from llama_index.core.schema import TextNode

from .scrape_core import canonical_url, normalize_vendor

# ✅ Scrapers that return TextNodes already (based on the code you shared)
from .nokia_scraper import scrape_nokia
from .huawei_scraper import scrape_huawei_nodes

# These may return nodes or records depending on your repo versions
from .ericsson_scraper import scrape_ericsson
from .cisco_scraper import scrape_cisco

# VARIoT in your current pipeline returns dict {"nodes": [...]}
from .variot_scraper import scrape_variot_nodes

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
def _as_nodes(result: Any, vendor: str) -> List[TextNode]:
    # Case A: scraper returns {"nodes": [...]}
    if isinstance(result, dict) and "nodes" in result:
        nodes = result.get("nodes") or []
        # if they are already TextNodes, return them
        if isinstance(nodes, list) and (not nodes or isinstance(nodes[0], TextNode)):
            return nodes
        return []

    # Case B: scraper returns list
    if isinstance(result, list):
        if not result:
            return []

        first = result[0]

        # list[TextNode]
        if isinstance(first, TextNode):
            return result

        # list[dict] records -> convert
        if isinstance(first, dict):
            return _records_to_nodes(result, vendor)

    return []


# -----------------------------------------------------------------------------
# Public API
# -----------------------------------------------------------------------------
def scrape_vendor_nodes(vendor: VendorName, check_qdrant: bool = True) -> List[TextNode]:
    """
    Run a single vendor scraper and ALWAYS return List[TextNode].
    """
    if vendor == "nokia":
        return scrape_nokia(check_qdrant=check_qdrant)

    if vendor == "huawei":
        return scrape_huawei_nodes(check_qdrant=check_qdrant)

    if vendor == "ericsson":
        return _as_nodes(scrape_ericsson(check_qdrant=check_qdrant), "ericsson")

    if vendor == "cisco":
        return _as_nodes(scrape_cisco(check_qdrant=check_qdrant), "cisco")

    if vendor == "variot":
        # returns dict {"nodes": [...]}
        return _as_nodes(scrape_variot_nodes(check_qdrant=check_qdrant), "variot")

    if vendor == "mitre_mobile":
        return _as_nodes(scrape_mitre_mobile(check_qdrant=check_qdrant), "mitre_mobile")

    raise ValueError(f"Unknown vendor: {vendor}")


def scrape_all_vendors_nodes(
    vendors: Optional[List[VendorName]] = None,
    check_qdrant: bool = True,
) -> List[TextNode]:
    """
    Calls each scraper, normalizes results into ONE flat list of TextNodes.
    """
    if vendors is None:
        vendors = ["nokia", "ericsson", "huawei", "cisco", "variot", "mitre_mobile"]

    all_nodes: List[TextNode] = []

    for v in vendors:
        try:
            nodes = scrape_vendor_nodes(v, check_qdrant=check_qdrant)
            print(f"🟣 [SCRAPER] {v}: {len(nodes)} new nodes", flush=True)
            all_nodes.extend(nodes)
        except Exception as e:
            # Keep it non-error-looking in logs
            print(f"🟠 [SCRAPER] {v} skipped: {e}", flush=True)

    print(f"🟣 [SCRAPER] Total new nodes (all vendors): {len(all_nodes)}", flush=True)
    return all_nodes
