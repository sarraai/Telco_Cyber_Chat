"""
ingest_pipeline.py

MUST expose: ingest_all_sources

This version:
- No dependency on node_builder.py (builders included here)
- Avoids heavy imports at module import-time (faster LangGraph startup)
- Optional check_qdrant dedupe:
    - vendor sources: vendor + url
    - MITRE: vendor + stix_id
"""

from __future__ import annotations

import os
import re
import json
import hashlib
import logging
from typing import Any, Dict, List, Optional, Tuple, Sequence

logger = logging.getLogger(__name__)

MITRE_SOURCE_KEY = "mitre_mobile"
_STIX_RE = re.compile(r"(?i)\b(stix--[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})\b")


# ---------------------------------------------------------------------
# Lightweight JSON loaders (no pandas at import-time)
# ---------------------------------------------------------------------
def _load_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _load_records_list(path: str) -> List[Dict[str, Any]]:
    data = _load_json(path)
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]
    if isinstance(data, dict):
        recs = data.get("records")
        if isinstance(recs, list):
            return [x for x in recs if isinstance(x, dict)]
    return []


def _load_mitre_objects(mitre_json_path: str) -> List[Dict[str, Any]]:
    bundle = _load_json(mitre_json_path)
    if not isinstance(bundle, dict):
        return []
    objs = bundle.get("objects", [])
    if not isinstance(objs, list):
        return []
    return [o for o in objs if isinstance(o, dict)]


def _vendor_inputs_from_env() -> List[Tuple[str, str]]:
    mapping = [
        ("nokia", os.getenv("NOKIA_JSON_PATH", "").strip()),
        ("huawei", os.getenv("HUAWEI_JSON_PATH", "").strip()),
        ("ericsson", os.getenv("ERICSSON_JSON_PATH", "").strip()),
        ("cisco", os.getenv("CISCO_JSON_PATH", "").strip()),
        ("variot", os.getenv("VARIOT_JSON_PATH", "").strip()),
    ]
    return [(v, p) for (v, p) in mapping if p]


def _get_collection_name(explicit: Optional[str]) -> str:
    return (explicit or os.getenv("QDRANT_COLLECTION", "Telco_CyberChat")).strip()


# ---------------------------------------------------------------------
# Minimal node builders (replaces deleted node_builder.py)
# ---------------------------------------------------------------------
def _stable_id(*parts: str, prefix: str = "") -> str:
    raw = "|".join(p or "" for p in parts)
    if prefix:
        raw = f"{prefix}|{raw}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _dump_json(v: Any) -> str:
    try:
        return json.dumps(v, ensure_ascii=False, sort_keys=True, indent=2)
    except Exception:
        return str(v)


def _record_to_text(rec: Dict[str, Any]) -> str:
    """Text = all fields except url."""
    lines: List[str] = []
    for k in sorted(rec.keys(), key=lambda s: str(s).lower()):
        ks = str(k).strip()
        if not ks or ks.lower() == "url":
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


def build_vendor_nodes(records: Sequence[Dict[str, Any]], vendor: str) -> List["TextNode"]:
    """metadata includes url + vendor so Qdrant filtering always works."""
    from llama_index.core.schema import TextNode  # lazy import

    from telco_cyber_chat.webscraping.scrape_core import canonical_url, normalize_vendor

    vnorm = normalize_vendor(vendor)
    nodes: List[TextNode] = []

    for rec in records or []:
        if not isinstance(rec, dict):
            continue

        url = rec.get("url")
        if not isinstance(url, str) or not url.strip():
            continue
        cu = canonical_url(url.strip())

        rec2 = dict(rec)
        rec2["vendor"] = rec2.get("vendor") or vnorm
        rec2["url"] = cu  # keep for later (but excluded from text)

        text = _record_to_text(rec2)
        if not text:
            continue

        nid = _stable_id(vnorm, cu, prefix="vendor")
        nodes.append(TextNode(id_=nid, text=text, metadata={"url": cu, "vendor": vnorm}))
    return nodes


def _mitre_external_id(obj: Dict[str, Any]) -> str:
    ext = obj.get("external_references") or []
    if isinstance(ext, list):
        for r in ext:
            if not isinstance(r, dict):
                continue
            eid = r.get("external_id")
            if isinstance(eid, str) and eid.strip():
                return eid.strip()
    return ""


def _mitre_url(obj: Dict[str, Any]) -> str:
    ext = obj.get("external_references") or []
    if isinstance(ext, list):
        for r in ext:
            if not isinstance(r, dict):
                continue
            u = r.get("url")
            if isinstance(u, str) and u.strip():
                return u.strip()
    u2 = obj.get("url")
    return u2.strip() if isinstance(u2, str) and u2.strip() else ""


def build_mitre_nodes(objects: Sequence[Dict[str, Any]], vendor: str = MITRE_SOURCE_KEY) -> List["TextNode"]:
    """Build TextNodes from STIX objects. metadata includes stix_id for dedupe."""
    from llama_index.core.schema import TextNode  # lazy import
    from telco_cyber_chat.webscraping.scrape_core import normalize_vendor

    vnorm = normalize_vendor(vendor)
    nodes: List[TextNode] = []

    for obj in objects or []:
        if not isinstance(obj, dict):
            continue

        stix_id = str(obj.get("id") or "").strip()
        url = _mitre_url(obj)
        external_id = _mitre_external_id(obj)

        rec = dict(obj)
        rec["vendor"] = vnorm
        if external_id:
            rec["external_id"] = external_id
        if stix_id:
            rec["stix_id"] = stix_id

        # rule: url not in text
        if "url" in rec:
            rec.pop("url", None)

        text = _record_to_text(rec)
        if not text:
            continue

        nid = _stable_id(vnorm, stix_id or external_id, obj.get("type", ""), prefix="mitre")
        meta: Dict[str, Any] = {"vendor": vnorm}
        if url:
            meta["url"] = url
        if stix_id:
            meta["stix_id"] = stix_id
        if external_id:
            meta["external_id"] = external_id

        nodes.append(TextNode(id_=nid, text=text, metadata=meta))

    return nodes


# ---------------------------------------------------------------------
# Qdrant helpers (lazy import qdrant_client)
# ---------------------------------------------------------------------
def _get_qdrant_client_safe():
    qurl = os.getenv("QDRANT_URL", "").strip()
    if not qurl:
        return None
    qkey = os.getenv("QDRANT_API_KEY", "").strip() or None
    try:
        from qdrant_client import QdrantClient
        return QdrantClient(url=qurl, api_key=qkey)
    except Exception:
        logger.exception("Failed to create QdrantClient; proceeding without DB checks.")
        return None


def _qdrant_has_vendor_url(client, collection: str, vendor: str, url: str) -> bool:
    from qdrant_client import models as qmodels
    from telco_cyber_chat.webscraping.scrape_core import normalize_vendor

    flt = qmodels.Filter(
        must=[
            qmodels.FieldCondition(key="vendor", match=qmodels.MatchValue(value=normalize_vendor(vendor))),
            qmodels.FieldCondition(key="url", match=qmodels.MatchValue(value=url)),
        ]
    )
    try:
        res = client.count(collection_name=collection, count_filter=flt, exact=False)
        return (res.count or 0) > 0
    except Exception:
        logger.warning("Qdrant vendor+url check failed; proceeding as not ingested.")
        return False


def _qdrant_has_mitre_stix(client, collection: str, vendor_value: str, stix_id: str) -> bool:
    from qdrant_client import models as qmodels
    from telco_cyber_chat.webscraping.scrape_core import normalize_vendor

    flt = qmodels.Filter(
        must=[
            qmodels.FieldCondition(key="vendor", match=qmodels.MatchValue(value=normalize_vendor(vendor_value))),
            qmodels.FieldCondition(key="stix_id", match=qmodels.MatchValue(value=stix_id)),
        ]
    )
    try:
        res = client.count(collection_name=collection, count_filter=flt, exact=False)
        return (res.count or 0) > 0
    except Exception:
        logger.warning("Qdrant stix_id check failed; proceeding as not ingested.")
        return False


def _extract_stix_from_text(text: str) -> str:
    if not text:
        return ""
    m = _STIX_RE.search(text)
    return m.group(1) if m else ""


# ---------------------------------------------------------------------
# REQUIRED export
# ---------------------------------------------------------------------
async def ingest_all_sources(*args, **kwargs) -> Dict[str, Any]:
    _ = args  # unused

    # Lazy import heavy modules used only at runtime
    from telco_cyber_chat.webscraping.node_embed import embed_nodes_hybrid
    from telco_cyber_chat.webscraping.qdrant_ingest import upsert_nodes_to_qdrant
    from telco_cyber_chat.webscraping.scrape_core import canonical_url, normalize_vendor

    qdrant_collection: Optional[str] = kwargs.get("qdrant_collection")
    collection = _get_collection_name(qdrant_collection)

    check_qdrant: bool = bool(kwargs.get("check_qdrant", False))
    verbose: bool = bool(kwargs.get("verbose", True))

    concurrency: int = int(kwargs.get("concurrency", 2))
    embed_batch_size: int = int(kwargs.get("embed_batch_size", 20))
    upsert_batch_size: int = int(kwargs.get("upsert_batch_size", 64))

    def _say(msg: str) -> None:
        logger.info(msg)
        if verbose:
            print(msg, flush=True)

    _say(f"[INGEST] Starting ingest_all_sources(check_qdrant={check_qdrant}) into collection='{collection}'")

    client = _get_qdrant_client_safe() if check_qdrant else None
    if check_qdrant and client is None:
        _say("[INGEST] check_qdrant=True but QDRANT_URL not set or client init failed -> skipping DB checks.")

    per_source: Dict[str, int] = {}
    all_nodes: List["TextNode"] = []

    # 1) Vendor JSONs
    for vendor, path in _vendor_inputs_from_env():
        try:
            records = _load_records_list(path)

            kept_records: List[Dict[str, Any]] = []
            skipped_existing = 0

            for rec in records:
                if not isinstance(rec, dict):
                    continue
                r2 = dict(rec)

                url = r2.get("url")
                if isinstance(url, str) and url.strip():
                    cu = canonical_url(url.strip())
                    r2["url"] = cu

                    if client is not None and _qdrant_has_vendor_url(client, collection, vendor, cu):
                        skipped_existing += 1
                        continue

                kept_records.append(r2)

            nodes = build_vendor_nodes(kept_records, vendor=vendor)
            all_nodes.extend(nodes)
            per_source[normalize_vendor(vendor)] = len(nodes)

            _say(
                f"[INGEST] vendor={vendor} file='{path}' "
                f"records={len(records)} kept={len(kept_records)} skipped_existing={skipped_existing} nodes={len(nodes)}"
            )
        except Exception:
            per_source[normalize_vendor(vendor)] = 0
            logger.exception("Failed loading/processing vendor=%s from %s", vendor, path)
            if verbose:
                print(f"[INGEST] ERROR vendor={vendor} from '{path}' (see stacktrace above)", flush=True)

    # 2) MITRE
    mitre_path = os.getenv("MITRE_JSON_PATH", "").strip()
    if mitre_path:
        try:
            objs = _load_mitre_objects(mitre_path)
            mitre_nodes_all = build_mitre_nodes(objs, vendor=MITRE_SOURCE_KEY)

            if client is not None:
                kept_nodes: List["TextNode"] = []
                skipped = 0

                for n in mitre_nodes_all:
                    stix = ""
                    # Prefer metadata stix_id (we set it)
                    stix = str(getattr(n, "metadata", {}).get("stix_id") or "").strip()

                    # Fallback to id_ or text scan
                    if not stix:
                        nid = str(getattr(n, "id_", "") or "")
                        if nid.lower().startswith("stix--"):
                            stix = nid
                        else:
                            stix = _extract_stix_from_text(getattr(n, "text", "") or "")

                    if stix and _qdrant_has_mitre_stix(client, collection, MITRE_SOURCE_KEY, stix):
                        skipped += 1
                        continue

                    kept_nodes.append(n)

                mitre_nodes_new = kept_nodes
                _say(
                    f"[INGEST] mitre file='{mitre_path}' nodes_all={len(mitre_nodes_all)} "
                    f"kept={len(mitre_nodes_new)} skipped_existing={skipped}"
                )
            else:
                mitre_nodes_new = mitre_nodes_all
                _say(f"[INGEST] mitre file='{mitre_path}' nodes_all={len(mitre_nodes_all)} (no DB check)")

            all_nodes.extend(mitre_nodes_new)
            per_source[MITRE_SOURCE_KEY] = len(mitre_nodes_new)

        except Exception:
            per_source[MITRE_SOURCE_KEY] = 0
            logger.exception("Failed loading MITRE from %s", mitre_path)
            if verbose:
                print(f"[INGEST] ERROR mitre from '{mitre_path}' (see stacktrace above)", flush=True)
    else:
        per_source[MITRE_SOURCE_KEY] = 0
        _say("[INGEST] MITRE_JSON_PATH not set -> skipping MITRE")

    if not all_nodes:
        _say("[INGEST] No new nodes to ingest (after filtering).")
        return {
            "ok": True,
            "nodes": 0,
            "upserted": 0,
            "per_source": per_source,
            "collection": collection,
            "message": "No nodes to ingest.",
        }

    _say(f"[INGEST] Embedding {len(all_nodes)} nodes (concurrency={concurrency}, batch={embed_batch_size}) ...")
    embeddings = await embed_nodes_hybrid(all_nodes, concurrency=concurrency, batch_size=embed_batch_size)

    _say(f"[INGEST] Upserting to Qdrant (batch={upsert_batch_size}) ...")
    upserted = upsert_nodes_to_qdrant(
        all_nodes,
        embeddings=embeddings,
        collection_name=collection,
        batch_size=upsert_batch_size,
    )

    _say(f"[INGEST] Done. upserted={int(upserted)} nodes={len(all_nodes)} per_source={per_source}")
    return {
        "ok": True,
        "nodes": len(all_nodes),
        "upserted": int(upserted),
        "per_source": per_source,
        "collection": collection,
    }
