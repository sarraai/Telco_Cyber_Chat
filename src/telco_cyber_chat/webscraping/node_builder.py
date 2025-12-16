"""
node_builder.py

Builds LlamaIndex TextNodes from:
- Telco vendor / VARIoT scrapers (records with vendor-specific fields)
- MITRE ATT&CK dataframe (non-relationship + relationship objects)

RULE:
- Put ALL fields except URL into node.text (key/value readable format)
- Put ONLY URL into node.metadata = {"url": "<canonical url>"} (or {} if missing)

NOTES:
- URL field is always called: "url"
- Every node must contain a "vendor" field in its TEXT (injected if missing)

GRAPH NOTE (important):
- LlamaIndex TextNode.relationships is mainly for doc structure (parent/child/prev/next/source),
  not arbitrary MITRE "relationship" edges.
- For MITRE graph behavior, extract edges using extract_mitre_edges(...) and build a graph index/store
  alongside your Qdrant vector store.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from llama_index.core.schema import TextNode


# ----------------------------
# Helpers
# ----------------------------

URL_KEYS = {"url"}  # url is always called "url"


def _stable_id(*parts: str, prefix: str = "") -> str:
    raw = "|".join(p or "" for p in parts)
    if prefix:
        raw = prefix + "|" + raw
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _coerce_str(x: Any) -> str:
    if isinstance(x, str):
        return x.strip()
    if x is None:
        return ""
    try:
        if pd.isna(x):  # type: ignore[arg-type]
            return ""
    except Exception:
        pass
    return str(x).strip()


def _first_nonempty_str(*vals: Any) -> str:
    for v in vals:
        s = _coerce_str(v)
        if s:
            return s
    return ""


def _pick_url(rec: Dict[str, Any]) -> str:
    v = rec.get("url")
    return v.strip() if isinstance(v, str) and v.strip() else ""


def _is_empty_value(v: Any) -> bool:
    if v is None:
        return True
    if isinstance(v, str) and not v.strip():
        return True
    if isinstance(v, (list, tuple, set, dict)) and len(v) == 0:
        return True
    return False


def _dump_json(v: Any) -> str:
    try:
        return json.dumps(v, ensure_ascii=False, indent=2, sort_keys=True)
    except Exception:
        return str(v)


def _build_text_from_record(
    rec: Dict[str, Any],
    *,
    exclude_keys: Optional[set[str]] = None,
) -> str:
    """
    Put ALL fields (except excluded ones) into text as readable key/value blocks.

    - Deterministic: keys are sorted alphabetically.
    - Skips empty values.
    - Always skips URL keys.
    """
    exclude_keys = exclude_keys or set()

    # Normalize keys to strings
    norm_rec: Dict[str, Any] = {}
    for k, v in (rec or {}).items():
        ks = _coerce_str(k)
        if ks:
            norm_rec[ks] = v

    lines: List[str] = []

    for k in sorted(norm_rec.keys(), key=lambda s: s.lower()):
        if k in exclude_keys:
            continue
        if k.lower() in URL_KEYS:
            continue

        v = norm_rec.get(k)
        if _is_empty_value(v):
            continue

        if isinstance(v, (dict, list)):
            lines.append(f"{k}:\n{_dump_json(v)}")
        else:
            vs = _coerce_str(v)
            if vs:
                lines.append(f"{k}: {vs}")

    return "\n".join(lines).strip()


# ----------------------------
# 1) Vendor / VARIoT records -> TextNodes
# ----------------------------

def build_vendor_nodes(
    records: List[Dict[str, Any]],
    vendor: str,
) -> List[TextNode]:
    """
    Convert vendor records into TextNodes.

    RULE:
      - text     = ALL fields except url (inject vendor if missing)
      - metadata = {"url": "..."} only
    """
    nodes: List[TextNode] = []

    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue

        url = _pick_url(rec)

        rec2 = dict(rec)
        if not _coerce_str(rec2.get("vendor")):
            rec2["vendor"] = vendor  # inject vendor into TEXT

        text = _build_text_from_record(rec2, exclude_keys=set(URL_KEYS))
        if not text:
            continue

        fallback = _first_nonempty_str(rec2.get("id"), rec2.get("title"), rec2.get("name"), str(idx))
        node_id = _stable_id(url or fallback, prefix=vendor.upper())

        metadata = {"url": url} if url else {}
        nodes.append(TextNode(id_=node_id, text=text, metadata=metadata))

    return nodes


# ----------------------------
# 2) MITRE ATT&CK -> TextNodes (content + relationship objects)
# ----------------------------

MITRE_RELATIONSHIP_TYPE = "relationship"


def build_mitre_nodes(
    df_mitre: pd.DataFrame,
) -> Tuple[List[TextNode], List[TextNode]]:
    """
    Returns:
      (content_nodes, relationship_nodes)

    RULE (same for both):
      - node.text     = ALL fields except url (inject vendor="mitre" if missing)
      - node.metadata = {"url": "..."} only
    """
    if "id" not in df_mitre.columns or "type" not in df_mitre.columns:
        raise KeyError("df_mitre must have at least 'id' and 'type' columns")

    type_series = df_mitre["type"].astype(str).str.strip().str.lower()

    # -------- content (non-relationship) --------
    df_content = df_mitre[~type_series.eq(MITRE_RELATIONSHIP_TYPE)].copy()
    content_nodes: List[TextNode] = []

    for rec in df_content.to_dict(orient="records"):
        if not isinstance(rec, dict):
            continue

        url = _pick_url(rec)

        rec2 = dict(rec)
        if not _coerce_str(rec2.get("vendor")):
            rec2["vendor"] = "mitre"

        text = _build_text_from_record(rec2, exclude_keys=set(URL_KEYS))
        if not text:
            continue

        obj_id = _coerce_str(rec.get("id"))  # STIX id
        node_id = obj_id or _stable_id(text[:128], prefix="MITRE")

        metadata = {"url": url} if url else {}
        content_nodes.append(TextNode(id_=node_id, text=text, metadata=metadata))

    # -------- relationships (as TextNodes too) --------
    df_rel = df_mitre[type_series.eq(MITRE_RELATIONSHIP_TYPE)].copy()
    relationship_nodes: List[TextNode] = []

    for rec in df_rel.to_dict(orient="records"):
        if not isinstance(rec, dict):
            continue

        url = _pick_url(rec)

        rec2 = dict(rec)
        if not _coerce_str(rec2.get("vendor")):
            rec2["vendor"] = "mitre"

        text = _build_text_from_record(rec2, exclude_keys=set(URL_KEYS))
        if not text:
            continue

        rel_id = _coerce_str(rec.get("id"))
        if not rel_id:
            rel_id = _stable_id(
                _coerce_str(rec.get("source_ref")),
                _coerce_str(rec.get("target_ref")),
                _coerce_str(rec.get("relationship_type")),
                prefix="MITRE_REL",
            )

        metadata = {"url": url} if url else {}
        relationship_nodes.append(TextNode(id_=rel_id, text=text, metadata=metadata))

    return content_nodes, relationship_nodes


# ----------------------------
# 3) Extract structured MITRE edges (for real graph behavior)
# ----------------------------

@dataclass(frozen=True)
class MitreEdge:
    id: str
    source_ref: str
    target_ref: str
    relationship_type: str


def extract_mitre_edges(df_mitre: pd.DataFrame) -> List[MitreEdge]:
    """
    Build structured edges from MITRE relationship objects.

    This is what you should feed into a graph index/store (PropertyGraphIndex),
    while still using Qdrant for vector retrieval.
    """
    if "type" not in df_mitre.columns:
        return []

    type_series = df_mitre["type"].astype(str).str.strip().str.lower()
    df_rel = df_mitre[type_series.eq(MITRE_RELATIONSHIP_TYPE)].copy()

    edges: List[MitreEdge] = []
    for rec in df_rel.to_dict(orient="records"):
        if not isinstance(rec, dict):
            continue
        src = _coerce_str(rec.get("source_ref"))
        tgt = _coerce_str(rec.get("target_ref"))
        rty = _coerce_str(rec.get("relationship_type"))
        if not (src and tgt and rty):
            continue

        rid = _coerce_str(rec.get("id")) or _stable_id(src, tgt, rty, prefix="MITRE_EDGE")
        edges.append(MitreEdge(id=rid, source_ref=src, target_ref=tgt, relationship_type=rty))

    return edges
