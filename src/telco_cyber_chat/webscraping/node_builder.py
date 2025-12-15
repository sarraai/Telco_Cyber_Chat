"""
node_builder.py

Builds LlamaIndex TextNodes (and RelationshipNodes) from:
- Telco vendor / VARIoT scrapers (records with vendor-specific fields)
- MITRE ATT&CK dataframe (non-relationship + relationship objects)

RULE:
- Put ALL fields except URL into node.text (key/value readable format)
- Put ONLY URL into node.metadata = {"url": "<canonical url>"} (or {} if missing)

Relationship objects (MITRE):
- represented as lightweight RelationshipNode dataclass objects
- RelationshipNode.text contains all relationship fields (except url)
- RelationshipNode.metadata contains only url (if any)
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from llama_index.core.schema import TextNode


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

URL_KEYS = {"url", "canonical_url", "link", "href"}


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
    # prefer "url", then some common variants
    for k in ("url", "canonical_url", "link", "href"):
        v = rec.get(k)
        if isinstance(v, str) and v.strip():
            return v.strip()
    return ""


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
    header_lines: Optional[List[str]] = None,
    exclude_keys: Optional[set[str]] = None,
) -> str:
    """
    Put ALL fields (except excluded ones) into text as readable key/value blocks.
    Deterministic order: preferred keys first, then the rest sorted.

    Example:
      vendor: huawei
      title: ...
      description: ...
      software_versions_and_fixes:
      [
        ...
      ]
    """
    exclude_keys = exclude_keys or set()

    # Normalize keys to strings (just in case)
    norm_rec: Dict[str, Any] = {}
    for k, v in (rec or {}).items():
        ks = _coerce_str(k)
        if ks:
            norm_rec[ks] = v

    lines: List[str] = []

    if header_lines:
        for h in header_lines:
            hs = _coerce_str(h)
            if hs:
                lines.append(hs)

    preferred = [
        "vendor",
        "source",
        "dataset",
        "title",
        "page_title",
        "name",
        "type",
        "id",
        "external_id",
        "description",
        "summary",
    ]

    keys = list(norm_rec.keys())

    # preferred first (if present)
    out_keys: List[str] = []
    for k in preferred:
        if k in keys and k not in exclude_keys:
            out_keys.append(k)

    # then the rest sorted
    rest = sorted([k for k in keys if k not in out_keys and k not in exclude_keys])
    out_keys.extend(rest)

    for k in out_keys:
        if k in exclude_keys:
            continue

        v = norm_rec.get(k)

        # IMPORTANT: skip URL-like keys always (even if caller forgot)
        if k.lower() in URL_KEYS:
            continue

        if _is_empty_value(v):
            continue

        if isinstance(v, (dict, list)):
            lines.append(f"{k}:\n{_dump_json(v)}")
        else:
            vs = _coerce_str(v)
            if vs:
                lines.append(f"{k}: {vs}")

    return "\n".join(lines).strip()


# ---------------------------------------------------------------------------
# Relationship node representation
# ---------------------------------------------------------------------------

@dataclass
class RelationshipNode:
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1) Vendor / VARIoT records -> TextNodes
# ---------------------------------------------------------------------------

def build_vendor_nodes(
    records: List[Dict[str, Any]],
    vendor: str,
) -> List[TextNode]:
    """
    Convert vendor records into TextNodes.

    RULE:
      - text     = ALL fields except url
      - metadata = {"url": "..."} only
    """
    nodes: List[TextNode] = []

    for idx, rec in enumerate(records):
        if not isinstance(rec, dict):
            continue

        url = _pick_url(rec)

        # Ensure vendor appears in text even if scraper didn't include it
        rec2 = dict(rec)
        if not _coerce_str(rec2.get("vendor")):
            rec2["vendor"] = vendor

        text = _build_text_from_record(
            rec2,
            header_lines=None,
            exclude_keys=set(URL_KEYS),
        )

        if not text.strip():
            continue

        # stable id: prefer URL; else fall back to title/name/id
        fallback = _first_nonempty_str(rec2.get("id"), rec2.get("title"), rec2.get("name"), str(idx))
        node_id = _stable_id(url or fallback, prefix=vendor.upper())

        metadata = {"url": url} if url else {}

        nodes.append(
            TextNode(
                id_=node_id,
                text=text,
                metadata=metadata,
            )
        )

    return nodes


# ---------------------------------------------------------------------------
# 2) MITRE ATT&CK: non-relationship -> TextNodes, relationship -> RelationshipNodes
# ---------------------------------------------------------------------------

MITRE_RELATIONSHIP_TYPE = "relationship"


def extract_primary_url_from_external_refs(external_references: Any) -> str:
    """
    Try to extract a canonical URL from MITRE external_references if present.
    Prefer mitre-attack / mitre-mobile-attack source_name, otherwise any url.
    """
    refs = external_references or []
    if not isinstance(refs, list):
        return ""

    for ref in refs:
        if not isinstance(ref, dict):
            continue
        src = (ref.get("source_name") or "").lower()
        if "mitre-attack" in src or "mitre-mobile-attack" in src:
            u = ref.get("url")
            if isinstance(u, str) and u.strip():
                return u.strip()

    for ref in refs:
        if not isinstance(ref, dict):
            continue
        u = ref.get("url")
        if isinstance(u, str) and u.strip():
            return u.strip()

    return ""


def build_mitre_nodes_from_df(
    df_mitre: pd.DataFrame,
) -> Tuple[List[TextNode], List[RelationshipNode]]:
    """
    RULE:
      - Non-relationship TextNodes: all fields except url in text; url only in metadata
      - RelationshipNodes: all fields except url in text; url only in metadata

    Returns:
      (content_nodes, relationship_nodes)
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

        # URL (if any) from external_references
        url = extract_primary_url_from_external_refs(rec.get("external_references"))

        rec2 = dict(rec)
        rec2.setdefault("vendor", "mitre")  # ensure it appears in text

        text = _build_text_from_record(
            rec2,
            exclude_keys=set(URL_KEYS),
        )
        if not text.strip():
            continue

        obj_id = _coerce_str(rec.get("id"))  # STIX id exists for content objects
        node_id = obj_id or _stable_id(text[:128], prefix="MITRE")

        metadata = {"url": url} if url else {}

        content_nodes.append(
            TextNode(
                id_=node_id,
                text=text,
                metadata=metadata,
            )
        )

    # -------- relationships --------
    relationship_nodes: List[RelationshipNode] = []

    has_src = "source_ref" in df_mitre.columns
    has_tgt = "target_ref" in df_mitre.columns
    has_rtype = "relationship_type" in df_mitre.columns

    if has_src and has_tgt and has_rtype:
        df_rel = df_mitre[type_series.eq(MITRE_RELATIONSHIP_TYPE)].copy()

        for rec in df_rel.to_dict(orient="records"):
            if not isinstance(rec, dict):
                continue

            url = extract_primary_url_from_external_refs(rec.get("external_references"))

            rec2 = dict(rec)
            rec2.setdefault("vendor", "mitre")  # ensure it appears in text

            text = _build_text_from_record(
                rec2,
                exclude_keys=set(URL_KEYS),
            )
            if not text.strip():
                continue

            rel_id = _coerce_str(rec.get("id"))
            # fallback stable id if missing
            if not rel_id:
                rel_id = _stable_id(
                    _coerce_str(rec.get("source_ref")),
                    _coerce_str(rec.get("target_ref")),
                    _coerce_str(rec.get("relationship_type")),
                    prefix="MITRE_REL",
                )

            metadata = {"url": url} if url else {}

            relationship_nodes.append(
                RelationshipNode(
                    id=rel_id,
                    text=text,
                    metadata=metadata,
                )
            )

    return content_nodes, relationship_nodes
