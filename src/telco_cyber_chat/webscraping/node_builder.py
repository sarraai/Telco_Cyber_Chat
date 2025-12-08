"""
node_builder.py

Builds LlamaIndex TextNodes (and simple RelationshipNodes) from:
- Telco vendor / VARIoT scrapers (unified {url, title, description} docs)
- MITRE ATT&CK dataframe (non-relationship + relationship objects)

Node content:
    "<title>\n\n<description>"

Node metadata:
    {"url": "<canonical url>", "source": "<vendor | dataset>", ...}

Relationship objects (MITRE):
    represented as lightweight RelationshipNode dataclasses.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd
from llama_index.core.schema import TextNode

# If you already created this in your previous step:
# from telco_cyber_chat.webscraping.scrape_orchestrator import scrape_all_vendors

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _stable_id(*parts: str, prefix: str = "") -> str:
    """
    Create a stable hex id from arbitrary string parts.
    Used for TextNode.id_ and relationship node ids.
    """
    raw = "|".join(p or "" for p in parts)
    if prefix:
        raw = prefix + "|" + raw
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _coerce_str(x: Any) -> str:
    """
    Safe string coercion: None/NaN -> "".
    """
    if isinstance(x, str):
        return x.strip()
    if x is None:
        return ""
    try:
        # pandas NaN
        if pd.isna(x):
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


# ---------------------------------------------------------------------------
# Simple relationship node representation
# ---------------------------------------------------------------------------


@dataclass
class RelationshipNode:
    """
    Lightweight representation of a relationship edge, so you can
    embed / store / reason over relationships separately if you want.
    """
    id: str
    text: str
    metadata: Dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# 1) Vendor / VARIoT docs -> TextNodes
# ---------------------------------------------------------------------------


def build_vendor_nodes(
    docs: List[Dict[str, str]],
    source: str,
) -> List[TextNode]:
    """
    Convert a list of vendor/VARIoT documents into TextNodes.

    Each doc is expected to have:
        { "url": str, "title": str, "description": str }

    Node:
        text     = "<title>\\n\\n<description>"
        metadata = { "url": url, "source": source, "title": title }
    """
    nodes: List[TextNode] = []

    for idx, doc in enumerate(docs):
        url = _coerce_str(doc.get("url"))
        title = _coerce_str(doc.get("title"))
        desc = _coerce_str(doc.get("description"))

        if not (title or desc):
            # Nothing to embed → skip
            continue

        content_parts = [p for p in [title, desc] if p]
        text = "\n\n".join(content_parts)

        node_id = _stable_id(url or title or str(idx), prefix=source.upper())

        metadata: Dict[str, Any] = {
            "url": url,
            "source": source,
        }
        if title:
            metadata["title"] = title

        nodes.append(
            TextNode(
                id_=node_id,
                text=text,
                metadata=metadata,
            )
        )

    return nodes


# ---------------------------------------------------------------------------
# 2) MITRE ATT&CK: content nodes (non-relationship) + relationship nodes
# ---------------------------------------------------------------------------

MITRE_RELATIONSHIP_TYPE = "relationship"


def _split_listish(val: Any) -> List[str]:
    """Tiny helper for splitting semicolon/comma/newline lists."""
    if val is None:
        return []
    if isinstance(val, list):
        out: List[str] = []
        for x in val:
            s = _coerce_str(x)
            if s:
                out.append(s)
        return out
    return [p.strip() for p in str(val).split(",") if p.strip()]


def build_mitre_text(r: Dict[str, Any]) -> str:
    """
    Build a rich text representation for a single MITRE ATT&CK object
    (non-relationship).
    """
    name = _coerce_str(r.get("name"))
    description = _coerce_str(r.get("description"))
    technique_id = _coerce_str(r.get("id"))
    external_id = _coerce_str(r.get("external_id") or r.get("externalId"))

    tactics = ", ".join(_split_listish(r.get("x_mitre_tactics")))
    platforms = ", ".join(_split_listish(r.get("x_mitre_platforms")))
    impact = ", ".join(_split_listish(r.get("x_mitre_impact_type")))
    permissions = ", ".join(_split_listish(r.get("x_mitre_permissions_required")))

    header_bits = [b for b in [technique_id, external_id, name] if b]
    header = " / ".join(header_bits)

    parts: List[str] = []
    if header:
        parts.append(header)
    if description:
        parts.append("Description:\n" + description)
    if tactics:
        parts.append("Tactics:\n" + tactics)
    if platforms:
        parts.append("Platforms:\n" + platforms)
    if permissions:
        parts.append("Permissions Required:\n" + permissions)
    if impact:
        parts.append("Impact Type:\n" + impact)

    return "\n\n".join(parts).strip() or header or name or "MITRE object"


def build_mitre_metadata(r: Dict[str, Any]) -> Dict[str, Any]:
    """
    Minimal metadata for MITRE content nodes.
    """
    return {
        "source": "MITRE",
        "url": "",  # MITRE JSON doesn't have a canonical URL here
        "id": _coerce_str(r.get("id")),
        "title": _coerce_str(r.get("name")),
        "type": _coerce_str(r.get("type")),
    }


def build_mitre_nodes_from_df(
    df_mitre: pd.DataFrame,
) -> Tuple[List[TextNode], List[RelationshipNode]]:
    """
    Build MITRE nodes from a MITRE ATT&CK DataFrame.

    - Non-relationship rows  -> TextNode (title + description in content,
      'url' in metadata left empty).
    - Relationship rows      -> RelationshipNode
        (source_ref -> target_ref with relationship_type).

    Returns:
        (content_nodes, relationship_nodes)
    """
    if "id" not in df_mitre.columns or "type" not in df_mitre.columns:
        raise KeyError("df_mitre must have at least 'id' and 'type' columns")

    type_series = df_mitre["type"].astype(str).str.strip().str.lower()

    # 1) Content nodes (non-relationship)
    mask_content = ~type_series.eq(MITRE_RELATIONSHIP_TYPE)
    df_content = df_mitre[mask_content].copy()

    content_nodes: List[TextNode] = []
    for rec in df_content.to_dict(orient="records"):
        text = build_mitre_text(rec)
        meta = build_mitre_metadata(rec)

        if not text.strip():
            continue

        obj_id = meta.get("id") or _coerce_str(rec.get("external_id")) or ""
        node_id = obj_id or _stable_id(text[:64], prefix="MITRE")

        content_nodes.append(
            TextNode(
                id_=node_id,
                text=text,
                metadata=meta,
            )
        )

    # 2) Relationship nodes
    has_src = "source_ref" in df_mitre.columns
    has_tgt = "target_ref" in df_mitre.columns
    has_rtype = "relationship_type" in df_mitre.columns

    relationship_nodes: List[RelationshipNode] = []

    if has_src and has_tgt and has_rtype:
        df_rel = df_mitre[type_series.eq(MITRE_RELATIONSHIP_TYPE)].copy()

        # map object id -> name/type for nicer text
        id_to_row: Dict[str, Dict[str, Any]] = {}
        for rec in df_mitre.to_dict(orient="records"):
            oid = _coerce_str(rec.get("id"))
            if oid:
                id_to_row[oid] = rec

        for rec in df_rel.to_dict(orient="records"):
            src_id = _coerce_str(rec.get("source_ref"))
            tgt_id = _coerce_str(rec.get("target_ref"))
            rtype = _coerce_str(rec.get("relationship_type")) or "<unspecified>"
            rel_id = _coerce_str(rec.get("id")) or _stable_id(src_id, tgt_id, rtype, prefix="MITRE_REL")

            src_row = id_to_row.get(src_id, {})
            tgt_row = id_to_row.get(tgt_id, {})
            src_name = _first_nonempty_str(src_row.get("name"), src_row.get("title"), src_id)
            tgt_name = _first_nonempty_str(tgt_row.get("name"), tgt_row.get("title"), tgt_id)

            text_lines = [
                f"MITRE Relationship: {rtype}",
                f"Source: {src_name} ({src_id})",
                f"Target: {tgt_name} ({tgt_id})",
            ]
            text = "\n".join(text_lines)

            metadata = {
                "source": "MITRE",
                "relationship_type": rtype,
                "source_ref": src_id,
                "target_ref": tgt_id,
                "relationship_id": rel_id,
            }

            relationship_nodes.append(
                RelationshipNode(
                    id=rel_id,
                    text=text,
                    metadata=metadata,
                )
            )

    return content_nodes, relationship_nodes


# ---------------------------------------------------------------------------
# 3) Optional convenience: build all scraped vendor nodes
# ---------------------------------------------------------------------------

# Example stub – uncomment and adapt once you have scrape_orchestrator implemented.
#
# def build_all_scraped_nodes(check_qdrant: bool = True) -> List[TextNode]:
#     """
#     High-level helper:
#       - Calls scrape_all_vendors(check_qdrant=...) to get a unified list
#         of {url, title, description, source} docs.
#       - Builds TextNodes for each vendor/dataset.
#
#     This is what your ingestion cron / LangGraph node will likely call.
#     """
#     docs = scrape_all_vendors(check_qdrant=check_qdrant)
#
#     nodes: List[TextNode] = []
#     for doc in docs:
#         src = _coerce_str(doc.get("source") or "unknown")
#         nodes.extend(build_vendor_nodes([doc], source=src))
#     return nodes
