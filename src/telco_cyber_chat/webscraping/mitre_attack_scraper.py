from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import List

from llama_index.core.schema import TextNode


# -------------------------------
# Paths & constants
# -------------------------------

CTI_DIR = Path("cti")  # repo will be cloned next to your code
CTI_REPO_URL = "https://github.com/mitre/cti.git"
MOBILE_JSON_PATH = CTI_DIR / "mobile-attack" / "mobile-attack.json"

BASE_URL = "https://github.com/mitre/cti/tree/master/mobile-attack"


# -------------------------------
# 1) Ensure repo is present & updated
# -------------------------------

def ensure_cti_repo() -> None:
    """
    Clone the MITRE cti repo if missing, otherwise update it with `git pull`.

    Safe to call on every cron / scraper run:
      - First run: shallow clone (fast)
      - Later runs: only pull latest changes
    """
    if not CTI_DIR.exists():
        # shallow clone is enough
        subprocess.run(
            ["git", "clone", "--depth", "1", CTI_REPO_URL, str(CTI_DIR)],
            check=True,
        )
        print("[MITRE] Repo cloned.")
    else:
        # keep it up-to-date
        subprocess.run(
            ["git", "-C", str(CTI_DIR), "pull", "--ff-only"],
            check=True,
        )
        print("[MITRE] Repo updated (git pull).")


# -------------------------------
# 2) Load mobile-attack JSON
# -------------------------------

def load_mobile_attack_bundle() -> dict:
    """
    Load the mobile-attack STIX bundle from the cloned repo.
    """
    if not MOBILE_JSON_PATH.exists():
        raise FileNotFoundError(
            f"Expected {MOBILE_JSON_PATH} – did the clone/pull succeed?"
        )

    with MOBILE_JSON_PATH.open("r", encoding="utf-8") as f:
        data = json.load(f)

    return data


# -------------------------------
# 3) Helper: build TextNode
# -------------------------------

def make_mitre_node(
    text: str,
    url: str,
    stix_id: str,
    stix_type: str,
    name: str | None = None,
) -> TextNode:
    """
    Create a TextNode compatible with your future ingestion logic.
    """
    metadata = {
        "source_url": url,
        "source_type": "mitre_mobile_attack",
        "stix_id": stix_id,
        "stix_type": stix_type,
    }
    if name:
        metadata["name"] = name

    return TextNode(text=text, metadata=metadata)


# -------------------------------
# 4) Main scraper: STIX objects -> TextNodes
# -------------------------------

def scrape_mitre_mobile() -> List[TextNode]:
    """
    High-level function:
      - ensure cti repo is present & up-to-date
      - load mobile-attack.json
      - turn relevant objects into TextNodes

    Later, you will call this function from your scrape_core / scraper_graph
    and pass the resulting nodes to your embedding + Qdrant ingestion.
    """
    ensure_cti_repo()
    bundle = load_mobile_attack_bundle()

    objects = bundle.get("objects", [])
    nodes: List[TextNode] = []

    for obj in objects:
        stix_id = obj.get("id")
        stix_type = obj.get("type")
        name = obj.get("name") or ""
        desc = obj.get("description") or ""

        if not stix_id or not desc:
            continue

        # Keep only relevant object types (you can tweak this list)
        if stix_type not in {
            "attack-pattern",
            "malware",
            "intrusion-set",
            "tool",
            "course-of-action",
        }:
            continue

        # Optional: you could also pull external references, platforms, etc. later
        # external_refs = obj.get("external_references", [])

        # Use a pseudo-URL per object so each node is uniquely addressable
        url = f"{BASE_URL}#{stix_id}"

        # Simple combined text for now. You can make this more structured later.
        text_parts = [
            name,
            "",
            f"Type: {stix_type}",
            f"STIX ID: {stix_id}",
            "",
            desc,
        ]
        text = "\n".join(p for p in text_parts if p is not None)

        node = make_mitre_node(
            text=text,
            url=url,
            stix_id=stix_id,
            stix_type=stix_type,
            name=name,
        )
        nodes.append(node)

    print(f"[MITRE] Built {len(nodes)} TextNodes from mobile-attack.json")
    return nodes
