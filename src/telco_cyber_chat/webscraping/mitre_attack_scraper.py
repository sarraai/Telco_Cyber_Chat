from __future__ import annotations

import json
import logging
from typing import List, Dict, Any, Optional

import requests

from telco_cyber_chat.webscraping.scrape_core import url_already_ingested

logger = logging.getLogger(__name__)

# -------------------------------
# Constants
# -------------------------------

# Direct raw JSON for the dataset you want to use
MITRE_MOBILE_JSON_URL = (
    "https://raw.githubusercontent.com/mitre/cti/master/mobile-attack/mobile-attack.json"
)

# Fallback base URL if we don't find a better one in external_references
FALLBACK_BASE_URL = (
    "https://github.com/mitre/cti/blob/master/mobile-attack/mobile-attack.json"
)

# Which STIX object types we keep
ALLOWED_TYPES = {
    "attack-pattern",
    "malware",
    "intrusion-set",
    "tool",
    "course-of-action",
}


# -------------------------------
# 1) Fetch bundle from GitHub
# -------------------------------

def fetch_mobile_attack_bundle(timeout: int = 30) -> Dict[str, Any]:
    """
    Download the mobile-attack STIX bundle JSON directly from GitHub.
    """
    logger.info("[MITRE] Downloading mobile-attack bundle from GitHub …")
    resp = requests.get(MITRE_MOBILE_JSON_URL, timeout=timeout)
    resp.raise_for_status()
    data = resp.json()
    if not isinstance(data, dict):
        raise ValueError(f"[MITRE] Unexpected JSON root type: {type(data)}")
    logger.info("[MITRE] Bundle downloaded successfully.")
    return data


# -------------------------------
# 2) Helpers to extract URL/title/description
# -------------------------------

def extract_mitre_url(obj: Dict[str, Any]) -> str:
    """
    Try to get a canonical URL for the STIX object.

    Priority:
      1. external_references[].url (ATT&CK technique/software URL)
      2. Fallback: JSON file URL + #<stix_id>
    """
    stix_id = obj.get("id") or ""
    exrefs = obj.get("external_references") or []
    if isinstance(exrefs, list):
        for ref in exrefs:
            if not isinstance(ref, dict):
                continue
            url = ref.get("url")
            if isinstance(url, str) and url.startswith("http"):
                return url.strip()

    # Fallback pseudo-URL so each object has something stable
    return f"{FALLBACK_BASE_URL}#{stix_id}" if stix_id else FALLBACK_BASE_URL


def build_description(obj: Dict[str, Any]) -> str:
    """
    Build a description text combining name, type, ID, and description.
    """
    stix_id = obj.get("id") or ""
    stix_type = obj.get("type") or ""
    name = (obj.get("name") or "").strip()
    desc = (obj.get("description") or "").strip()

    parts: List[str] = []

    if name:
        parts.append(name)
        parts.append("")  # blank line

    if stix_type:
        parts.append(f"Type: {stix_type}")
    if stix_id:
        parts.append(f"STIX ID: {stix_id}")

    if stix_type or stix_id:
        parts.append("")

    if desc:
        parts.append(desc)

    return "\n".join(parts).strip()


def stix_obj_to_doc(obj: Dict[str, Any]) -> Optional[Dict[str, str]]:
    """
    Convert a single STIX object to a {title, url, description} doc.
    Returns None if the object is not relevant.
    """
    stix_type = obj.get("type")
    if stix_type not in ALLOWED_TYPES:
        return None

    desc = (obj.get("description") or "").strip()
    stix_id = obj.get("id")
    if not stix_id or not desc:
        return None

    name = (obj.get("name") or "").strip()
    url = extract_mitre_url(obj)
    description = build_description(obj)

    title = name or f"MITRE mobile {stix_type} {stix_id}"

    return {
        "title": title,
        "url": url,
        "description": description or title,
    }


# -------------------------------
# 3) Main scraper with Qdrant dedupe
# -------------------------------

def scrape_mitre_mobile(check_qdrant: bool = True) -> List[Dict[str, str]]:
    """
    High-level function:

      - Downloads mobile-attack JSON from GitHub
      - Converts relevant STIX objects into {title, url, description} docs
      - Optional: skips URLs already ingested in Qdrant
      - Logs Seen / Skipped / New counts (Huawei/VARIoT-style)
    """
    logger.info("[MITRE] Starting mobile-attack scrape (check_qdrant=%s)", check_qdrant)

    try:
        bundle = fetch_mobile_attack_bundle()
    except Exception as e:
        logger.error("[MITRE] Failed to download mobile-attack bundle: %s", e)
        return []

    objects = bundle.get("objects") or []
    if not isinstance(objects, list):
        logger.error("[MITRE] Unexpected 'objects' type: %s", type(objects))
        return []

    docs: List[Dict[str, str]] = []

    seen_urls: set[str] = set()
    n_seen = 0
    n_skipped = 0
    n_new = 0

    for obj in objects:
        if not isinstance(obj, dict):
            continue

        doc = stix_obj_to_doc(obj)
        if doc is None:
            continue

        url = doc["url"].strip()
        if not url:
            continue

        # De-duplicate within this run
        if url in seen_urls:
            continue
        seen_urls.add(url)

        n_seen += 1

        # Qdrant dedupe BEFORE indexing
        if check_qdrant:
            try:
                if url_already_ingested(url):
                    n_skipped += 1
                    logger.info("[MITRE] Skipping already-ingested URL: %s", url)
                    continue
            except Exception as e:
                logger.error("[MITRE] Qdrant check failed for %s: %s", url, e)
                # Conservative: continue scraping anyway

        # New doc for this run
        n_new += 1
        docs.append(doc)

    logger.info(
        "[MITRE] Seen=%d, skipped=%d (already in Qdrant), new=%d",
        n_seen, n_skipped, n_new,
    )
    logger.info("[MITRE] Scraped %d new MITRE mobile objects (documents).", len(docs))
    return docs


# -------------------------------
# 4) CLI debug (optional)
# -------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    docs = scrape_mitre_mobile(check_qdrant=False)
    print(f"Total docs: {len(docs)}")
    if docs:
        print("\nExample doc:")
        import pprint
        pprint.pprint(docs[0])
    else:
        print("No documents extracted.")
