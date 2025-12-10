import os
import re
import logging
from typing import List, Dict, Optional

import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

from telco_cyber_chat.webscraping.scrape_core import url_already_ingested

logger = logging.getLogger(__name__)

# ----------------------------------------
# Global config flags
# ----------------------------------------

# Nokia PSA index page
NOKIA_PSA_INDEX_URL = os.getenv(
    "NOKIA_PSA_INDEX_URL",
    "https://www.nokia.com/about-us/security-and-privacy/product-security-advisory/",
)

# HTTP config
TIMEOUT = int(os.getenv("NOKIA_HTTP_TIMEOUT", "30"))
HTML_HEADERS = {
    "User-Agent": "Mozilla/5.0 (TelcoCyberChatBot/1.0)",
    "Accept-Language": "en-US,en;q=0.9",
}

# Verbose logging toggle
VERBOSE: bool = os.getenv("NOKIA_VERBOSE", "0").lower() in (
    "1",
    "true",
    "yes",
)


# ----------------------------------------
# Small helpers
# ----------------------------------------

def _clean(text: Optional[str]) -> str:
    if not text:
        return ""
    return " ".join(text.replace("\xa0", " ").split())


def get(url: str, session: Optional[requests.Session] = None) -> requests.Response:
    """
    Thin wrapper around requests.get with headers + timeout.
    Raises for non-2xx responses.
    """
    sess = session or requests.Session()
    resp = sess.get(url, headers=HTML_HEADERS, timeout=TIMEOUT)
    resp.raise_for_status()
    return resp


# ----------------------------------------
# Nokia advisory parsing
# ----------------------------------------

def extract_one_advisory(html: str) -> Dict:
    """
    Parse a single Nokia Product Security Advisory page into a dict.

    The Nokia PSA pages typically have:
      - H1 title
      - A table with keys like "Vulnerability type", "CVSS vector", "CVSS score"
      - Sections such as "Description", "Affected products and versions",
        "Mitigation plan", etc.
    This parser is intentionally heuristic but robust enough for the
    advisory_dict_to_document pipeline.
    """
    soup = BeautifulSoup(html, "lxml")

    adv: Dict[str, any] = {
        "title": "",
        "description": "",
        "vulnerability_type": "",
        "cvss_score": "",
        "cvss_vector": "",
        "affected_products_and_versions": [],
        "cves": [],
        "mitigation_plan": "",
    }

    # --- Title ---
    h1 = soup.find("h1")
    if h1:
        adv["title"] = _clean(h1.get_text())
    else:
        # Fallback to <title>
        if soup.title:
            adv["title"] = _clean(soup.title.get_text())

    # --- Key/value table (Vulnerability type, CVSS, etc.) ---
    for table in soup.find_all("table"):
        for tr in table.find_all("tr"):
            cells = tr.find_all(["th", "td"])
            if len(cells) < 2:
                continue
            key = _clean(cells[0].get_text()).lower()
            val = _clean(cells[1].get_text())
            if not key or not val:
                continue

            if "vulnerability type" in key:
                adv["vulnerability_type"] = val
            elif "cvss vector" in key:
                adv["cvss_vector"] = val
            elif "cvss score" in key:
                adv["cvss_score"] = val
            elif "mitigation plan" in key:
                adv["mitigation_plan"] = val

    # --- Description section ---
    desc_parts: List[str] = []
    for heading in soup.find_all(["h2", "h3", "h4"]):
        if "description" in _clean(heading.get_text()).lower():
            for sib in heading.find_next_siblings():
                if getattr(sib, "name", None) in {"h2", "h3", "h4"}:
                    break
                if getattr(sib, "name", None) in {"p", "li"}:
                    txt = _clean(sib.get_text())
                    if txt:
                        desc_parts.append(txt)
            break
    adv["description"] = "\n".join(desc_parts)

    # --- Affected products and versions ---
    affected: List[str] = []
    affected_heading = None
    for h in soup.find_all(["h2", "h3", "h4"]):
        if "affected products and versions" in _clean(h.get_text()).lower():
            affected_heading = h
            break

    if affected_heading is not None:
        for sib in affected_heading.find_next_siblings():
            if getattr(sib, "name", None) in {"h2", "h3", "h4"}:
                break
            if getattr(sib, "name", None) in {"ul", "ol"}:
                for li in sib.find_all("li"):
                    txt = _clean(li.get_text())
                    if txt:
                        affected.append(txt)
            elif getattr(sib, "name", None) == "p":
                txt = _clean(sib.get_text())
                if txt:
                    affected.append(txt)

    adv["affected_products_and_versions"] = affected

    # --- Mitigation plan section (fallback if not captured in table) ---
    if not adv.get("mitigation_plan"):
        mit_parts: List[str] = []
        mit_heading = None
        for h in soup.find_all(["h2", "h3", "h4"]):
            if "mitigation plan" in _clean(h.get_text()).lower():
                mit_heading = h
                break
        if mit_heading is not None:
            for sib in mit_heading.find_next_siblings():
                if getattr(sib, "name", None) in {"h2", "h3", "h4"}:
                    break
                if getattr(sib, "name", None) in {"p", "li"}:
                    txt = _clean(sib.get_text())
                    if txt:
                        mit_parts.append(txt)
        if mit_parts:
            adv["mitigation_plan"] = "\n".join(mit_parts)

    # --- Collect CVEs from page text ---
    cve_pattern = re.compile(r"CVE-\d{4}-\d+", re.IGNORECASE)
    page_text = soup.get_text(" ", strip=True)
    cves = sorted({c.upper() for c in cve_pattern.findall(page_text)})
    adv["cves"] = cves

    return adv


# ----------------------------------------
# Nokia advisory discovery
# ----------------------------------------

def crawl_all_advisory_links(session: Optional[requests.Session] = None) -> List[str]:
    """
    Discover all Nokia PSA advisory URLs by crawling the index page
    https://www.nokia.com/about-us/security-and-privacy/product-security-advisory/

    We look for links that contain "product-security-advisory" AND "cve-".
    """
    sess = session or requests.Session()
    logger.info("[NOKIA] Fetching PSA index: %s", NOKIA_PSA_INDEX_URL)

    resp = get(NOKIA_PSA_INDEX_URL, session=sess)
    soup = BeautifulSoup(resp.text, "lxml")

    urls: set[str] = set()

    for a in soup.find_all("a", href=True):
        href = a["href"]
        if not href:
            continue
        href = href.strip()
        # Heuristic: advisory detail pages contain this pattern
        if (
            "product-security-advisory" in href
            and "cve-" in href.lower()
        ):
            full = urljoin(NOKIA_PSA_INDEX_URL, href)
            urls.add(full)

    logger.info("[NOKIA] Discovered %d advisory URLs from index.", len(urls))
    return sorted(urls)


def advisory_dict_to_document(url: str, adv: Dict) -> Dict[str, str]:
    """
    Convert a parsed Nokia advisory dict (from extract_one_advisory)
    into a simple document with:
      - url
      - title
      - description (merged description with vuln details)
    """
    title = adv.get("title") or "Nokia Product Security Advisory"

    # Build a single rich description that merges everything you want
    desc_parts: List[str] = []

    # Vulnerability type
    if adv.get("vulnerability_type"):
        desc_parts.append(f"Vulnerability type: {adv['vulnerability_type']}")

    # CVSS (vector + score)
    cvss_bits = []
    if adv.get("cvss_score"):
        cvss_bits.append(f"score {adv['cvss_score']}")
    if adv.get("cvss_vector"):
        cvss_bits.append(f"vector {adv['cvss_vector']}")
    if cvss_bits:
        desc_parts.append("CVSS " + ", ".join(cvss_bits))

    # Affected products and versions
    if adv.get("affected_products_and_versions"):
        affected_str = "; ".join(adv["affected_products_and_versions"])
        desc_parts.append("Affected products and versions: " + affected_str)

    # CVEs
    if adv.get("cves"):
        desc_parts.append("Related CVEs: " + ", ".join(adv["cves"]))

    # Mitigation
    if adv.get("mitigation_plan"):
        desc_parts.append("Mitigation plan: " + adv["mitigation_plan"])

    # Original description text
    if adv.get("description"):
        desc_parts.append("Technical description: " + adv["description"])

    # Final merged description
    merged_description = "\n".join(desc_parts).strip()

    return {
        "url": url,
        "title": title,  # unified key name so ingest_pipeline + node_builder can use it
        "description": merged_description or title,
    }


def fetch_nokia_advisory_urls(
    session: Optional[requests.Session] = None,
) -> List[str]:
    """
    Discover all Nokia Product Security Advisory URLs.
    Currently we only use the HTML index crawler (crawl_all_advisory_links).
    """
    sess = session or requests.Session()
    try:
        urls = crawl_all_advisory_links(sess)
    except Exception as e:
        logger.error("[NOKIA] Failed to crawl advisory index: %s", e)
        return []

    return urls


# ----------------------------------------
# Public entrypoint (used by your ingest pipeline)
# ----------------------------------------

def scrape_nokia(check_qdrant: bool = True) -> List[Dict[str, str]]:
    """
    Main scraping entrypoint for Nokia:

    - discovers advisory URLs
    - (optionally) skips URLs already stored in Qdrant
    - fetches + parses each page into a dict using extract_one_advisory(...)
    - reshapes it into a compact document: {url, title, description}

    No TextNodes or embeddings here – that happens later in ingest_pipeline.
    """
    logger.info("[NOKIA] Starting Nokia advisory scrape (check_qdrant=%s)", check_qdrant)

    session = requests.Session()
    urls = fetch_nokia_advisory_urls(session=session)
    docs: List[Dict[str, str]] = []

    logger.info("[NOKIA] Total URLs discovered: %d", len(urls))

    for u in urls:
        # ✅ Qdrant dedupe BEFORE heavy page fetch/parse
        if check_qdrant and url_already_ingested(u, filter={"source": "nokia"}):
            logger.info("[NOKIA] Skipping already-ingested URL: %s", u)
            continue

        try:
            resp = get(u, session=session)
            adv = extract_one_advisory(resp.text)
        except Exception as e:
            if VERBOSE:
                logger.warning("[NOKIA] Error scraping %s: %s", u, e)
            else:
                logger.debug("[NOKIA] Error scraping %s: %s", u, e)
            continue

        doc = advisory_dict_to_document(u, adv)
        docs.append(doc)

    logger.info("[NOKIA] Scraped %d advisories (documents).", len(docs))
    return docs
