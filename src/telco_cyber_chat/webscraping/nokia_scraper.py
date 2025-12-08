from typing import List, Dict

from telco_cyber_chat.webscraping.scrape_core import url_already_ingested

def advisory_dict_to_document(url: str, adv: Dict) -> Dict:
    """
    Convert a parsed Nokia advisory dict (from extract_one_advisory)
    into a simple document with:
      - url
      - title
      - text (merged description with vuln details)
      - source_type
    """
    title = adv.get("title") or "Nokia Product Security Advisory"

    # Build a single rich description that merges everything you want
    desc_parts = []

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

    # Full text = title + merged description
    if merged_description:
        full_text = f"{title}\n\n{merged_description}"
    else:
        full_text = title

    return {
        "url": url,
        "title": title,
        "text": full_text,
        "source_type": "nokia_psa",
    }


def fetch_nokia_advisory_urls() -> List[str]:
    """
    Discover all Nokia Product Security Advisory URLs using sitemap + index crawl.
    Uses your existing discovery helpers.
    """
    session = requests.Session()
    urls = set()
    if USE_SITEMAP:
        urls.update(harvest_from_sitemap(session))
    urls.update(crawl_all_advisory_links(session))
    return sorted(urls)


def scrape_nokia(check_qdrant: bool = True) -> List[Dict]:
    """
    Main scraping entrypoint for Nokia:

      - discovers advisory URLs
      - (optionally) skips URLs already stored in Qdrant
      - fetches + parses each page into a dict using extract_one_advisory(...)
      - reshapes it into a compact document:
            {url, title, text, source_type}

    No TextNodes or embeddings here – that happens later in scrape_core.
    """
    session = requests.Session()
    urls = fetch_nokia_advisory_urls()
    docs: List[Dict] = []

    for u in urls:
        # 🔁 Avoid duplicates in Qdrant if desired
        if check_qdrant and url_already_ingested(u):
            continue

        try:
            resp = get(u, session=session)
            adv = extract_one_advisory(resp.text)
        except Exception as e:
            if VERBOSE:
                print(f"[error] {u}: {e}")
            continue

        doc = advisory_dict_to_document(u, adv)
        docs.append(doc)

    print(f"[NOKIA] Scraped {len(docs)} advisories (documents).")
    return docs
