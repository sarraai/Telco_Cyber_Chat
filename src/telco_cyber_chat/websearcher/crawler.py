from __future__ import annotations
import re
from collections import deque
from typing import Iterable, List, Optional, Set, Tuple
from urllib.parse import urljoin, urlparse

import requests
from bs4 import BeautifulSoup

PDF_RE = re.compile(r"\.pdf(\?|#|$)", re.IGNORECASE)

def _in_allowed_domains(url: str, allowed: Optional[List[str]]) -> bool:
    if not allowed:
        return True
    host = urlparse(url).netloc.lower()
    return any(host == d.lower() or host.endswith("." + d.lower()) for d in allowed)

def extract_links(base_url: str, html: str) -> Tuple[Set[str], Set[str]]:
    soup = BeautifulSoup(html, "lxml")
    page_links: Set[str] = set()
    pdf_links: Set[str] = set()

    for a in soup.select("a[href]"):
        href = a.get("href", "").strip()
        if not href:
            continue
        u = urljoin(base_url, href)
        if PDF_RE.search(u):
            pdf_links.add(u)
        else:
            page_links.add(u)

    return page_links, pdf_links

def discover_pdf_urls(
    seed_urls: Iterable[str],
    allowed_domains: Optional[List[str]] = None,
    max_pages: int = 50,
    max_depth: int = 2,
    timeout_s: int = 25,
) -> List[str]:
    visited_pages: Set[str] = set()
    found_pdfs: Set[str] = set()

    q = deque([(u, 0) for u in seed_urls])

    sess = requests.Session()
    headers = {"User-Agent": "telco-cyber-chat-websearcher/1.0"}

    while q and len(visited_pages) < max_pages:
        url, depth = q.popleft()
        if url in visited_pages:
            continue
        if depth > max_depth:
            continue
        if not _in_allowed_domains(url, allowed_domains):
            continue

        visited_pages.add(url)

        try:
            r = sess.get(url, timeout=timeout_s, headers=headers)
            if r.status_code >= 400:
                continue
            ctype = (r.headers.get("Content-Type") or "").lower()
            if "text/html" not in ctype:
                continue

            page_links, pdf_links = extract_links(url, r.text)
            found_pdfs |= {p for p in pdf_links if _in_allowed_domains(p, allowed_domains)}

            for nxt in page_links:
                if nxt not in visited_pages and _in_allowed_domains(nxt, allowed_domains):
                    q.append((nxt, depth + 1))

        except requests.RequestException:
            continue

    return sorted(found_pdfs)
