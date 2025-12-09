# pip install requests beautifulsoup4 lxml

import sys
import time
import json
import re
import logging
import requests
import xml.etree.ElementTree as ET
from typing import Dict, List, Optional
from datetime import datetime, timezone
from urllib.parse import urlparse
from bs4 import BeautifulSoup, Tag
from pathlib import Path

from telco_cyber_chat.webscraping.scrape_core import url_already_ingested

logger = logging.getLogger(__name__)

# ===== OAuth & API =====
CLIENT_ID = "ucprtmc76uqkhda36q26z8f9"      # <- your Cisco client_id
CLIENT_SECRET = "gNjqQMquNyejxerJ2SYFPdqX"  # <- your Cisco client_secret
TOKEN_URL = "https://id.cisco.com/oauth2/default/v1/token"
BASE_URL  = "https://apix.cisco.com/security/advisories/v2"

# ===== Range / Colab defaults =====
START_YEAR = 2024                             # <-- from 2024 onward
END_YEAR   = datetime.now(timezone.utc).year

# ===== Output =====
OUT_JSON = f"cisco_advisories_{START_YEAR}_{END_YEAR}.json"

# ===== Networking =====
TIMEOUT   = 60
PAUSE_SEC = 0.25
HTML_HEADERS = {"User-Agent": "Mozilla/5.0", "Accept-Language": "en-US,en;q=0.9"}

# Canonical pages
MIRROR_BASE = "https://www.cisco.com/c/en/us/support/docs/csa/"
CANON_BASE  = "https://sec.cloudapps.cisco.com/security/center/content/CiscoSecurityAdvisory/"

# ===== Utils =====
def _clean(s: Optional[str]) -> str:
    return " ".join((s or "").replace("\xa0", " ").split())


def _iso_date_only(s: Optional[str]) -> str:
    if not s:
        return ""
    try:
        s2 = s.strip()
        if "T" in s2:
            return datetime.fromisoformat(s2.replace("Z", "+00:00")).date().isoformat()
        if len(s2) >= 10 and s2[4] == "-" and s2[7] == "-":
            return s2[:10]
    except Exception:
        pass
    return s


def _parse_dt(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    try:
        if "T" in s:
            return datetime.fromisoformat(s.replace("Z", "+00:00"))
        return datetime.fromisoformat(s + "T00:00:00+00:00")
    except Exception:
        return None


def _is_valid_http_url(u: Optional[str]) -> bool:
    if not u:
        return False
    up = u.strip().upper()
    if up in {"NA", "N/A", "-", "NONE"}:
        return False
    try:
        p = urlparse(u)
        return p.scheme in ("http", "https") and bool(p.netloc)
    except Exception:
        return False


# ===== HTTP helpers =====
def get_token(client_id: str, client_secret: str) -> str:
    r = requests.post(
        TOKEN_URL,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=TIMEOUT,
    )
    if not r.ok:
        raise SystemExit(f"Token request failed: HTTP {r.status_code}\n{r.text}")
    tok = r.json().get("access_token")
    if not tok:
        raise SystemExit("No access_token in token response")
    return tok


def get_json(endpoint: str, token: str, params: Optional[Dict] = None) -> Dict:
    url = f"{BASE_URL}/{endpoint.lstrip('/')}"
    r = requests.get(
        url,
        headers={
            "Accept": "application/json",
            "Authorization": f"Bearer {token}",
        },
        params=params or {},
        timeout=TIMEOUT,
    )
    if not r.ok:
        raise RuntimeError(f"GET {url} failed: HTTP {r.status_code}\n{r.text}")
    return r.json()


def get_text(url: str, token: str) -> Optional[str]:
    try:
        r = requests.get(
            url,
            headers={"Accept": "*/*", "Authorization": f"Bearer {token}"},
            timeout=TIMEOUT,
        )
        return r.text if r.ok else None
    except Exception:
        return None


# ===== Fetch advisory =====
def fetch_advisory_by_id(token: str, advisory_id: str) -> Optional[Dict]:
    try:
        data = get_json(f"advisory/{advisory_id}", token)
        if isinstance(data, dict) and "advisory" in data:
            return data["advisory"]
        if isinstance(data, dict) and (data.get("advisoryId") or data.get("id")):
            return data
    except Exception:
        pass
    # Fallback: scan recent years
    for y in range(END_YEAR, START_YEAR - 1, -1):
        try:
            for adv in get_json(f"year/{y}", token).get("advisories", []):
                aid = adv.get("advisoryId") or adv.get("id") or ""
                if aid == advisory_id:
                    return adv
        except Exception:
            continue
        time.sleep(PAUSE_SEC)
    return None


def fetch_latest_advisory(token: str, start_year: int, end_year: int) -> Optional[Dict]:
    best, best_dt = None, None
    for y in range(end_year, start_year - 1, -1):
        try:
            items = get_json(f"year/{y}", token).get("advisories", [])
        except Exception:
            continue
        for adv in items:
            fp = adv.get("firstPublished")
            lu = adv.get("lastUpdated")
            dt = _parse_dt(lu) or _parse_dt(fp)
            if dt and (best_dt is None or dt > best_dt):
                best_dt, best = dt, adv
        time.sleep(PAUSE_SEC)
    return best


# ===== CVRF parsing =====
def _lname(tag: str) -> str:
    return tag.split("}", 1)[-1] if "}" in tag else tag


def _iter_by_name(node: ET.Element, name: str):
    for el in node.iter():
        if _lname(el.tag) == name:
            yield el


def _get_all_text(el: Optional[ET.Element]) -> str:
    return "" if el is None else _clean("".join(el.itertext()))


def parse_cvrf(xml_text: str) -> Dict:
    out = {
        "summary": "",
        "source": "",
        "workarounds": "",
        "revision_history": [],
        "affected_products": "",
        "vulnerable_products": [],
        "products_confirmed_not_vulnerable": [],
        "fixed_software": [],
        "cvss_max_base": None,
    }
    try:
        root = ET.fromstring(xml_text)
    except Exception:
        return out

    # Notes
    notes_by_title = {}
    for note in _iter_by_name(root, "Note"):
        title = _clean(note.attrib.get("Title", ""))
        if title:
            notes_by_title[title.lower()] = _get_all_text(note)

    def first_note(*contains: str) -> str:
        for k, v in notes_by_title.items():
            if all(x.lower() in k for x in contains):
                return v
        return ""

    out["summary"] = first_note("summary")
    out["source"] = first_note("source", "credit", "acknowledg")

    # Workarounds (+ Remediation mit/wa)
    wa = [
        v for k, v in notes_by_title.items()
        if "workaround" in k or "mitigation" in k
    ]
    for rem in _iter_by_name(root, "Remediation"):
        rtype = _clean(rem.attrib.get("Type", ""))
        if "workaround" in rtype.lower() or "mitigation" in rtype.lower():
            wa.append(_get_all_text(rem))
    if wa:
        out["workarounds"] = "\n\n".join(
            dict.fromkeys([w.strip() for w in wa if w.strip()])
        )

    # Revision history
    for rh in _iter_by_name(root, "Revision"):
        num = _get_all_text(
            next((c for c in rh if _lname(c.tag) == "Number"), None)
        )
        date = _get_all_text(
            next((c for c in rh if _lname(c.tag) == "Date"), None)
        )
        desc = _get_all_text(
            next((c for c in rh if _lname(c.tag) == "Description"), None)
        )
        out["revision_history"].append(
            {
                "version": num,
                "date": _iso_date_only(date),
                "description": desc,
            }
        )

    # Product map
    prod_map = {}
    for fp in _iter_by_name(root, "FullProductName"):
        pid = (
            fp.attrib.get("ProductID")
            or fp.attrib.get("ProductIDRef")
            or fp.attrib.get("ProductIDref")
            or ""
        )
        if pid:
            prod_map[pid] = _get_all_text(fp)

    # Vulnerable / not-vulnerable
    vuln_ids, not_vuln_ids = set(), set()
    for pstatus in _iter_by_name(root, "ProductStatuses"):
        for status in _iter_by_name(pstatus, "Status"):
            stype = _clean(status.attrib.get("Type", ""))
            ids = [
                _clean(el.attrib.get("ProductID", "")) or _clean(el.text or "")
                for el in _iter_by_name(status, "ProductID")
            ]
            if "known affected" in stype.lower() or "vulnerable" in stype.lower():
                vuln_ids.update(ids)
            if "known not affected" in stype.lower() or "not vulnerable" in stype.lower():
                not_vuln_ids.update(ids)

    out["vulnerable_products"] = sorted(
        {prod_map.get(pid, pid) for pid in vuln_ids if pid}
    )
    out["products_confirmed_not_vulnerable"] = sorted(
        {prod_map.get(pid, pid) for pid in not_vuln_ids if pid}
    )

    # Fixed Software (VendorFix)
    fixed_rows = []
    for rem in _iter_by_name(root, "Remediation"):
        rtype = _clean(rem.attrib.get("Type", ""))
        if (
            "vendor fix" in rtype.lower()
            or "vendorfix" in rtype.lower()
            or rtype.lower() == "fix"
        ):
            desc = _get_all_text(rem)
            prod_ids = [
                _clean(el.attrib.get("ProductID", "")) or _clean(el.text or "")
                for el in _iter_by_name(rem, "ProductID")
            ]
            if not prod_ids:
                fixed_rows.append({"product": "", "fix": desc})
            else:
                for pid in prod_ids:
                    fixed_rows.append({"product": prod_map.get(pid, pid), "fix": desc})

    # Dedup
    seen, dedup = set(), []
    for r in fixed_rows:
        key = (r.get("product", ""), r.get("fix", ""))
        if key not in seen:
            seen.add(key)
            dedup.append(r)
    out["fixed_software"] = dedup

    # CVSS (max BaseScore)
    cvss_vals = []
    for b in _iter_by_name(root, "BaseScore"):
        try:
            cvss_vals.append(float(_get_all_text(b)))
        except Exception:
            pass
    if cvss_vals:
        out["cvss_max_base"] = max(cvss_vals)

    return out


# ===== HTML helpers (for fallbacks) =====
def _txt(node: Tag) -> str:
    return _clean(node.get_text(" ", strip=True)) if isinstance(node, Tag) else ""


def _parse_html_table(table: Tag) -> Optional[Dict]:
    try:
        headers = []
        thead = table.find("thead")
        if thead:
            headers = [_txt(th) for th in thead.find_all("th")]
        if not headers:
            fr = table.find("tr")
            if fr:
                headers = [_txt(th) for th in fr.find_all(["th", "td"])]
        headers = [h for h in headers if h]

        body = table.find_all("tr")
        if body and headers:
            first = [_txt(c) for c in body[0].find_all(["th", "td"])]
            if first and all(c in headers for c in first):
                body = body[1:]

        rows = []
        for tr in body:
            cells = [_txt(td) for td in tr.find_all(["td", "th"])]
            if not cells:
                continue
            if headers:
                if len(cells) < len(headers):
                    cells += [""] * (len(headers) - len(cells))
                elif len(cells) > len(headers):
                    cells = cells[:len(headers)]
            rows.append(cells)

        records = []
        if headers:
            for r in rows:
                records.append(
                    {headers[i]: (r[i] if i < len(headers) else "") for i in range(len(headers))}
                )

        if headers or rows:
            return {"columns": headers, "rows": rows, "records": records}
    except Exception:
        return None
    return None


def html_workarounds_fallback(advisory_id: str, publication_url: Optional[str]) -> str:
    candidates = [
        f"{MIRROR_BASE}{advisory_id}.html" if advisory_id else None,
        f"{CANON_BASE}{advisory_id}" if advisory_id else None,
        publication_url or None,
    ]
    keywords = ("workaround", "work-arounds", "work around", "mitigation", "mitigations")
    for u in filter(None, candidates):
        try:
            r = requests.get(u, headers=HTML_HEADERS, timeout=40)
            if not r.ok:
                continue
            soup = BeautifulSoup(r.text, "lxml")

            for anchor_id in ("workaroundsfield", "workaroundfield", "mitigationfield"):
                cont = soup.find(id=anchor_id)
                if cont:
                    parts = [_txt(tag) for tag in cont.find_all(["p", "li"]) if _txt(tag)]
                    if parts:
                        return "\n\n".join(dict.fromkeys(parts))

            heading = None
            for h in soup.find_all(["h2", "h3", "h4", "h5"]):
                ht = _txt(h).lower()
                if any(k in ht for k in keywords):
                    heading = h
                    break
            if not heading:
                continue

            chunks = []
            for sib in heading.next_siblings:
                if isinstance(sib, Tag) and sib.name in {"h2", "h3", "h4"}:
                    break
                if isinstance(sib, Tag) and sib.name in {"p", "li", "div"}:
                    t = _txt(sib)
                    if t:
                        chunks.append(t)
            if chunks:
                return "\n\n".join(dict.fromkeys(chunks))
        except Exception:
            continue
    return ""


def html_affected_products_fallback(
    advisory_id: str, publication_url: Optional[str]
) -> Dict[str, str]:
    out = {
        "affected_products": "",
        "vulnerable_products": "",
        "products_confirmed_not_vulnerable": "",
    }
    candidates = [
        f"{MIRROR_BASE}{advisory_id}.html" if advisory_id else None,
        f"{CANON_BASE}{advisory_id}" if advisory_id else None,
        publication_url or None,
    ]

    def collect(container: Tag) -> str:
        parts = [_txt(tag) for tag in container.find_all(["p", "li"]) if _txt(tag)]
        return "\n\n".join(dict.fromkeys(parts))

    for u in filter(None, candidates):
        try:
            r = requests.get(u, headers=HTML_HEADERS, timeout=40)
            if not r.ok:
                continue
            soup = BeautifulSoup(r.text, "lxml")

            aff = soup.find(id="affectfield")
            if aff:
                out["affected_products"] = collect(aff)
            vp = soup.find(id="vulnerableproducts")
            if vp:
                out["vulnerable_products"] = collect(vp)
            nv = soup.find(id="productsconfirmednotvulnerable")
            if nv:
                out["products_confirmed_not_vulnerable"] = collect(nv)

            if not out["affected_products"]:
                heading = None
                for h in soup.find_all(["h2", "h3", "h4"]):
                    if "affected products" in _txt(h).lower():
                        heading = h
                        break
                if heading:
                    buf = soup.new_tag("div")
                    for sib in heading.next_siblings:
                        if isinstance(sib, Tag) and sib.name in {"h2", "h3"}:
                            break
                        if isinstance(sib, Tag):
                            buf.append(sib)
                    out["affected_products"] = collect(buf)

            if any(out.values()):
                return out
        except Exception:
            continue
    return out


# HTML fallback for Source (credits/acknowledgments)
def html_source_fallback(advisory_id: str, publication_url: Optional[str]) -> str:
    candidates = [
        f"{MIRROR_BASE}{advisory_id}.html" if advisory_id else None,
        f"{CANON_BASE}{advisory_id}" if advisory_id else None,
        publication_url or None,
    ]
    for u in filter(None, candidates):
        try:
            r = requests.get(u, headers=HTML_HEADERS, timeout=40)
            if not r.ok:
                continue
            soup = BeautifulSoup(r.text, "lxml")

            cont = soup.find(id="sourcefield")
            if cont:
                parts = [_txt(tag) for tag in cont.find_all(["p", "li"]) if _txt(tag)]
                if parts:
                    return "\n\n".join(dict.fromkeys(parts))

            heading = None
            for h in soup.find_all(["h2", "h3", "h4", "h5"]):
                if "source" in _txt(h).lower():
                    heading = h
                    break
            if heading:
                chunks = []
                for sib in heading.next_siblings:
                    if isinstance(sib, Tag) and sib.name in {"h2", "h3", "h4"}:
                        break
                    if isinstance(sib, Tag) and sib.name in {"p", "li", "div"}:
                        t = _txt(sib)
                        if t:
                            chunks.append(t)
                if chunks:
                    return "\n\n".join(dict.fromkeys(chunks))
        except Exception:
            continue
    return ""


# HTML fallback for Version/Status and Cisco Bug IDs
def html_version_status_and_bugs(
    advisory_id: str, publication_url: Optional[str]
) -> Dict[str, Optional[str]]:
    result = {"version": None, "version_status": None, "bug_ids": []}
    candidates = [
        f"{MIRROR_BASE}{advisory_id}.html" if advisory_id else None,
        f"{CANON_BASE}{advisory_id}" if advisory_id else None,
        publication_url or None,
    ]
    bug_re = re.compile(r"\bCSC[a-z]\w+\b", re.I)
    version_re = re.compile(r"\bVersion\s*([0-9]+(?:\.[0-9]+)*)\s*:", re.I)

    for u in filter(None, candidates):
        try:
            r = requests.get(u, headers=HTML_HEADERS, timeout=40)
            if not r.ok:
                continue
            soup = BeautifulSoup(r.text, "lxml")

            rows = soup.select(
                ".pubheaderrow, .divPaddingTen, .ud-headerrow, .headerRow"
            )
            if rows:
                for row in rows:
                    label = row.find(
                        class_=re.compile(
                            r"ud-divHeaderLabelSpacing|headerLabel|label", re.I
                        )
                    )
                    if not label:
                        labs = row.find_all("div", recursive=False)
                        label = labs[0] if labs else None
                    label_txt = _txt(label).lower() if label else ""
                    if "version" in label_txt:
                        val = row.find(
                            class_=re.compile(r"divLabelContent|value|content", re.I)
                        )
                        ver_text = _txt(label)
                        val_text = _txt(val)
                        m = version_re.search(ver_text) or version_re.search(val_text)
                        if m:
                            result["version"] = m.group(1)
                        if val:
                            link = val.find("a")
                            if link:
                                st = _txt(link)
                                if st:
                                    result["version_status"] = st
                            else:
                                vt = _txt(val)
                                if "final" in vt.lower():
                                    result["version_status"] = "Final"
                                elif "interim" in vt.lower():
                                    result["version_status"] = "Interim"

            if not result["version"]:
                m2 = version_re.search(soup.get_text(" ", strip=True))
                if m2:
                    result["version"] = m2.group(1)

            ddts = soup.find(class_=re.compile(r"ddtsList|bugList|bugs", re.I))
            if ddts:
                for a in ddts.find_all("a", href=True):
                    m = bug_re.search(a.get_text())
                    if m:
                        result["bug_ids"].append(m.group(0).upper())

            if not result["bug_ids"]:
                label_candidates = []
                for div in soup.find_all(["div", "span", "p", "h3", "h4"]):
                    t = _txt(div)
                    if not t:
                        continue
                    if "cisco bug id" in t.lower():
                        label_candidates.append(div)
                for lab in label_candidates:
                    container = lab.parent if lab.parent else lab
                    for a in container.find_all("a", href=True):
                        m = bug_re.search(a.get_text())
                        if m:
                            result["bug_ids"].append(m.group(0).upper())
                    for sib in lab.next_siblings:
                        if isinstance(sib, Tag):
                            for a in sib.find_all("a", href=True):
                                m = bug_re.search(a.get_text())
                                if m:
                                    result["bug_ids"].append(m.group(0).upper())
                result["bug_ids"] = sorted(set(result["bug_ids"]))

            if not result["bug_ids"]:
                all_text = soup.get_text(" ", strip=True)
                result["bug_ids"] = sorted(
                    {m.group(0).upper() for m in bug_re.finditer(all_text)}
                )

            if result["version"] or result["version_status"] or result["bug_ids"]:
                return result

        except Exception:
            continue

    return result


def html_fixed_software_fallback(
    advisory_id: str, publication_url: Optional[str]
) -> Dict:
    result = {"text_rows": [], "tables": []}
    candidates = [
        f"{MIRROR_BASE}{advisory_id}.html" if advisory_id else None,
        f"{CANON_BASE}{advisory_id}" if advisory_id else None,
        publication_url or None,
    ]
    for u in filter(None, candidates):
        try:
            r = requests.get(u, headers=HTML_HEADERS, timeout=40)
            if not r.ok:
                continue
            soup = BeautifulSoup(r.text, "lxml")

            sections = []
            cont = soup.find(id="fixedsoftfield")
            if cont:
                sections.append(cont)

            if not sections:
                anchor = soup.find("a", attrs={"name": "fs"})
                if anchor:
                    buf = []
                    for sib in anchor.parent.next_siblings:
                        if isinstance(sib, Tag) and sib.name in {"h2", "h3"}:
                            break
                        if isinstance(sib, Tag):
                            buf.append(sib)
                    if buf:
                        fake = soup.new_tag("div")
                        for b in buf:
                            fake.append(b)
                        sections.append(fake)

            if not sections:
                head = None
                for h in soup.find_all(["h2", "h3"]):
                    if "fixed software" in _txt(h).lower():
                        head = h
                        break
                if head:
                    buf = []
                    for sib in head.next_siblings:
                        if isinstance(sib, Tag) and sib.name in {"h2", "h3"}:
                            break
                        if isinstance(sib, Tag):
                            buf.append(sib)
                    fake = soup.new_tag("div")
                    for b in buf:
                        fake.append(b)
                    sections.append(fake)

            text_rows = []
            current_product = None
            for sec in sections:
                for h in sec.find_all(["h3", "h4"]):
                    current_product = _txt(h) or current_product
                for li in sec.find_all("li"):
                    line = _txt(li)
                    if line:
                        text_rows.append(
                            {
                                "product": current_product or "Fixed Software",
                                "fix": line,
                            }
                        )
                for p in sec.find_all("p"):
                    line = _txt(p)
                    if line:
                        text_rows.append(
                            {
                                "product": current_product or "Fixed Software",
                                "fix": line,
                            }
                        )
                for tbl in sec.find_all("table"):
                    parsed = _parse_html_table(tbl)
                    if parsed and (parsed["columns"] or parsed["rows"]):
                        result["tables"].append(parsed)
                        for row in parsed.get("rows", []) or []:
                            if len(row) >= 2:
                                text_rows.append({"product": row[0], "fix": row[1]})
                            elif len(row) == 1:
                                text_rows.append(
                                    {
                                        "product": current_product
                                        or "Fixed Software Table",
                                        "fix": row[0],
                                    }
                                )

            # Explicit "Fixed Releases" table
            h3_fr = None
            for h3 in soup.find_all("h3"):
                if "fixed releases" in _txt(h3).lower():
                    h3_fr = h3
                    break
            if h3_fr:
                tbl = h3_fr.find_next("table")
                if tbl:
                    parsed = _parse_html_table(tbl)
                    if parsed and (parsed["columns"] or parsed["rows"]):
                        if parsed not in result["tables"]:
                            result["tables"].append(parsed)
                        for row in parsed.get("rows", []) or []:
                            if len(row) >= 2:
                                text_rows.append({"product": row[0], "fix": row[1]})
                            elif len(row) == 1:
                                text_rows.append(
                                    {
                                        "product": "Fixed Releases",
                                        "fix": row[0],
                                    }
                                )

            # de-dup rows
            seen, dedup = set(), []
            for r in text_rows:
                key = (r.get("product") or "", r.get("fix") or "")
                if r.get("fix") and key not in seen:
                    seen.add(key)
                    dedup.append(
                        {
                            "product": r.get("product") or None,
                            "fix": r["fix"],
                        }
                    )
            result["text_rows"] = dedup

            if result["text_rows"] or result["tables"]:
                return result
        except Exception:
            continue
    return result


def html_cvss_fallback(advisory_id: str, publication_url: Optional[str]) -> Optional[float]:
    candidates = [f"{CANON_BASE}{advisory_id}", publication_url]
    patt = re.compile(r"(CVSS|Base Score)[^0-9]{0,10}(\d{1,2}\.\d)", re.I)
    for u in filter(None, candidates):
        try:
            r = requests.get(u, headers=HTML_HEADERS, timeout=30)
            if not r.ok:
                continue
            text = BeautifulSoup(r.text, "lxml").get_text(" ", strip=True)
            matches = patt.findall(text)
            scores = []
            for _, s in matches:
                try:
                    v = float(s)
                    if 0 <= v <= 10:
                        scores.append(v)
                except Exception:
                    pass
            if scores:
                return max(scores)
        except Exception:
            continue
    return None


# ===== Normalize 2-col Fixed Releases =====
def parse_fixed_releases_from_tables(tables: List[Dict]) -> List[Dict]:
    out = []
    for t in tables or []:
        for r in t.get("rows", []) or []:
            left = _clean(r[0]) if len(r) >= 1 else ""
            right = _clean(r[1]) if len(r) >= 2 else ""
            if left or right:
                out.append({"release": left, "fixed_in": right})
    # dedupe
    seen = set()
    uniq = []
    for m in out:
        key = (m["release"], m["fixed_in"])
        if key not in seen:
            seen.add(key)
            uniq.append(m)
    return uniq


# ===== Transform one advisory (FULL ENRICHED RECORD) =====
def transform_advisory(adv: Dict, token: str) -> Dict:
    aid = adv.get("advisoryId") or adv.get("id")
    title = _clean(adv.get("title") or adv.get("advisoryTitle") or "")
    first_p = _iso_date_only(adv.get("firstPublished"))
    last_u = _iso_date_only(adv.get("lastUpdated") or adv.get("lastUpdatedDate"))

    # CVRF enrich
    cvrf_url = adv.get("cvrfUrl") or adv.get("cvrfURL")
    cvrf = {}
    if _is_valid_http_url(cvrf_url):
        xml = get_text(cvrf_url, token)
        if xml:
            cvrf = parse_cvrf(xml)
            time.sleep(PAUSE_SEC)

    # Fixed Software (text + normalized table)
    fs = html_fixed_software_fallback(aid, adv.get("publicationUrl"))
    fixed_sw_text = [
        row["fix"]
        for row in (fs.get("text_rows") or [])
        if row.get("fix")
    ]
    fixed_sw_tables = fs.get("tables") or []
    fixed_releases_norm = parse_fixed_releases_from_tables(fixed_sw_tables)
    fixed_software = {
        "text": fixed_sw_text,
        "fixed_releases": fixed_releases_norm,
    }

    # Workarounds (CVRF -> HTML fallback)
    workarounds = (cvrf.get("workarounds") or "").strip() if cvrf else ""
    if not workarounds:
        workarounds = html_workarounds_fallback(aid, adv.get("publicationUrl"))

    # Affected Products (CVRF -> HTML fallback)
    affected_products_text = (
        cvrf.get("affected_products") or ""
    ).strip() if cvrf else ""
    vulnerable_list = cvrf.get("vulnerable_products", []) if cvrf else []
    not_vuln_list = cvrf.get("products_confirmed_not_vulnerable", []) if cvrf else []

    if not affected_products_text or (not vulnerable_list and not not_vuln_list):
        html_aff = html_affected_products_fallback(aid, adv.get("publicationUrl"))
        if html_aff.get("affected_products") and not affected_products_text:
            affected_products_text = html_aff["affected_products"]
        if not vulnerable_list and html_aff.get("vulnerable_products"):
            vulnerable_list = [
                line for line in html_aff["vulnerable_products"].split("\n\n")
                if line.strip()
            ]
        raw_nv = (
            html_aff.get("products_confirmed_not_vulnerable")
            or html_aff.get("products_confirmednotvulnerable")
            or ""
        )
        if not not_vuln_list and raw_nv:
            not_vuln_list = [
                line for line in raw_nv.split("\n\n")
                if line.strip()
            ]

    # CVSS (JSON -> CVRF -> HTML)
    cvss_candidates = []
    for k in ("cvssBaseScore", "cvssScore"):
        try:
            v = adv.get(k)
            if v is not None:
                cvss_candidates.append(float(v))
        except Exception:
            pass
    if cvrf and cvrf.get("cvss_max_base") is not None:
        try:
            cvss_candidates.append(float(cvrf["cvss_max_base"]))
        except Exception:
            pass
    html_cvss = html_cvss_fallback(aid, adv.get("publicationUrl"))
    if html_cvss is not None:
        cvss_candidates.append(html_cvss)
    cvss_score = max([v for v in cvss_candidates if v is not None], default=None)

    # Source (CVRF -> HTML fallback)
    source_text = (cvrf.get("source") or "").strip() if cvrf else ""
    if not source_text:
        source_text = html_source_fallback(aid, adv.get("publicationUrl"))

    # Version, Version Status, Cisco Bug IDs
    vers_info = html_version_status_and_bugs(aid, adv.get("publicationUrl"))
    version = vers_info.get("version")
    version_status = vers_info.get("version_status")
    bug_ids = vers_info.get("bug_ids", [])

    url = f"{CANON_BASE}{aid}" if aid else (adv.get("publicationUrl") or "")

    return {
        "Title": title,
        "Advisory ID": aid,
        "First Published": first_p,
        "Last Updated": last_u,
        "Summary": cvrf.get("summary", "") if cvrf else "",
        "Source": source_text,
        "Version": version,
        "Version Status": version_status,
        "Cisco Bug IDs": bug_ids,
        "CVSS Score": cvss_score,
        "Workarounds": workarounds,
        "Affected Products": affected_products_text,
        "Vulnerable Products (list)": vulnerable_list,
        "Products Confirmed Not Vulnerable (list)": not_vuln_list,
        "Revision History": cvrf.get("revision_history", []) if cvrf else [],
        "Fixed Software": fixed_software,
        "URL": url,
    }


# ===== Bulk fetch from 2024 onward (INTERNAL RECORDS) =====
def fetch_all_advisories(
    start_year: int,
    end_year: int,
    check_qdrant: bool = True,   # <- Qdrant dedupe ON by default
) -> List[Dict]:
    """
    Iterates year endpoints and transforms each advisory with full enrichment.
    Dedupe by Advisory ID. Returns list of enriched advisory dicts.

    If check_qdrant=True, we skip advisories whose canonical URL
    is already present in Qdrant (via url_already_ingested).
    """
    token = get_token(CLIENT_ID, CLIENT_SECRET)
    seen_ids = set()
    items: List[Dict] = []

    for y in range(end_year, start_year - 1, -1):
        try:
            payload = get_json(f"year/{y}", token)
            year_list = payload.get("advisories", []) or []
        except Exception as e:
            logger.warning("[CISCO] Fetch year %s failed: %s", y, e)
            continue

        for adv in year_list:
            aid = adv.get("advisoryId") or adv.get("id")
            if not aid or aid in seen_ids:
                continue

            # Canonical URL used for dedupe
            url = f"{CANON_BASE}{aid}" if aid else (adv.get("publicationUrl") or "")

            # 🔍 Qdrant check BEFORE heavy CVRF + HTML parsing
            if check_qdrant and url and url_already_ingested(url):
                logger.info("[CISCO] Skipping already-ingested URL: %s", url)
                continue

            try:
                item = transform_advisory(adv, token)
                items.append(item)
                seen_ids.add(aid)
            except Exception as e:
                logger.warning("[CISCO] Transform failed for %s: %s", aid, e)

            time.sleep(PAUSE_SEC)

        time.sleep(PAUSE_SEC)

    # sort newest first by Last Updated (then First Published)
    def sort_key(d):
        lu = _parse_dt(d.get("Last Updated")) \
             or _parse_dt(d.get("First Published")) \
             or datetime.fromtimestamp(0, tz=timezone.utc)
        return lu

    items.sort(key=sort_key, reverse=True)
    return items


def save_json(path: str, data: List[Dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    print(f"[OK] Saved {len(data)} advisories to {path}")


# ===== COLAB HELPERS (OPTIONAL, FULL JSON) =====
def fetch_one_advisory_colab(advisory_id: Optional[str] = None) -> Dict:
    token = get_token(CLIENT_ID, CLIENT_SECRET)
    adv = (
        fetch_advisory_by_id(token, advisory_id)
        if advisory_id
        else fetch_latest_advisory(token, START_YEAR, END_YEAR)
    )
    if not adv:
        raise RuntimeError("Advisory not found")
    item = transform_advisory(adv, token)
    print(json.dumps(item, indent=2, ensure_ascii=False))
    return item


def fetch_all_advisories_colab():
    # For full export/debug, disable Qdrant dedupe
    data = fetch_all_advisories(START_YEAR, END_YEAR, check_qdrant=False)
    save_json(OUT_JSON, data)
    if data:
        print("\n[Preview of first item]")
        print(json.dumps(data[0], indent=2, ensure_ascii=False))


# ===== RECORD → {url, title, description} FOR RAG =====
def cisco_record_to_document(rec: Dict[str, object]) -> Dict[str, str]:
    """
    Build minimal RAG document:

      - url
      - title
      - description:
          * Summary (as Description)
          * Workarounds
          * Affected products text
          * Vulnerable / not-vulnerable lists
          * Fixed software (text rows + fixed releases)
          * Cisco Bug IDs
          * CVSS Score
    """
    url = (rec.get("URL") or "").strip()  # type: ignore[union-attr]
    title = (
        (rec.get("Title") or "Cisco Security Advisory")  # type: ignore[operator]
        .strip()
    )

    parts: List[str] = []

    # 1) Summary → main description
    summary = rec.get("Summary")
    if isinstance(summary, str) and summary.strip():
        parts.append("Description: " + summary.strip())

    # 2) Workarounds
    wa = rec.get("Workarounds")
    if isinstance(wa, str) and wa.strip():
        parts.append("Workarounds: " + wa.strip())

    # 3) Affected products text
    aff_text = rec.get("Affected Products")
    if isinstance(aff_text, str) and aff_text.strip():
        parts.append("Affected products: " + aff_text.strip())

    # 4) Vulnerable products list
    vuln_list = rec.get("Vulnerable Products (list)") or []
    if isinstance(vuln_list, list) and vuln_list:
        lines = [str(x).strip() for x in vuln_list if str(x).strip()]
        if lines:
            parts.append("Vulnerable products (list):\n" + "\n".join(lines))

    # 5) Not-vulnerable products list
    nv_list = rec.get("Products Confirmed Not Vulnerable (list)") or []
    if isinstance(nv_list, list) and nv_list:
        lines = [str(x).strip() for x in nv_list if str(x).strip()]
        if lines:
            parts.append(
                "Products confirmed not vulnerable:\n" + "\n".join(lines)
            )

    # 6) Fixed Software (text + normalized fixed_releases)
    fixed = rec.get("Fixed Software") or {}
    if isinstance(fixed, dict):
        text_rows = fixed.get("text") or []
        if isinstance(text_rows, list) and text_rows:
            lines = [str(x).strip() for x in text_rows if str(x).strip()]
            if lines:
                parts.append("Fixed software (notes):\n" + "\n".join(lines))

        fixed_releases = fixed.get("fixed_releases") or []
        if isinstance(fixed_releases, list) and fixed_releases:
            fr_lines: List[str] = []
            for row in fixed_releases:
                if not isinstance(row, Dict):
                    continue
                rel = (row.get("release") or "").strip()
                fixed_in = (row.get("fixed_in") or "").strip()
                bits: List[str] = []
                if rel:
                    bits.append(f"release {rel}")
                if fixed_in:
                    bits.append(f"fixed in {fixed_in}")
                if bits:
                    fr_lines.append(" – ".join(bits))
            if fr_lines:
                parts.append(
                    "Fixed releases:\n" + "\n".join(fr_lines)
                )

    # 7) Cisco Bug IDs
    bug_ids = rec.get("Cisco Bug IDs") or []
    if isinstance(bug_ids, list) and bug_ids:
        parts.append("Cisco Bug IDs: " + ", ".join(map(str, bug_ids)))

    # 8) CVSS score
    cvss_score = rec.get("CVSS Score")
    if cvss_score is not None:
        parts.append(f"CVSS score: {cvss_score}")

    description = "\n".join(parts).strip()

    return {
        "url": url,
        "title": title,
        "description": description or title,
    }


# ===== PUBLIC ENTRYPOINT FOR YOUR PIPELINE =====
def scrape_cisco(
    start_year: int = START_YEAR,
    end_year: int = END_YEAR,
    check_qdrant: bool = True,
) -> List[Dict[str, str]]:
    """
    High-level scraper used by your cron / scraping graph.

      - Fetches & enriches Cisco advisories from Cisco API (CVRF + HTML).
      - If check_qdrant=True: skips canonical URLs already in Qdrant
        BEFORE heavy CVRF/HTML calls, via url_already_ingested().
      - Returns list of minimal RAG documents: {url, title, description}
    """
    records = fetch_all_advisories(
        start_year=start_year,
        end_year=end_year,
        check_qdrant=check_qdrant,
    )

    docs: List[Dict[str, str]] = []
    for rec in records:
        url = (rec.get("URL") or "").strip()  # type: ignore[union-attr]
        if not url:
            continue
        doc = cisco_record_to_document(rec)
        docs.append(doc)

    logger.info("[CISCO] Scraped %d advisories (documents).", len(docs))
    return docs


# === Example usage as a script (optional) ===
if __name__ == "__main__":
    # For Colab / debugging: full JSON dump (no Qdrant dedupe).
    # fetch_all_advisories_colab()

    # For RAG testing: minimal docs, no Qdrant dedupe in local tests
    docs = scrape_cisco(check_qdrant=False)
    print(f"Sample docs: {len(docs)}")
    if docs:
        print(json.dumps(docs[0], indent=2, ensure_ascii=False))
