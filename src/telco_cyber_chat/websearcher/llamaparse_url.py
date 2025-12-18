from __future__ import annotations
import os
import time
from typing import Any, Dict, Optional

import requests

V2_PARSE_URL = "https://api.cloud.llamaindex.ai/api/v2alpha1/parse/url"
V1_JOB = "https://api.cloud.llamaindex.ai/api/v1/parsing/job/{job_id}"
V1_MD  = "https://api.cloud.llamaindex.ai/api/v1/parsing/job/{job_id}/result/markdown"

def _auth_headers(api_key: str) -> Dict[str, str]:
    return {"Authorization": f"Bearer {api_key}"}

def submit_parse_url(
    source_url: str,
    api_key: str,
    tier: str = "agentic",
    version: str = "latest",
    max_pages: int = 200,
) -> str:
    payload: Dict[str, Any] = {
        "source_url": source_url,
        "parse_options": {"tier": tier, "version": version},
        "page_ranges": {"max_pages": max_pages},
    }
    r = requests.post(
        V2_PARSE_URL,
        json=payload,
        headers={**_auth_headers(api_key), "Content-Type": "application/json"},
        timeout=60,
    )
    r.raise_for_status()
    data = r.json()

    # v2 docs say response is a ParsingJob object (same structure as v1),
    # so job id is typically in "id" (but we guard a few keys).
    job_id = data.get("id") or data.get("job_id") or data.get("jobId")
    if not job_id:
        raise RuntimeError(f"Could not find job id in response: keys={list(data.keys())}")
    return str(job_id)

def wait_job_done(job_id: str, api_key: str, timeout_s: int = 600) -> Dict[str, Any]:
    start = time.time()
    while True:
        r = requests.get(V1_JOB.format(job_id=job_id), headers=_auth_headers(api_key), timeout=60)
        r.raise_for_status()
        job = r.json()

        status = (job.get("status") or job.get("state") or "").upper()
        if status in {"SUCCESS", "SUCCEEDED", "COMPLETED", "DONE"}:
            return job
        if status in {"ERROR", "FAILED", "FAILURE"}:
            raise RuntimeError(f"LlamaParse job failed: {job}")

        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timed out waiting for job {job_id}. Last status={status}")

        time.sleep(3.0)

def fetch_markdown(job_id: str, api_key: str) -> str:
    r = requests.get(V1_MD.format(job_id=job_id), headers=_auth_headers(api_key), timeout=120)
    r.raise_for_status()
    data = r.json()
    md = data.get("markdown")
    if not isinstance(md, str) or not md.strip():
        raise RuntimeError(f"No markdown in result for job {job_id}: keys={list(data.keys())}")
    return md

def parse_url_to_markdown(
    source_url: str,
    api_key: str,
    tier: str,
    version: str,
    max_pages: int,
) -> str:
    job_id = submit_parse_url(source_url, api_key, tier=tier, version=version, max_pages=max_pages)
    wait_job_done(job_id, api_key)
    return fetch_markdown(job_id, api_key)
