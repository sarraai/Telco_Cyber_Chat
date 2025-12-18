from __future__ import annotations
from typing import List

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200) -> List[str]:
    text = text or ""
    text = text.strip()
    if not text:
        return []

    if overlap >= chunk_size:
        overlap = max(0, chunk_size // 5)

    chunks: List[str] = []
    start = 0
    n = len(text)

    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)

    return chunks
