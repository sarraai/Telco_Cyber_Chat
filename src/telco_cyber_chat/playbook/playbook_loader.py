# src/telco_cyber_chat/playbook_loader.py
from __future__ import annotations
from pathlib import Path
from functools import lru_cache

PLAYBOOK_DIR = Path(__file__).resolve().parent / "playbook"
ORDER = ["SYSTEM.md", "SAFETY.md", "TOOL_USE.md", "DOMAIN_TELCO.md", "EXAMPLES.md"]

@lru_cache(maxsize=1)
def load_playbook_text() -> str:
    parts = []
    for name in ORDER:
        text = (PLAYBOOK_DIR / name).read_text(encoding="utf-8")
        parts.append(f"\n\n### {name}\n{text}")
    return "\n".join(parts).strip()
