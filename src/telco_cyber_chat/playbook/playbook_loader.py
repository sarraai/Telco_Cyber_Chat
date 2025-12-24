from __future__ import annotations
from importlib import resources

def load_playbook_text() -> str:
    pkg = "telco_cyber_chat.playbook"
    base = resources.files(pkg)

    md_files = sorted([p for p in base.iterdir() if p.is_file() and p.name.endswith(".md")],
                      key=lambda p: p.name)

    parts = []
    for f in md_files:
        text = f.read_text(encoding="utf-8").strip()
        if text:
            parts.append(f"## {f.name}\n{text}")

    return "\n\n---\n\n".join(parts)
