from pathlib import Path
from typing import List

PLAYBOOK_DIR = Path(__file__).resolve().parent  # ✅ no "/playbook"

DEFAULT_ORDER = [
    "SYSTEM.md",
    "SAFETY.md",
    "TOOL_USE.md",
    "DOMAIN_TELCO.md",
    "EXAMPLES.md",
    "LEARNED_RULES.md",  # optional
]

def load_playbook_text(playbook_dir: Path = PLAYBOOK_DIR) -> str:
    sections: List[str] = []
    for name in DEFAULT_ORDER:
        p = playbook_dir / name
        if not p.exists():
            continue  # don't crash the whole graph
        content = p.read_text(encoding="utf-8").strip()
        if content:
            sections.append(f"### {p.name}\n{content}")
    return "\n\n".join(sections).strip()
