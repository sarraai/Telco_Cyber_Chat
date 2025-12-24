from __future__ import annotations
from pathlib import Path
from typing import List, Optional

PLAYBOOK_DIR = Path(__file__).resolve().parent  # folder containing the .md files

BASE_ORDER = [
    "SYSTEM.md",
    "SAFETY.md",
    "TOOL_USE.md",
    "DOMAIN_TELCO.md",
    "EXAMPLES.md",
    "LEARNED_RULES.md",  # optional
]

ROLE_FILE = {
    "end_user": "ROLE_END_USER.md",
    "it_specialist": "ROLE_IT_SPECIALIST.md",
    "network_admin": "ROLE_NETWORK_ADMIN.md",
}

def load_playbook_text(role: Optional[str] = None, playbook_dir: Path = PLAYBOOK_DIR) -> str:
    role = (role or "").strip().lower()
    role_md = ROLE_FILE.get(role)

    order: List[str] = BASE_ORDER.copy()
    if role_md:
        # put role guidance near the top so it strongly affects style
        order.insert(1, role_md)

    sections: List[str] = []
    for name in order:
        p = playbook_dir / name
        if not p.exists():
            continue
        content = p.read_text(encoding="utf-8").strip()
        if content:
            sections.append(f"### {p.name}\n{content}")

    return "\n\n".join(sections).strip()
