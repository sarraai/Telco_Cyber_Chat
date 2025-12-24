from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

import requests


SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"].strip()

REST = f"{SUPABASE_URL}/rest/v1"
HEADERS = {
    "apikey": SUPABASE_SERVICE_ROLE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    "Content-Type": "application/json",
}

# Repo paths
LEARNED_RULES_PATH = Path("src/telco_cyber_chat/playbook/LEARNED_RULES.md")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def fetch_unprocessed(limit: int = 500) -> List[Dict[str, Any]]:
    # PostgREST filters are query params (processed_at=is.null)
    url = (
        f"{REST}/chat_feedback"
        f"?select=user_id,run_id,role,helpful,safe,question,answer,created_at"
        f"&processed_at=is.null"
        f"&order=created_at.asc"
        f"&limit={limit}"
    )
    r = requests.get(url, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


def mark_processed(rows: List[Dict[str, Any]]) -> None:
    # Table unique key is (user_id, run_id) in your UI upsert
    ts = _now_iso()
    for row in rows:
        user_id = row.get("user_id")
        run_id = row.get("run_id")
        if not user_id or not run_id:
            continue
        url = f"{REST}/chat_feedback?user_id=eq.{user_id}&run_id=eq.{run_id}"
        r = requests.patch(url, headers=HEADERS, json={"processed_at": ts}, timeout=60)
        r.raise_for_status()


def summarize(rows: List[Dict[str, Any]]) -> Tuple[str, List[str]]:
    """
    Turn binary feedback into concrete playbook tweaks.
    (Much stronger if you later add a free-text 'comment' field.)
    """
    by_role: Dict[str, Dict[str, Any]] = {}

    def role_key(x: Any) -> str:
        x = (x or "unknown").strip().lower()
        return x if x else "unknown"

    for r in rows:
        role = role_key(r.get("role"))
        st = by_role.setdefault(role, {"n": 0, "helpful_1": 0, "safe_1": 0, "unsafe_0": 0, "not_helpful_0": 0})
        st["n"] += 1
        if r.get("helpful") == 1:
            st["helpful_1"] += 1
        if r.get("helpful") == 0:
            st["not_helpful_0"] += 1
        if r.get("safe") == 1:
            st["safe_1"] += 1
        if r.get("safe") == 0:
            st["unsafe_0"] += 1

    lines: List[str] = []
    lines.append(f"- Update generated from {len(rows)} new feedback rows.")

    # Heuristics -> playbook rules
    rules: List[str] = []

    for role, st in sorted(by_role.items()):
        n = max(st["n"], 1)
        helpful_rate = st["helpful_1"] / n
        unsafe_count = st["unsafe_0"]

        lines.append(
            f"- Role `{role}`: n={st['n']}, helpful_rate={helpful_rate:.0%}, unsafe_flags={unsafe_count}"
        )

        # If users say "not helpful": shorten + be more direct
        if helpful_rate < 0.70:
            if role == "end_user":
                rules.append(
                    "For **end_user**: default to a short, plain-language **direct answer only** (no long evidence section unless asked)."
                )
            else:
                rules.append(
                    f"For **{role}**: start with a concise direct answer, then provide only the *minimum* steps/checklist needed."
                )

        # If anyone flags unsafe: strengthen refusal behavior
        if unsafe_count > 0:
            rules.append(
                f"For **{role}**: strengthen safety behavior—refuse requests for offensive actions, and redirect to defensive guidance."
            )

    if not rules:
        rules.append("No rule changes triggered by this batch (feedback looks stable).")

    # De-duplicate rules while preserving order
    seen = set()
    deduped_rules = []
    for rr in rules:
        k = rr.strip().lower()
        if k not in seen:
            seen.add(k)
            deduped_rules.append(rr)

    return "\n".join(lines), deduped_rules


def apply_to_learned_rules(summary_block: str, rules: List[str]) -> bool:
    LEARNED_RULES_PATH.parent.mkdir(parents=True, exist_ok=True)

    if LEARNED_RULES_PATH.exists():
        text = LEARNED_RULES_PATH.read_text(encoding="utf-8")
    else:
        text = "# Learned Rules (Auto-updated)\n\n## Latest\n- (empty)\n"

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    section_header = f"## {today}"
    section = [section_header, "", "### Batch summary", summary_block, "", "### New/updated rules"]
    section += [f"- {r}" for r in rules]
    section_text = "\n".join(section).strip() + "\n"

    if section_header in text:
        # If today's section already exists, replace it (idempotent)
        pattern = rf"## {re.escape(today)}\n.*?(?=\n## |\Z)"
        new_text, n = re.subn(pattern, section_text + "\n", text, flags=re.S)
        changed = (n > 0 and new_text != text)
        if changed:
            LEARNED_RULES_PATH.write_text(new_text, encoding="utf-8")
        return changed

    # Insert before next section or at end
    if "## Latest" in text:
        new_text = text + "\n" + section_text
    else:
        new_text = text + "\n\n## Latest\n\n" + section_text

    changed = new_text != text
    if changed:
        LEARNED_RULES_PATH.write_text(new_text, encoding="utf-8")
    return changed


def main() -> int:
    rows = fetch_unprocessed(limit=500)
    if not rows:
        print("No new feedback rows.")
        return 0

    summary_block, rules = summarize(rows)
    changed = apply_to_learned_rules(summary_block, rules)

    # Mark processed even if no file changes (so we don’t loop forever)
    mark_processed(rows)

    if changed:
        print("Playbook updated.")
    else:
        print("No playbook file changes, but feedback marked processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
