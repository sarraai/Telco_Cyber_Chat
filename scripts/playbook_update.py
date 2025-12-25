from __future__ import annotations

import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

# -----------------------------------------------------------------------------
# Config
# -----------------------------------------------------------------------------
SUPABASE_URL = os.environ["SUPABASE_URL"].rstrip("/")
SUPABASE_SERVICE_ROLE_KEY = os.environ["SUPABASE_SERVICE_ROLE_KEY"].strip()

FEEDBACK_TABLE = os.getenv("FEEDBACK_TABLE", "chat_feedback").strip()
NEG_THRESHOLD = int(os.getenv("ACE_NEG_THRESHOLD", "2"))
WRITE_CANONICAL = os.getenv("ACE_WRITE_CANONICAL", "1").strip() not in {"0", "false", "False"}
ENABLE_LANGSMITH = os.getenv("ACE_ENABLE_LANGSMITH", "1").strip() not in {"0", "false", "False"}

REST = f"{SUPABASE_URL}/rest/v1"
HEADERS = {
    "apikey": SUPABASE_SERVICE_ROLE_KEY,
    "Authorization": f"Bearer {SUPABASE_SERVICE_ROLE_KEY}",
    "Content-Type": "application/json",
}

PLAYBOOK_DIR = Path("src/telco_cyber_chat/playbook")
LEARNED_RULES_PATH = PLAYBOOK_DIR / "LEARNED_RULES.md"
SYSTEM_PATH = PLAYBOOK_DIR / "SYSTEM.md"
SAFETY_PATH = PLAYBOOK_DIR / "SAFETY.md"
TOOL_USE_PATH = PLAYBOOK_DIR / "TOOL_USE.md"
EXAMPLES_PATH = PLAYBOOK_DIR / "EXAMPLES.md"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _today() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%d")


def _safe_str(x: Any) -> str:
    return "" if x is None else str(x)


def _to_int_or_none(x: Any) -> Optional[int]:
    if x is None:
        return None
    if isinstance(x, bool):
        return 1 if x else 0
    if isinstance(x, int):
        return x
    if isinstance(x, float):
        return int(x)
    if isinstance(x, str):
        s = x.strip()
        if s == "":
            return None
        if s.lower() in {"true", "yes"}:
            return 1
        if s.lower() in {"false", "no"}:
            return 0
        try:
            return int(float(s))
        except Exception:
            return None
    return None


def _is_negative(score: Optional[int]) -> bool:
    return score is not None and score <= NEG_THRESHOLD


def _get_first_key(row: Dict[str, Any], keys: List[str]) -> Any:
    for k in keys:
        if k in row:
            return row.get(k)
    return None


def _get_helpfulness(row: Dict[str, Any]) -> Optional[int]:
    return _to_int_or_none(_get_first_key(row, ["helpfulness", "helfullness", "helpful"]))


def _get_safety(row: Dict[str, Any]) -> Optional[int]:
    return _to_int_or_none(_get_first_key(row, ["safety", "safe"]))


def _get_role(row: Dict[str, Any]) -> str:
    r = _safe_str(row.get("role")).strip().lower()
    return r or "unknown"


def _get_assistant_id(row: Dict[str, Any]) -> str:
    a = _safe_str(row.get("assistant_id")).strip()
    return a or "unknown"


def _update_markdown_daily_section(path: Path, title: str, summary_block: str, bullets: List[str]) -> bool:
    path.parent.mkdir(parents=True, exist_ok=True)
    date_str = _today()
    section_header = f"## {date_str}"

    if path.exists():
        text = path.read_text(encoding="utf-8")
    else:
        text = f"# {title}\n\n"

    section_lines: List[str] = [
        section_header,
        "",
        "### Batch summary",
        summary_block.strip() if summary_block.strip() else "- (no summary)",
        "",
        "### New/updated rules",
    ]
    section_lines += [f"- {b}" for b in bullets] if bullets else ["- (no rule changes)"]
    section_text = "\n".join(section_lines).strip() + "\n"

    if section_header in text:
        pattern = rf"{re.escape(section_header)}\n.*?(?=\n## |\Z)"
        new_text, n = re.subn(pattern, section_text + "\n", text, flags=re.S)
        changed = (n > 0 and new_text != text)
        if changed:
            path.write_text(new_text, encoding="utf-8")
        return changed

    new_text = text.rstrip() + "\n\n" + section_text
    changed = new_text != text
    if changed:
        path.write_text(new_text, encoding="utf-8")
    return changed


def fetch_unprocessed(limit: int = 500) -> List[Dict[str, Any]]:
    base = f"{REST}/{FEEDBACK_TABLE}?select=*&processed_at=is.null&limit={limit}"
    url_ordered = base + "&order=created_at.asc"
    r = requests.get(url_ordered, headers=HEADERS, timeout=60)
    if r.status_code >= 400:
        r = requests.get(base, headers=HEADERS, timeout=60)
    r.raise_for_status()
    return r.json()


def mark_processed(rows: List[Dict[str, Any]]) -> None:
    ts = _now_iso()

    ids = [str(r["id"]) for r in rows if r.get("id") is not None]
    if ids:
        ids_csv = ",".join(ids)
        url = f"{REST}/{FEEDBACK_TABLE}?id=in.({ids_csv})"
        rr = requests.patch(url, headers=HEADERS, json={"processed_at": ts}, timeout=60)
        rr.raise_for_status()
        return

    for row in rows:
        run_id = row.get("run_id")
        user_id = row.get("user_id")
        if user_id and run_id:
            url = f"{REST}/{FEEDBACK_TABLE}?user_id=eq.{user_id}&run_id=eq.{run_id}"
            rr = requests.patch(url, headers=HEADERS, json={"processed_at": ts}, timeout=60)
            rr.raise_for_status()
            continue

        if run_id:
            url = f"{REST}/{FEEDBACK_TABLE}?run_id=eq.{run_id}"
            rr = requests.patch(url, headers=HEADERS, json={"processed_at": ts}, timeout=60)
            rr.raise_for_status()


@dataclass
class RunSignal:
    tool_error: bool = False
    weak_evidence: bool = False
    overlong_answer: bool = False
    notes: List[str] = None

    def __post_init__(self):
        if self.notes is None:
            self.notes = []


def _hydrate_run_signals(run_id: str) -> RunSignal:
    sig = RunSignal()
    if not ENABLE_LANGSMITH:
        return sig

    try:
        from langsmith import Client  # type: ignore
    except Exception:
        sig.notes.append("langsmith_client_unavailable")
        return sig

    try:
        ls = Client()
        run = ls.read_run(run_id)
    except Exception as e:
        sig.notes.append(f"langsmith_read_failed:{type(e).__name__}")
        return sig

    err = getattr(run, "error", None)
    if err:
        sig.tool_error = True
        sig.notes.append("run_error_present")

    outputs = getattr(run, "outputs", None) or {}
    if isinstance(outputs, dict):
        sources = outputs.get("sources")
        context = outputs.get("context")
        if sources == [] or sources == {}:
            sig.weak_evidence = True
            sig.notes.append("empty_sources")
        if isinstance(context, str) and context.strip() == "":
            sig.weak_evidence = True
            sig.notes.append("empty_context")

        answer = outputs.get("answer") or outputs.get("output") or outputs.get("response")
        if isinstance(answer, str) and len(answer) > 1200:
            sig.overlong_answer = True
            sig.notes.append("answer_len_gt_1200")

    return sig


def summarize(rows: List[Dict[str, Any]]) -> Tuple[str, Dict[str, List[str]]]:
    by_assistant: Dict[str, Dict[str, Any]] = {}

    n_hydrated = 0
    n_tool_errors = 0
    n_weak_evidence = 0
    n_overlong = 0

    learned_rules: List[str] = []
    safety_rules: List[str] = []
    tool_use_rules: List[str] = []
    system_rules: List[str] = []
    example_rules: List[str] = []

    for r in rows:
        assistant_id = _get_assistant_id(r)
        role = _get_role(r)
        helpful = _get_helpfulness(r)
        safe = _get_safety(r)

        st = by_assistant.setdefault(
            assistant_id,
            {"n": 0, "by_role": {}, "help_neg": 0, "safe_neg": 0},
        )
        st["n"] += 1
        if _is_negative(helpful):
            st["help_neg"] += 1
        if _is_negative(safe):
            st["safe_neg"] += 1

        rr = st["by_role"].setdefault(role, {"n": 0, "help_neg": 0, "safe_neg": 0})
        rr["n"] += 1
        if _is_negative(helpful):
            rr["help_neg"] += 1
        if _is_negative(safe):
            rr["safe_neg"] += 1

        if _is_negative(helpful) or _is_negative(safe):
            run_id = r.get("run_id")
            if run_id:
                sig = _hydrate_run_signals(str(run_id))
                n_hydrated += 1
                if sig.tool_error:
                    n_tool_errors += 1
                if sig.weak_evidence:
                    n_weak_evidence += 1
                if sig.overlong_answer:
                    n_overlong += 1

    lines: List[str] = []
    lines.append(f"- Update generated from {len(rows)} new feedback rows (table `{FEEDBACK_TABLE}`).")
    if ENABLE_LANGSMITH:
        lines.append(
            f"- LangSmith hydration attempted for {n_hydrated} negative rows: tool_errors={n_tool_errors}, weak_evidence={n_weak_evidence}, overlong={n_overlong}."
        )

    for assistant_id, st in sorted(by_assistant.items(), key=lambda x: x[0]):
        n = max(int(st["n"]), 1)
        help_neg = int(st["help_neg"])
        safe_neg = int(st["safe_neg"])

        lines.append(f"- Assistant `{assistant_id}`: n={st['n']}, not_helpful={help_neg}, unsafe={safe_neg}")

        if help_neg / n >= 0.30:
            system_rules.append(
                f"For assistant `{assistant_id}`: start with a concise direct answer (2–6 sentences), then add bullets only if needed. Avoid long explanations unless the user asks."
            )

        if safe_neg > 0:
            safety_rules.append(
                f"For assistant `{assistant_id}`: strengthen refusal behavior—do not provide offensive/operational steps; redirect to defensive guidance (hardening, detection, patching)."
            )
            example_rules.append(
                "Add/refresh an example: when asked 'how to exploit X', refuse briefly and offer safe alternatives (hardening + detection ideas)."
            )

        if n_tool_errors > 0:
            tool_use_rules.append(
                f"For assistant `{assistant_id}`: when a tool call fails, say what failed (briefly), what you can do instead, and provide a safe fallback path."
            )
        if n_weak_evidence > 0:
            tool_use_rules.append(
                f"For assistant `{assistant_id}`: if retrieval is empty/weak, explicitly say it’s not in indexed sources and use web search (when allowed) rather than guessing."
            )
        if n_overlong > 0:
            system_rules.append(
                f"For assistant `{assistant_id}`: reduce verbosity by default; prefer short answers and ask one focused follow-up question if needed."
            )

        for role, rr in sorted(st["by_role"].items(), key=lambda x: x[0]):
            rn = max(int(rr["n"]), 1)
            if rr["help_neg"] / rn >= 0.40 and role == "end_user":
                system_rules.append(
                    "For **end_user**: keep it simple—plain language impact + 2–4 safe next steps. No deep configuration unless asked."
                )

    if not (safety_rules or tool_use_rules or system_rules or example_rules):
        learned_rules.append("No rule changes triggered by this batch (feedback looks stable).")

    learned_rules += safety_rules + tool_use_rules + system_rules + example_rules

    def dedup(seq: List[str]) -> List[str]:
        seen = set()
        out = []
        for s in seq:
            k = s.strip().lower()
            if k and k not in seen:
                seen.add(k)
                out.append(s)
        return out

    updates = {
        "LEARNED_RULES": dedup(learned_rules),
        "SAFETY": dedup(safety_rules),
        "TOOL_USE": dedup(tool_use_rules),
        "SYSTEM": dedup(system_rules),
        "EXAMPLES": dedup(example_rules),
    }
    return "\n".join(lines), updates


def apply_updates(summary_block: str, updates: Dict[str, List[str]]) -> bool:
    changed_any = False

    # Always write LEARNED_RULES.md (delta log)
    if LEARNED_RULES_PATH.exists():
        text = LEARNED_RULES_PATH.read_text(encoding="utf-8")
    else:
        text = "# Learned Rules (Auto-updated)\n\n## Latest\n- (empty)\n"
        LEARNED_RULES_PATH.parent.mkdir(parents=True, exist_ok=True)
        LEARNED_RULES_PATH.write_text(text, encoding="utf-8")

    today = _today()
    section_header = f"## {today}"
    section = [section_header, "", "### Batch summary", summary_block, "", "### New/updated rules"]
    section += [f"- {r}" for r in updates.get("LEARNED_RULES", [])]
    section_text = "\n".join(section).strip() + "\n"

    if section_header in text:
        pattern = rf"## {re.escape(today)}\n.*?(?=\n## |\Z)"
        new_text, n = re.subn(pattern, section_text + "\n", text, flags=re.S)
        if n > 0 and new_text != text:
            LEARNED_RULES_PATH.write_text(new_text, encoding="utf-8")
            changed_any = True
    else:
        new_text = text.rstrip() + "\n\n" + section_text
        if new_text != text:
            LEARNED_RULES_PATH.write_text(new_text, encoding="utf-8")
            changed_any = True

    # Optionally promote into canonical files
    if WRITE_CANONICAL:
        if updates.get("SAFETY"):
            changed_any |= _update_markdown_daily_section(
                SAFETY_PATH, "Safety & Policy Rules (Telco_CyberChat)", summary_block, updates["SAFETY"]
            )
        if updates.get("TOOL_USE"):
            changed_any |= _update_markdown_daily_section(
                TOOL_USE_PATH, "Tool Use Rules", summary_block, updates["TOOL_USE"]
            )
        if updates.get("SYSTEM"):
            changed_any |= _update_markdown_daily_section(
                SYSTEM_PATH, "Telco_CyberChat Playbook — System Rules", summary_block, updates["SYSTEM"]
            )
        if updates.get("EXAMPLES"):
            changed_any |= _update_markdown_daily_section(
                EXAMPLES_PATH, "Examples", summary_block, updates["EXAMPLES"]
            )

    return changed_any


def main() -> int:
    rows = fetch_unprocessed(limit=500)
    if not rows:
        print("No new feedback rows.")
        return 0

    summary_block, updates = summarize(rows)
    changed = apply_updates(summary_block, updates)

    mark_processed(rows)

    if changed:
        print("Playbook updated (LEARNED_RULES + optional canonical files).")
    else:
        print("No playbook file changes, but feedback marked processed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
