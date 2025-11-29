import os
import re
import aiohttp
import asyncio
from typing import Dict, Any

# -----------------------------------------------------------------------------
# Small-talk guard (prevents canned telco bullets on greetings/thanks)
# -----------------------------------------------------------------------------
GREETING_REPLY = os.getenv(
    "GREETING_REPLY",
    "Hello! I'm your telecom-cybersecurity assistant.",
)
GOODBYE_REPLY = os.getenv("GOODBYE_REPLY", "Goodbye!")

# More precise regex that matches ENTIRE string
_GREET_RE = re.compile(
    r"^\s*(hi|hello|hey|greetings|good\s+(morning|afternoon|evening|day))\s*[!.?]*\s*$",
    re.I,
)
_BYE_RE = re.compile(
    r"^\s*(bye|goodbye|see\s+you|see\s+ya|thanks?\s*,?\s*bye|farewell)\s*[!.?]*\s*$",
    re.I,
)
_THANKS_RE = re.compile(
    r"^\s*(thanks|thank\s+you|thx)\s*[!.?]*\s*$", re.I
)


def _smalltalk_reply(text: str):
    """Return an immediate reply for greetings/thanks/very short small-talk,
    otherwise None.
    """
    t = (text or "").strip()
    if _GREET_RE.search(t):
        return GREETING_REPLY
    if _BYE_RE.search(t):
        return GOODBYE_REPLY
    if _THANKS_RE.search(t):
        return "You're welcome!"
    # Extremely short non-question → treat as small talk
    if len(re.findall(r"\w+", t)) <= 2 and not t.endswith("?"):
        return GREETING_REPLY
    return None


# -----------------------------------------------------------------------------
# Shared policy helpers (kept for role-based logic)
# -----------------------------------------------------------------------------
POLICY = {
    "end_user": {
        "deny": {
            "S1 Violent",
            "S2 Non-violent Crimes",
            "S3 Sex-crime",
            "S4 CSE",
            "S7 Privacy",
            "S9 Indiscriminate Weapons",
            "S10 Hate",
            "S11 Self-Harm",
            "S12 Sexual",
            "S13 Elections",
            "Malicious Code",
            "Phishing/SE",
        },
        "allow_defense_only": set(),
    },
    "it_specialist": {
        "deny": {
            "S7 Privacy",
            "S9 Indiscriminate Weapons",
            "S10 Hate",
            "S11 Self-Harm",
            "S12 Sexual",
        },
        "allow_defense_only": {
            "Malicious Code",
            "Phishing/SE",
            "S2 Non-violent Crimes",
            "S6 Specialized Illicit Advice",
        },
    },
    "network_admin": {
        "deny": {
            "S7 Privacy",
            "S9 Indiscriminate Weapons",
            "S10 Hate",
            "S11 Self-Harm",
            "S12 Sexual",
        },
        "allow_defense_only": {
            "Malicious Code",
            "Phishing/SE",
            "S2 Non-violent Crimes",
            "S6 Specialized Illicit Advice",
        },
    },
    "admin": {
        "deny": {
            "S7 Privacy",
            "S9 Indiscriminate Weapons",
            "S10 Hate",
            "S11 Self-Harm",
            "S12 Sexual",
            "S4 CSE",
        },
        "allow_defense_only": {
            "Malicious Code",
            "Phishing/SE",
            "S2 Non-violent Crimes",
            "S6 Specialized Illicit Advice",
        },
    },
}


def _canon_role(role: str) -> str:
    r = (role or "").strip().lower().replace(" ", "_")
    return r if r in POLICY else "end_user"


def role_directive(role: str) -> str:
    r = (role or "").lower().replace(" ", "_")
    if r == "end_user":
        return (
            "Audience: non-technical end user. Explain simply, avoid jargon.\n"
            "Output: ≤6 short bullets.\n"
        )
    if r == "it_specialist":
        return (
            "Audience: IT specialist. Provide technical bullets + brief rationale.\n"
        )
    if r == "network_admin":
        return (
            "Audience: network admin. Focus on configs, controls, rollout steps.\n"
        )
    if r == "admin":
        return (
            "Audience: executive/admin. 5-line summary: risk, impact, priority, owners.\n"
        )
    return "Audience: general user. Be concise.\n"


SAMPLING_PRESETS = {
    "factual": {"temperature": 0.3, "top_p": 0.9},
    "balanced": {"temperature": 0.7, "top_p": 0.9},
    "creative": {"temperature": 1.2, "top_p": 0.92},
}

# -----------------------------------------------------------------------------
# TELCO LLM BACKEND: LangServe via HTTP (ngrok) - ASYNC ONLY
# -----------------------------------------------------------------------------

# ❌ No hard-coded URL anymore – must be provided via env
TELCO_LLM_URL = os.getenv("TELCO_LLM_URL", "").strip()
TELCO_LLM_TIMEOUT = int(os.getenv("TELCO_LLM_TIMEOUT", "120"))


async def _call_telco_llm_async(inputs: Dict[str, Any]) -> str:
    """Async HTTP client for the Telco LLM LangServe endpoint.

    `inputs` will be passed as the `"input"` field for
    POST /ask_secure/invoke, e.g.:

        {
          "question": "...",
          "context": "...",
          "role": "it_specialist"
        }
    """
    if not TELCO_LLM_URL:
        raise RuntimeError(
            "TELCO_LLM_URL is not set. Configure it in your environment or .env."
        )

    payload = {"input": inputs}
    timeout = aiohttp.ClientTimeout(total=TELCO_LLM_TIMEOUT)

    try:
        async with aiohttp.ClientSession(timeout=timeout) as session:
            async with session.post(TELCO_LLM_URL, json=payload) as resp:
                resp.raise_for_status()
                data = await resp.json()
    except aiohttp.ClientError as e:
        return f"Error contacting Telco LLM backend (HTTP error): {e}"
    except asyncio.TimeoutError:
        return f"Error contacting Telco LLM backend: Request timed out after {TELCO_LLM_TIMEOUT}s"
    except Exception as e:
        return f"Error contacting Telco LLM backend: {e}"

    # LangServe /invoke returns: {"output": "..."} by default
    if isinstance(data, dict) and "output" in data:
        return data["output"]

    # Fallback: just stringify whatever we got
    return str(data)


# -----------------------------------------------------------------------------
# Final answer cleaning helper (shared with graph)
# -----------------------------------------------------------------------------
def clean_answer(raw: str) -> str:
    if not isinstance(raw, str):
        raw = str(raw or "")

    text = raw.strip()
    lower = text.lower()

    # 1) Remove any 'Rationale:' section (used by RAAT/judge)
    rationale_markers = ["\nrationale:", " rationale:", "rationale:"]
    for marker in rationale_markers:
        idx = lower.find(marker)
        if idx != -1 and idx > 0:
            text = text[:idx].strip()
            lower = text.lower()
            break

    # 2) Remove trailing 'Not enough evidence in context.' if there's text before
    guard_phrase = "not enough evidence in context"
    idx = lower.find(guard_phrase)
    if idx != -1 and idx > 0:
        text = text[:idx].strip()
        lower = text.lower()

    # 3) Remove arrow-style junk like '->', '-->', '<--', '=>', etc.
    arrow_patterns = [
        r'-->', r'->', r'<--', r'<-', r'==>', r'=>', r'<='
    ]
    for pat in arrow_patterns:
        text = re.sub(pat, " ", text)

    # 4) Collapse multiple spaces and tidy newlines
    text = re.sub(r'[ \t]+', ' ', text)           # collapse spaces/tabs
    text = re.sub(r' *\n *', '\n', text)          # clean spaces around newlines
    text = re.sub(r'\n{3,}', '\n\n', text).strip()  # at most 2 newlines in a row

    return text


# -----------------------------------------------------------------------------
# Main QA entrypoint - ASYNC ONLY
# -----------------------------------------------------------------------------
async def ask_secure_async(
    question: str,
    *,
    context: str = "",
    role: str = "end_user",
    max_new_tokens: int = 400,
    preset: str = "balanced",
    seed: int | None = None,
) -> str:
    """Async main QA entrypoint for the graph.

    Delegates to the remote Telco LLM backend (LangServe)
    via the /ask_secure/invoke endpoint, and then cleans the output.
    """
    # Small-talk fast path (local, no HTTP)
    st = _smalltalk_reply(question)
    if st is not None:
        return st

    inputs: Dict[str, Any] = {
        "question": question,
        "context": context,
        "role": role,
        "max_new_tokens": max_new_tokens,
        "preset": preset,
        "seed": seed,
    }

    raw = await _call_telco_llm_async(inputs)
    # Clean RAAT / guard decorations and arrow noise for user-facing answers
    return clean_answer(raw)


async def generate_text_async(prompt: str, **decoding) -> str:
    """Async compatibility wrapper: generate free-form text using the Telco LLM.

    Many RAG components call `generate_text(prompt, ...)`.
    Internally we route this through `ask_secure_async` with no context.
    """
    role = decoding.pop("role", "it_specialist")
    max_new_tokens = int(decoding.get("max_new_tokens", 400))

    return await ask_secure_async(
        question=prompt,
        context="",
        role=role,
        max_new_tokens=max_new_tokens,
        preset=decoding.get("preset", "balanced"),
        seed=decoding.get("seed"),
    )
