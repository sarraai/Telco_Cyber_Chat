import os
import re
import numpy as np
import requests
from functools import lru_cache
from typing import List, Dict, Any
from huggingface_hub import InferenceClient

# -----------------------------------------------------------------------------
# Global config
# -----------------------------------------------------------------------------
USE_REMOTE = os.getenv("USE_REMOTE_HF", "false").lower() == "true"

# HF router (OpenAI-compatible) base (kept but unused now)
REMOTE_BASE = os.getenv("HF_BASE_URL", "https://router.huggingface.co/v1")

# Pin both generator and guard to featherless-ai by default (kept for compat)
REMOTE_MODEL_ID = os.getenv(
    "REMOTE_MODEL_ID",
    "fdtn-ai/Foundation-Sec-8B-Instruct:featherless-ai",
)
REMOTE_GUARD_ID = os.getenv(
    "REMOTE_GUARD_ID",
    "meta-llama/Llama-Guard-3-8B:featherless-ai",
)

# Force a specific provider everywhere unless overridden (unused now)
HF_PROVIDER = os.getenv("HF_PROVIDER", "featherless-ai")
# Keep guard on the same provider as generator (true by default)
HF_ALIGN_GUARD_PROVIDER = (
    os.getenv("HF_ALIGN_GUARD_PROVIDER", "true").lower() == "true"
)
# Optional: allow streaming; we'll buffer tokens so callers still get a string
HF_STREAM = os.getenv("HF_STREAM", "false").lower() == "true"

HF_TOKEN = (
    os.getenv("HF_TOKEN")
    or os.getenv("HUGGINGFACE_HUB_TOKEN")
    or os.getenv("LLM_TOKEN")
)

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
# BGE similarity via HF Inference (SYNC)
# -----------------------------------------------------------------------------
BGE_MODEL_ID = os.getenv("BGE_MODEL_ID", "BAAI/bge-m3")


@lru_cache(maxsize=1)
def _get_bge_client() -> InferenceClient:
    if not HF_TOKEN:
        raise RuntimeError("HF_TOKEN env var is missing for BGE client")
    return InferenceClient(api_key=HF_TOKEN)


def bge_sentence_similarity(
    source: str, candidates: List[str], model: str = BGE_MODEL_ID
) -> List[float]:
    """Synchronous sentence similarity using HF Inference.
    Falls back to embedding + cosine if sentence_similarity endpoint is unavailable.
    """
    client = _get_bge_client()
    try:
        return client.sentence_similarity(
            {"source_sentence": source, "sentences": candidates},
            model=model,
        )
    except Exception:
        # Fallback: embed + cosine similarity
        def _l2(v):
            return v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-12)

        s = np.array(client.feature_extraction(source, model=model))
        C = np.array(client.feature_extraction(candidates, model=model))

        if s.ndim == 1:
            # Sometimes HF returns [hidden] or [[hidden]]
            if s.size and isinstance(s[0], (list, np.ndarray)):
                s = s[0][None, :]
            else:
                s = s[None, :]
        if C.ndim == 1:
            C = C[None, :]

        s, C = _l2(s)[0], _l2(C)
        return (C @ s).tolist()


# -----------------------------------------------------------------------------
# Shared policy / helpers (kept so imports don't break; real safety is remote)
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

HAZARDS = (
    "S1 Violent; S2 Non-violent Crimes; S3 Sex-crime; S4 CSE; S5 Defamation; "
    "S6 Specialized Illicit Advice; S7 Privacy; S8 IP; S9 Indiscriminate Weapons; "
    "S10 Hate; S11 Self-Harm; S12 Sexual; S13 Elections; S14 Code Interpreter Abuse; "
    "Malicious Code; Phishing/SE"
)

FORMAT_HELP = (
    "\nRespond EXACTLY in this format:\n"
    "DECISION: SAFE or UNSAFE\nCATEGORIES: [comma-separated subset from the list above]\n"
)
PROMPT_IN = (
    "Classify the USER INPUT as SAFE or UNSAFE; list categories from: {haz}."
    + FORMAT_HELP
    + "USER INPUT:\n{text}\n\nDECISION:"
)
PROMPT_OUT = (
    "Classify the MODEL RESPONSE as SAFE or UNSAFE; list categories from: {haz}."
    + FORMAT_HELP
    + "MODEL RESPONSE:\n{text}\n\nDECISION:"
)
PATTERN = re.compile(
    r"DECISION:\s*(SAFE|UNSAFE).*?CATEGORIES:\s*\[([^\]]*)\]", re.I | re.S
)


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


def refusal_message() -> str:
    return (
        "I can't help with that because it's outside my allowed use. "
        "Here's a safe alternative: high-level risks, mitigations, and references.\n"
    )


def build_prompt(
    question: str, context: str, *, role: str = "end_user", defense_only: bool = False
) -> str:
    """Kept for compatibility; real prompting is now in the remote backend."""
    question = (question or "").strip()
    context = (context or "")

    if not context.strip():
        return (
            f"{role_directive(role)}"
            "You are a telecom-cybersecurity assistant.\n"
            "Answer the question naturally and concisely.\n\n"
            f"Question:\n{question}\n\nAnswer:"
        )

    safety = (
        "Provide defensive mitigations only. Do NOT include exploit code, payloads, or targeting steps.\n"
        if defense_only
        else ""
    )

    return (
        f"{role_directive(role)}{safety}"
        "You are a telecom-cybersecurity assistant.\n"
        "- Use the Context when relevant to answer the question.\n"
        "- For greetings or small-talk, respond naturally.\n"
        "- If the question requires specific info not in context, say: 'Not enough evidence in context.'\n"
        "- Cite snippets with [D#]. No chain-of-thought. No sensitive data.\n\n"
        f"Context:\n{context.strip()}\n\nQuestion:\n{question}\n\nAnswer:"
    )


SAMPLING_PRESETS = {
    "factual": {"temperature": 0.3, "top_p": 0.9},
    "balanced": {"temperature": 0.7, "top_p": 0.9},
    "creative": {"temperature": 1.2, "top_p": 0.92},
}

# -----------------------------------------------------------------------------
# TELCO LLM BACKEND: LangServe via HTTP (ngrok)
# -----------------------------------------------------------------------------

TELCO_LLM_URL = os.getenv(
    "TELCO_LLM_URL",
    " https://3e885f090fc9.ngrok-free.app/ask_secure/invoke",
)
TELCO_LLM_TIMEOUT = int(os.getenv("TELCO_LLM_TIMEOUT", "120"))


def _call_telco_llm(inputs: Dict[str, Any]) -> str:
    """Low-level HTTP client for the Telco LLM LangServe endpoint.

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

    try:
        resp = requests.post(
            TELCO_LLM_URL, json=payload, timeout=TELCO_LLM_TIMEOUT
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        # Graceful error instead of crashing the whole graph
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
    """Clean RAAT/guard artifacts and arrow junk from the backend output.

    - Strips 'Rationale:' sections
    - Strips trailing 'Not enough evidence in context.'
    - Removes arrow noise like ->, -->, <=, =>
    - Collapses extra spaces / newlines
    """
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
# Guards (no-op on client; real safety lives in backend)
# -----------------------------------------------------------------------------
def guard_pre(user_text: str, role: str = "end_user"):
    """Client-side no-op guard.

    Real safety (Llama Guard + policy) happens inside the Colab backend.
    Kept so existing imports don't break.
    """
    return True, {
        "defense_only": False,
        "categories": [],
        "role": _canon_role(role),
    }


def guard_post(model_text: str, role: str = "end_user"):
    """Client-side no-op guard (see guard_pre)."""
    return True, {"categories": [], "role": _canon_role(role)}


# -----------------------------------------------------------------------------
# Main QA entrypoint
# -----------------------------------------------------------------------------
def ask_secure(
    question: str,
    *,
    context: str = "",
    role: str = "end_user",
    max_new_tokens: int = 400,
    preset: str = "balanced",
    seed: int | None = None,
) -> str:
    """Main QA entrypoint for the graph.

    Now delegates to the remote Telco LLM backend (Colab + LangServe)
    via the ngrok /ask_secure/invoke endpoint, and then cleans the output.
    """
    # Small-talk fast path (local, no HTTP)
    st = _smalltalk_reply(question)
    if st is not None:
        return st

    inputs: Dict[str, Any] = {
        "question": question,
        "context": context,
        "role": role,
        # Extra knobs (backend can ignore them safely)
        "max_new_tokens": max_new_tokens,
        "preset": preset,
        "seed": seed,
    }

    raw = _call_telco_llm(inputs)
    # Clean RAAT / guard decorations and arrow noise for user-facing answers
    return clean_answer(raw)


def generate_text(prompt: str, **decoding) -> str:
    """Compatibility wrapper: generate free-form text using the Telco LLM.

    Many RAG components call `generate_text(prompt, ...)`.
    Internally we route this through `ask_secure` with no context.
    """
    role = decoding.pop("role", "it_specialist")
    max_new_tokens = int(decoding.get("max_new_tokens", 400))

    return ask_secure(
        question=prompt,
        context="",
        role=role,
        max_new_tokens=max_new_tokens,
        preset=decoding.get("preset", "balanced"),
        seed=decoding.get("seed"),
    )
