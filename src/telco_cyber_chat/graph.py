import os
import re
import json
import logging
import hashlib
import ast
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple, Union

import numpy as np
import asyncio

# ===================== REMOTE BGE-M3 (NO LOCAL MODEL) =====================
from .embed_loader import get_query_embeddings
from .rerank_loader import get_rerank_scores  # <── NEW: remote reranker client

# Vector DB
from qdrant_client import QdrantClient, models as qmodels
from urllib.parse import urlparse, urlunparse

# LangChain bits
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, AIMessage, AnyMessage

# LangGraph
from langgraph.graph import StateGraph, START, END, MessagesState

# Remote helpers (your LLM only) - ASYNC
from .llm_loader import ask_secure_async

# ===================== Logging Configuration =====================
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(levelname)s: %(message)s",
)
log = logging.getLogger(__name__)

# ===================== Config / Secrets =====================
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "Telco_CyberChat")
if not QDRANT_URL or not QDRANT_API_KEY:
    raise RuntimeError("Set QDRANT_URL and QDRANT_API_KEY in env (use .env or Studio secrets).")

DEFAULT_ROLE = os.getenv("DEFAULT_ROLE", "it_specialist")

# Vector names in Qdrant
DENSE_NAME = os.getenv("DENSE_FIELD", "dense")
SPARSE_NAME = os.getenv("SPARSE_FIELD", "sparse")

# Sparse config (for lexical query → sparse vector)
BGE_TOKEN2ID_PATH = os.getenv("BGE_TOKEN2ID_PATH", "").strip()
SPARSE_MAX_TERMS = int(os.getenv("SPARSE_MAX_TERMS", "256"))
IDX_HASH_SIZE = int(os.getenv("IDX_HASH_SIZE", str(2**20)))

# Orchestration knobs
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "2"))
AGENT_TOPK_PER_STEP = int(os.getenv("AGENT_TOPK_PER_STEP", "3"))
AGENT_FORCE_FIRST_SEARCH = os.getenv("AGENT_FORCE_FIRST_SEARCH", "true").lower() == "true"
AGENT_MIN_STEPS = int(os.getenv("AGENT_MIN_STEPS", "1"))

RERANK_KEEP_TOPK = int(os.getenv("RERANK_KEEP_TOPK", "8"))
RERANK_PASS_THRESHOLD = float(os.getenv("RERANK_PASS_THRESHOLD", "0.25"))

SHORT_QUERY_MAX_WORDS = int(os.getenv("SHORT_QUERY_MAX_WORDS", "15"))

# ===================== Greeting/Goodbye Detection =====================
GREETING_REPLY = "Hello! I'm your telecom-cybersecurity assistant."
GOODBYE_REPLY = "Goodbye!"
THANKS_REPLY = "You're welcome!"

_GREET_RE = re.compile(
    r"^\s*(hi|hello|hey|greetings|good\s+(morning|afternoon|evening|day))\s*[!.?]*\s*$",
    re.I,
)
_BYE_RE = re.compile(
    r"^\s*(bye|goodbye|see\s+you|see\s+ya|thanks?\s*,?\s*bye|farewell)\s*[!.?]*\s*$",
    re.I,
)
_THANKS_RE = re.compile(r"^\s*(thanks|thank\s+you|thx|ty)\s*[!.?]*\s*$", re.I)


def _extract_text_from_query(query: Union[str, Dict, List[Any], Any]) -> str:
    """Extract plain user text from various query formats (LangSmith messages, dicts, etc.)."""
    if isinstance(query, list):
        parts: List[str] = []
        for item in query:
            if isinstance(item, dict):
                txt = item.get("text") or item.get("content")
                if isinstance(txt, str):
                    parts.append(txt)
            else:
                sub = _extract_text_from_query(item)
                if sub:
                    parts.append(sub)
        return " ".join(p.strip() for p in parts if p).strip()

    if isinstance(query, dict):
        return (str(query.get("text") or query.get("content") or "")).strip()

    if isinstance(query, str):
        s = query.strip()
        # Sometimes LangSmith wraps text as stringified dict {"text": "..."}
        if s.startswith("{") and s.endswith("}") and "text" in s:
            try:
                obj = ast.literal_eval(s)
                if isinstance(obj, dict):
                    return (str(obj.get("text") or obj.get("content") or "")).strip() or s
            except Exception:
                pass
        return s

    if hasattr(query, "text"):
        return str(query.text).strip()
    if hasattr(query, "content"):
        return str(query.content).strip()

    return str(query).strip()


def _is_greeting(text: Union[str, Dict, Any]) -> bool:
    return bool(_GREET_RE.search(_extract_text_from_query(text)))


def _is_goodbye(text: Union[str, Dict, Any]) -> bool:
    return bool(_BYE_RE.search(_extract_text_from_query(text)))


def _is_thanks(text: Union[str, Dict, Any]) -> bool:
    return bool(_THANKS_RE.search(_extract_text_from_query(text)))


def _is_smalltalk(text: Union[str, Dict, Any]) -> bool:
    s = _extract_text_from_query(text)
    return _is_greeting(s) or _is_goodbye(s) or _is_thanks(s)


# ===================== Qdrant helpers (LAZY ASYNC) =====================
def _normalize_qdrant_url(raw: str) -> str:
    u = urlparse(raw)
    scheme = u.scheme or "https"
    netloc = u.netloc or u.path
    if scheme == "https" and ":" in netloc:
        netloc = netloc.split(":", 1)[0]
    return urlunparse((scheme, netloc, "", "", "", ""))


def _make_qdrant_client(url: str, api_key: Optional[str]) -> QdrantClient:
    """Blocking Qdrant client creation (called in thread pool)."""
    url_norm = _normalize_qdrant_url(url)
    client = QdrantClient(url=url_norm, api_key=api_key, timeout=15.0)
    try:
        _ = client.get_collections()
    except Exception as e:
        u = urlparse(url_norm)
        if u.scheme == "http" and ":" not in u.netloc:
            client = QdrantClient(
                host=u.hostname, port=6333, api_key=api_key, https=False, timeout=15.0
            )
            _ = client.get_collections()
        else:
            raise RuntimeError(
                f"Could not reach Qdrant at '{url}'. Normalized: '{url_norm}'. "
                f"Original error: {repr(e)}"
            )
    return client


_qdrant_client: Optional[QdrantClient] = None
_qdrant_lock = asyncio.Lock()


async def get_qdrant_client() -> QdrantClient:
    """Lazy async Qdrant client initialization."""
    global _qdrant_client

    if _qdrant_client is not None:
        return _qdrant_client

    async with _qdrant_lock:
        if _qdrant_client is not None:
            return _qdrant_client

        log.info("[QDRANT] Initializing client (lazy async load)")
        loop = asyncio.get_running_loop()

        _qdrant_client = await loop.run_in_executor(
            None, lambda: _make_qdrant_client(QDRANT_URL, QDRANT_API_KEY)
        )

        # Verify collection exists (also in thread pool)
        await loop.run_in_executor(
            None,
            lambda: _qdrant_client.scroll(
                collection_name=QDRANT_COLLECTION,
                limit=1,
                with_payload=False,
            ),
        )

        log.info("[QDRANT] Client initialized successfully")
        return _qdrant_client


# ===================== Lexical sparse (query side) =====================
_token2id_cache: Optional[Dict[str, int]] = None


def _get_token2id() -> Dict[str, int]:
    """Load token2id mapping (cached)."""
    global _token2id_cache

    if _token2id_cache is not None:
        return _token2id_cache

    if BGE_TOKEN2ID_PATH and os.path.exists(BGE_TOKEN2ID_PATH):
        try:
            with open(BGE_TOKEN2ID_PATH, "r", encoding="utf-8") as f:
                mapping = json.load(f)
            if isinstance(mapping, dict) and "vocab" in mapping and isinstance(mapping["vocab"], dict):
                mapping = mapping["vocab"]
            _token2id_cache = {str(k): int(v) for k, v in mapping.items()}
            return _token2id_cache
        except Exception as e:
            log.warning(f"[BGE] Failed to load token2id at {BGE_TOKEN2ID_PATH}: {e}")

    log.warning(
        "[BGE] No token2id mapping provided. Falling back to hashing buckets "
        f"(size={IDX_HASH_SIZE}). For best sparse recall, set BGE_TOKEN2ID_PATH."
    )
    _token2id_cache = {}
    return _token2id_cache


_WORD_RE = re.compile(r"[A-Za-z0-9_]+(?:-[A-Za-z0-9_]+)?", re.U)


def _tokenize_query_simple(q: str) -> List[str]:
    return [t.lower() for t in _WORD_RE.findall(q or "") if t.strip()]


def _hash_idx(term: str) -> int:
    h = hashlib.sha1(term.encode("utf-8", errors="ignore")).hexdigest()
    return int(h, 16) % IDX_HASH_SIZE


def _lexicalize_query(q: str) -> qmodels.SparseVector:
    """Lexical sparse vector creation (fast, stays sync)."""
    toks = _tokenize_query_simple(q)
    if not toks:
        return qmodels.SparseVector(indices=[], values=[])

    counts: Dict[str, int] = {}
    for t in toks:
        counts[t] = counts.get(t, 0) + 1
    max_tf = max(counts.values())

    items = [(t, (counts[t] ** 0.5) / (max_tf ** 0.5 + 1e-9)) for t in counts]
    if SPARSE_MAX_TERMS > 0 and len(items) > SPARSE_MAX_TERMS:
        items = sorted(items, key=lambda kv: kv[1], reverse=True)[:SPARSE_MAX_TERMS]

    token2id = _get_token2id()
    indices, values = [], []
    if token2id:
        for tok, w in items:
            tid = token2id.get(tok)
            if tid is not None:
                indices.append(int(tid))
                values.append(float(w))
    else:
        for tok, w in items:
            indices.append(_hash_idx(tok))
            values.append(float(w))

    return qmodels.SparseVector(indices=indices, values=values)


# ===================== REMOTE BGE-M3 DENSE EMBEDDING =====================

async def _embed_bge_remote_async(text: str) -> Optional[List[float]]:
    """
    Get dense embedding from REMOTE BGE-M3 service (via embed_loader.get_query_embeddings).
    NO local model loading inside LangSmith runtime.
    """
    if not text:
        return None

    try:
        log.debug(f"[DENSE] Requesting remote embedding for: '{text[:50]}...'")

        dense_vec, _ = await get_query_embeddings(
            text,
            return_dense=True,
            return_sparse=False,
            max_length=8192,
        )

        if dense_vec is None:
            log.error("[DENSE] Remote service returned None")
            return None

        vec = np.array(dense_vec, dtype="float32")
        n = np.linalg.norm(vec) + 1e-12
        normalized = (vec / n).tolist()

        log.debug(f"[DENSE] Got embedding: dim={len(normalized)}")
        return normalized

    except Exception as e:
        log.error(f"[DENSE] Remote BGE-M3 embedding failed: {e}")
        return None


# ===================== Retrieval =====================

@dataclass
class RetrievalCfg:
    top_k: int = 6
    rrf_k: int = 60
    alpha_dense: float = 0.6
    overfetch: int = 3
    dense_name: str = DENSE_NAME
    sparse_name: str = SPARSE_NAME
    text_key: str = "node_content"
    source_key: str = "node_id"


CFG = RetrievalCfg()


async def _search_dense_async(q: str, k: int):
    """Async dense search with REMOTE BGE-M3."""
    dense_vec = await _embed_bge_remote_async(q)
    if dense_vec is None:
        log.warning("[SEARCH] Failed to get dense embedding, skipping dense search")
        return []

    try:
        client = await get_qdrant_client()
        loop = asyncio.get_running_loop()

        resp = await loop.run_in_executor(
            None,
            lambda: client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=dense_vec,
                using=CFG.dense_name,
                limit=k,
                with_payload=True,
                with_vectors=False,
            ),
        )
        log.debug(f"[SEARCH] Dense search returned {len(resp.points)} points")
        return resp.points
    except Exception as e:
        log.warning(f"Dense search failed: {e}")
        return []


async def _search_sparse_async(q: str, k: int):
    """Async sparse search (lexical query embedding)."""
    sparse_vec = _lexicalize_query(q)

    if not getattr(sparse_vec, "indices", None):
        log.warning("[SEARCH] No sparse indices generated, skipping sparse search")
        return []

    try:
        client = await get_qdrant_client()
        loop = asyncio.get_running_loop()

        resp = await loop.run_in_executor(
            None,
            lambda: client.query_points(
                collection_name=QDRANT_COLLECTION,
                query=sparse_vec,
                using=CFG.sparse_name,
                limit=k,
                with_payload=True,
                with_vectors=False,
            ),
        )
        log.debug(f"[SEARCH] Sparse search returned {len(resp.points)} points")
        return resp.points
    except Exception as e:
        log.warning(f"Sparse search failed: {e}")
        return []


def _rrf_fuse(dense_hits, sparse_hits, k_rrf: int, alpha_dense: float):
    """Reciprocal Rank Fusion of dense + sparse hits."""
    def rankmap(h): return {str(x.id): r for r, x in enumerate(h, 1)}

    rd, rs = rankmap(dense_hits or []), rankmap(sparse_hits or [])
    ids = set(rd) | set(rs)
    fused = []
    for pid in ids:
        sd = 1.0 / (k_rrf + rd.get(pid, 10**6))
        ss = 1.0 / (k_rrf + rs.get(pid, 10**6))
        score = alpha_dense * sd + (1.0 - alpha_dense) * ss
        hit = next((h for h in (dense_hits or []) if str(h.id) == pid), None) or \
            next((h for h in (sparse_hits or []) if str(h.id) == pid), None)
        fused.append((score, hit))
    fused.sort(key=lambda t: t[0], reverse=True)
    return [h for _, h in fused]


def _to_docs(points) -> List[Document]:
    """Convert Qdrant points to LangChain Documents."""
    docs = []
    for i, h in enumerate(points or [], 1):
        pl = h.payload or {}
        point_id = str(getattr(h, "id", f"doc{i}"))
        src = pl.get(CFG.source_key, "")
        docs.append(
            Document(
                page_content=str(pl.get(CFG.text_key, "") or "")[:2000],
                metadata={
                    "doc_id": point_id,
                    "source": src,
                    "score": float(getattr(h, "score", 0.0) or 0.0),
                },
            )
        )
    return docs


async def hybrid_search_async(q: str, top_k: int = None) -> List[Document]:
    """
    Async hybrid search with REMOTE embeddings and lazy initialization.
    All blocking operations are done in thread pools.
    """
    k = top_k or CFG.top_k

    log.info(f"[HYBRID_SEARCH] Query: '{q[:50]}...', top_k={k}")

    dense_hits, sparse_hits = await asyncio.gather(
        _search_dense_async(q, k * CFG.overfetch),
        _search_sparse_async(q, k * CFG.overfetch),
    )

    fused = _rrf_fuse(dense_hits, sparse_hits, CFG.rrf_k, CFG.alpha_dense)[:k]
    docs = _to_docs(fused)

    log.info(f"[HYBRID_SEARCH] Returned {len(docs)} documents")
    return docs


# ===================== Graph state / helpers =====================

class ChatState(MessagesState):
    query: str
    intent: str
    docs: List[Document]
    answer: str
    eval: Dict[str, Any]
    trace: List[str]
    cot: Dict[str, Any]


def _coerce_str(x: Any) -> str:
    if isinstance(x, str):
        return x
    if isinstance(x, list):
        return "\n".join(_coerce_str(e) for e in x if e is not None)
    if x is None:
        return ""
    return str(x)


def _last_user(state: "ChatState") -> Any:
    for msg in reversed(state.get("messages", [])):
        if isinstance(msg, HumanMessage):
            return getattr(msg, "content", "")
    return state.get("query", "")


def _fmt_ctx(docs: List[Document], cap: int = 12) -> str:
    out = []
    for i, d in enumerate(docs[:cap], 1):
        did = d.metadata.get("doc_id") or d.metadata.get("source") or f"doc{i}"
        chunk = _coerce_str(d.page_content).strip()
        out.append(f"[{did}] {chunk[:1200]}")
    return "\n\n".join(out) if out else "No context."


# ===================== Reranker (REMOTE, via web service) =====================

async def _apply_rerank_for_query_async(
    docs: List[Document], q: str, keep: int = RERANK_KEEP_TOPK
) -> List[Document]:
    """
    Async reranking using REMOTE BGE reranker service (LangServe) via get_rerank_scores.
    No local FlagEmbedding model is loaded inside LangSmith runtime.
    """
    if not docs:
        return []

    texts = [_coerce_str(d.page_content)[:1024] for d in docs]

    # Call remote reranker
    scores = await get_rerank_scores(q, texts)

    if scores is None:
        log.warning("[RERANK] Remote reranker unavailable; skipping rerank.")
        return docs[:keep]

    if len(scores) != len(docs):
        log.warning(
            f"[RERANK] Score/doc length mismatch (scores={len(scores)}, docs={len(docs)}); skipping rerank."
        )
        return docs[:keep]

    ranked = sorted(zip(docs, scores), key=lambda t: float(t[1]), reverse=True)
    for d, s in ranked:
        d.metadata["rerank_score"] = float(s)
    return [d for d, _ in ranked][:keep]


def _avg_rerank(docs: List[Document], k: int = 5) -> float:
    if not docs:
        return 0.0
    vals = []
    for d in docs[:k]:
        s = d.metadata.get("rerank_score", None)
        if s is not None:
            vals.append(float(s))
    return (sum(vals) / len(vals)) if vals else 0.0


def _infer_role(intent: str) -> str:
    if intent == "policy":
        return "admin"
    if intent in ("diagnostic", "incident", "mitigation"):
        return "network_admin"
    return DEFAULT_ROLE


# ===================== Orchestrator (heuristic classify & route) =====================

def _wordcount_classify(q: str) -> Tuple[str, str, str]:
    """Heuristic classifier based on word count."""
    tokens = re.findall(r"\w+", q)
    wc = len(tokens)

    if wc <= SHORT_QUERY_MAX_WORDS:
        clarity = "clear"
        reasoning = f"Query has {wc} words (<= {SHORT_QUERY_MAX_WORDS}); single-hop ReAct."
    else:
        clarity = "multi-hop"
        reasoning = f"Query has {wc} words (> {SHORT_QUERY_MAX_WORDS}); multi-hop."

    intent = "informational"
    return intent, clarity, reasoning


async def orchestrator_node(state: ChatState) -> Dict:
    """Async orchestrator node."""
    q_raw = _last_user(state) or state.get("query") or ""
    q = _extract_text_from_query(q_raw)

    log.info(f"[ORCHESTRATOR] Query: '{q}'")

    cot = dict(state.get("cot") or {})

    # Small talk short-circuit
    if _is_greeting(q):
        log.info("[ORCHESTRATOR] GREETING detected")
        return {
            "query": q,
            "intent": "greeting",
            "eval": {
                "intent": "greeting",
                "clarity": "clear",
                "role": DEFAULT_ROLE,
                "skip_rag": True,
                "skip_reason": "smalltalk_greeting",
            },
            "messages": [AIMessage(content=GREETING_REPLY)],
            "answer": GREETING_REPLY,
            "docs": [],
            "trace": ["orchestrator(greeting)->END"],
            "cot": cot,
        }

    if _is_goodbye(q):
        log.info("[ORCHESTRATOR] GOODBYE detected")
        return {
            "query": q,
            "intent": "goodbye",
            "eval": {
                "intent": "goodbye",
                "clarity": "clear",
                "role": DEFAULT_ROLE,
                "skip_rag": True,
                "skip_reason": "smalltalk_goodbye",
            },
            "messages": [AIMessage(content=GOODBYE_REPLY)],
            "answer": GOODBYE_REPLY,
            "docs": [],
            "trace": ["orchestrator(goodbye)->END"],
            "cot": cot,
        }

    if _is_thanks(q):
        log.info("[ORCHESTRATOR] THANKS detected")
        return {
            "query": q,
            "intent": "thanks",
            "eval": {
                "intent": "thanks",
                "clarity": "clear",
                "role": DEFAULT_ROLE,
                "skip_rag": True,
                "skip_reason": "smalltalk_thanks",
            },
            "messages": [AIMessage(content=THANKS_REPLY)],
            "answer": THANKS_REPLY,
            "docs": [],
            "trace": ["orchestrator(thanks)->END"],
            "cot": cot,
        }

    # Very short queries → ask user to clarify
    word_count = len(re.findall(r"\w+", q))
    if word_count <= 2 and not q.endswith("?"):
        log.info(f"[ORCHESTRATOR] Very short query ({word_count} words)")
        msg = "I'm here to help with telecom and cybersecurity questions. What would you like to know?"
        return {
            "query": q,
            "intent": "smalltalk",
            "eval": {
                "intent": "smalltalk",
                "clarity": "clear",
                "role": DEFAULT_ROLE,
                "skip_rag": True,
                "skip_reason": "too_short",
            },
            "messages": [AIMessage(content=msg)],
            "answer": msg,
            "docs": [],
            "trace": ["orchestrator(smalltalk_short)->END"],
            "cot": cot,
        }

    # Heuristic classification
    intent, clarity, orch_reasoning = _wordcount_classify(q)
    role = _infer_role(intent)

    ev = dict(state.get("eval") or {})
    ev.update(
        {
            "intent": intent,
            "clarity": clarity,
            "role": role,
            "orch_model": "heuristic_wordcount_v1",
            "orch_steps": int(ev.get("orch_steps", 0)) + 1,
            "skip_rag": False,
        }
    )

    if orch_reasoning:
        cot["orchestrator"] = orch_reasoning

    return {
        "query": q,
        "intent": intent,
        "eval": ev,
        "cot": cot,
        "trace": state.get("trace", [])
        + [f"orchestrator(intent={intent},clarity={clarity})->react"],
    }


async def route_orchestrator(state: ChatState) -> str:
    """Async router for orchestrator."""
    ev = state.get("eval") or {}

    if ev.get("skip_rag"):
        log.info("[ROUTE_ORCH] skip_rag=True -> END")
        return "end"

    log.info("[ROUTE_ORCH] -> react")
    return "react"


# ===================== Agents =====================

async def react_loop_node(state: ChatState) -> Dict:
    """Simple ReAct with loop detection and safety limits."""
    q = state["query"]
    ev = dict(state.get("eval") or {})
    step = int(ev.get("react_step", 0))
    docs = state.get("docs", []) or []

    # Loop counter for safety
    loop_count = int(ev.get("loop_counter", 0))
    ev["loop_counter"] = loop_count + 1

    if loop_count > 10:
        log.error(f"[REACT] INFINITE LOOP DETECTED! Loop count: {loop_count}")
        raise RuntimeError("React loop exceeded safety limit (10 iterations)")

    # Hard stop at max steps
    if step >= AGENT_MAX_STEPS:
        log.warning(f"[REACT] HARD STOP at step {step}/{AGENT_MAX_STEPS}")
        ev["force_exit"] = True
        return {
            "docs": docs,
            "eval": ev,
            "trace": state.get("trace", [])
            + [f"react_step({step}) HARD_STOP"],
        }

    # Safety: abort on small talk
    if _is_smalltalk(q):
        log.warning(f"[REACT] Small talk detected: '{q}'")
        ev["force_exit"] = True
        return {
            "docs": docs,
            "eval": ev,
            "trace": state.get("trace", [])
            + [f"react_step({step}) ABORTED - smalltalk"],
        }

    # Core search (REMOTE embeddings)
    log.info(f"[REACT] Starting step {step}, loop_count={loop_count}")
    hop_docs = await hybrid_search_async(q, top_k=AGENT_TOPK_PER_STEP)
    docs = docs + hop_docs

    ev.setdefault("queries", []).append(q)
    ev["react_step"] = step + 1
    ev["last_agent"] = "react"

    log.info(f"[REACT] Completed step {step}: got {len(hop_docs)} docs (total={len(docs)})")

    return {
        "docs": docs,
        "eval": ev,
        "trace": state.get("trace", [])
        + [f"react_step({step}, q='{q[:30]}...') -> {len(docs)} docs"],
    }


async def self_ask_loop_node(state: ChatState) -> Dict:
    """Currently not used, left for future experiments."""
    q = state["query"]
    ev = dict(state.get("eval") or {})
    subqs = ev.get("selfask_subqs")
    idx = int(ev.get("selfask_idx", 0))
    docs = state.get("docs", []) or []

    if not subqs:
        subqs = [q]
        ev["selfask_subqs"] = subqs
        idx = 0

    subq = subqs[min(idx, max(0, len(subqs) - 1))]
    hop_docs = await hybrid_search_async(subq, top_k=AGENT_TOPK_PER_STEP)
    docs = docs + hop_docs

    ev.setdefault("queries", []).append(subq)
    ev["selfask_idx"] = idx + 1
    ev["last_agent"] = "self_ask"

    return {
        "docs": docs,
        "eval": ev,
        "trace": state.get("trace", [])
        + [f"self_ask_step({idx} '{subq}') -> {len(docs)} docs"],
    }


# ===================== Reranker (pass/fail gating) =====================

async def reranker_node(state: ChatState) -> Dict:
    """Async reranker with REMOTE BGE reranker service."""
    q = state["query"]
    ev = dict(state.get("eval") or {})
    docs = state.get("docs", []) or []

    log.info(f"[RERANK] Entry: react_step={ev.get('react_step', 0)}, docs={len(docs)}")

    if not docs:
        log.warning("[RERANK] No docs to rerank!")
        return {"eval": ev, "trace": state.get("trace", []) + ["rerank(EMPTY)"]}

    docs2 = await _apply_rerank_for_query_async(docs, q, RERANK_KEEP_TOPK)
    avg_top = _avg_rerank(docs2, k=min(5, len(docs2)))
    ev["avg_rerank_top"] = float(avg_top)

    react_steps = int(ev.get("react_step", 0))
    last_agent = ev.get("last_agent", "react")

    pass_gate = avg_top >= RERANK_PASS_THRESHOLD
    budget_exhausted = (last_agent == "react" and react_steps >= AGENT_MAX_STEPS)

    decision = "final" if (pass_gate or budget_exhausted) else "retry_react"

    log.info(
        f"[RERANK] Decision: {decision} "
        f"(avg={avg_top:.3f}, threshold={RERANK_PASS_THRESHOLD}, "
        f"steps={react_steps}/{AGENT_MAX_STEPS}, pass_gate={pass_gate}, "
        f"budget_exhausted={budget_exhausted})"
    )

    return {
        "docs": docs2,
        "eval": ev,
        "trace": state.get("trace", [])
        + [f"rerank(avg={avg_top:.3f}, keep={len(docs2)}) -> {decision}"],
    }


async def route_rerank(state: ChatState) -> str:
    """Async router for rerank decisions."""
    ev = state.get("eval") or {}

    if ev.get("force_exit"):
        log.warning("[ROUTE_RERANK] Force exit detected -> final")
        return "final"

    last = ev.get("last_agent", "react")
    avg_top = float(ev.get("avg_rerank_top", 0.0))
    react_steps = int(ev.get("react_step", 0))
    loop_count = int(ev.get("loop_counter", 0))

    if loop_count > 10:
        log.error(f"[ROUTE_RERANK] Loop count {loop_count} exceeded -> forcing final")
        return "final"

    pass_gate = avg_top >= RERANK_PASS_THRESHOLD
    budget_exhausted = (last == "react" and react_steps >= AGENT_MAX_STEPS)
    min_steps_met = react_steps >= AGENT_MIN_STEPS

    if not min_steps_met and not budget_exhausted:
        log.info(
            f"[ROUTE_RERANK] Enforcing MIN_STEPS: {react_steps}/{AGENT_MIN_STEPS} -> retry_react"
        )
        return "retry_react"

    if pass_gate or budget_exhausted:
        log.info(
            f"[ROUTE_RERANK] -> final "
            f"(avg={avg_top:.3f}, pass={pass_gate}, steps={react_steps}/{AGENT_MAX_STEPS})"
        )
        return "final"

    log.info(
        f"[ROUTE_RERANK] -> retry_react "
        f"(avg={avg_top:.3f} < {RERANK_PASS_THRESHOLD}, steps={react_steps}/{AGENT_MAX_STEPS})"
    )
    return "retry_react"


# ===================== LLM (final answer) =====================

async def llm_node(state: ChatState) -> Dict:
    """Async LLM generation node"""
    docs = state.get("docs", [])
    role = (state.get("eval") or {}).get("role") or DEFAULT_ROLE

    if not docs:
        msg = (
            f"No evidence found in Qdrant collection '{QDRANT_COLLECTION}'. I won't fabricate an answer.\n\n"
            "Troubleshooting:\n"
            "- Verify the collection has points (and correct payload keys)\n"
            f"- Ensure dense name='{DENSE_NAME}' and sparse name='{SPARSE_NAME}' match your collection\n"
            "- Ensure embeddings match BGE-M3 (dense dim=1024) and sparse vocab mapping or hashing size"
        )
        return {
            "messages": [AIMessage(content=msg)],
            "answer": msg,
            "trace": state.get("trace", []) + ["llm(NO_CONTEXT)"],
        }

    try:
        text = await ask_secure_async(
            question=state["query"],
            context=_fmt_ctx(docs, cap=12),
            role=role,
            preset="factual",
            max_new_tokens=400,
        )
    except Exception as e:
        log.error(f"[LLM] ask_secure_async failed: {e}")
        msg = (
            "The retrieval step succeeded, but the LLM backend timed out or failed while generating the answer. "
            "Please retry your question later or check the LLM provider configuration."
        )
        return {
            "messages": [AIMessage(content=msg)],
            "answer": msg,
            "trace": state.get("trace", []) + ["llm(ERROR_BACKEND)"],
        }

    return {
        "messages": [AIMessage(content=text)],
        "answer": text,
        "trace": state.get("trace", []) + ["llm"],
    }


# ===================== Graph wiring =====================

state_graph = StateGraph(ChatState)

# Add nodes
state_graph.add_node("orchestrator", orchestrator_node)
state_graph.add_node("react_loop", react_loop_node)
state_graph.add_node("self_ask_loop", self_ask_loop_node)
state_graph.add_node("rerank", reranker_node)
state_graph.add_node("llm", llm_node)

state_graph.add_edge(START, "orchestrator")

# Async routers
state_graph.add_conditional_edges(
    "orchestrator",
    route_orchestrator,
    {
        "react": "react_loop",
        "self_ask": "self_ask_loop",
        "end": END,
    },
)

state_graph.add_edge("react_loop", "rerank")
state_graph.add_edge("self_ask_loop", "rerank")

state_graph.add_conditional_edges(
    "rerank",
    route_rerank,
    {
        "retry_react": "react_loop",
        "retry_self_ask": "self_ask_loop",
        "final": "llm",
    },
)

state_graph.add_edge("llm", END)

graph = state_graph.compile()


# ===================== Top-level helpers =====================

def chat_with_greeting_precheck(query: Union[str, Dict, Any], **kwargs):
    """
    Top-level entry point with fast path for greetings / small talk.

    NOTE: This is a SYNC wrapper around the compiled graph, which internally
    runs async nodes safely (no asyncio.run here).
    """
    q = _extract_text_from_query(query)
    log.info(f"[PRE-CHECK] Query: '{q}'")

    if _is_greeting(q):
        log.info("[PRE-CHECK] GREETING - fast path")
        return {
            "messages": [HumanMessage(content=q), AIMessage(content=GREETING_REPLY)],
            "answer": GREETING_REPLY,
            "query": q,
            "intent": "greeting",
            "docs": [],
            "eval": {"intent": "greeting", "skip_reason": "pre_check"},
            "trace": ["pre_check_greeting"],
        }

    if _is_goodbye(q):
        log.info("[PRE-CHECK] GOODBYE - fast path")
        return {
            "messages": [HumanMessage(content=q), AIMessage(content=GOODBYE_REPLY)],
            "answer": GOODBYE_REPLY,
            "query": q,
            "intent": "goodbye",
            "docs": [],
            "eval": {"intent": "goodbye", "skip_reason": "pre_check"},
            "trace": ["pre_check_goodbye"],
        }

    if _is_thanks(q):
        log.info("[PRE-CHECK] THANKS - fast path")
        return {
            "messages": [HumanMessage(content=q), AIMessage(content=THANKS_REPLY)],
            "answer": THANKS_REPLY,
            "query": q,
            "intent": "thanks",
            "docs": [],
            "eval": {"intent": "thanks", "skip_reason": "pre_check"},
            "trace": ["pre_check_thanks"],
        }

    log.info("[PRE-CHECK] Technical query - invoking graph")
    initial_state = {
        "query": q,
        "messages": [HumanMessage(content=q)],
        "trace": [],
        "cot": {},
    }
    return graph.invoke(initial_state)


# Wrap graph.invoke to catch any unexpected internal errors
_original_graph_invoke = graph.invoke


def _wrapped_graph_invoke(input_data, *args, **kwargs):
    """Global safety wrapper for graph.invoke to handle errors gracefully."""
    if isinstance(input_data, dict):
        query = input_data.get("query")
        if not query and input_data.get("messages"):
            last_msg = input_data["messages"][-1] if input_data["messages"] else None
            if isinstance(last_msg, dict):
                query = last_msg.get("content", "")
            elif hasattr(last_msg, "content"):
                query = last_msg.content
        if query is None:
            query = ""
    else:
        query = input_data

    q = _extract_text_from_query(query)
    log.info(f"[INVOKE-WRAPPER] Query: '{q}'")

    if _is_greeting(q):
        log.info("[INVOKE-WRAPPER] GREETING - bypassing graph")
        return {
            "messages": [HumanMessage(content=q), AIMessage(content=GREETING_REPLY)],
            "answer": GREETING_REPLY,
            "query": q,
            "intent": "greeting",
            "docs": [],
            "eval": {"intent": "greeting", "skip_reason": "invoke_wrapper"},
            "trace": ["invoke_wrapper_greeting"],
        }

    if _is_goodbye(q):
        log.info("[INVOKE-WRAPPER] GOODBYE - bypassing graph")
        return {
            "messages": [HumanMessage(content=q), AIMessage(content=GOODBYE_REPLY)],
            "answer": GOODBYE_REPLY,
            "query": q,
            "intent": "goodbye",
            "docs": [],
            "eval": {"intent": "goodbye", "skip_reason": "invoke_wrapper"},
            "trace": ["invoke_wrapper_goodbye"],
        }

    if _is_thanks(q):
        log.info("[INVOKE-WRAPPER] THANKS - bypassing graph")
        return {
            "messages": [HumanMessage(content=q), AIMessage(content=THANKS_REPLY)],
            "answer": THANKS_REPLY,
            "query": q,
            "intent": "thanks",
            "docs": [],
            "eval": {"intent": "thanks", "skip_reason": "invoke_wrapper"},
            "trace": ["invoke_wrapper_thanks"],
        }

    if isinstance(input_data, dict):
        input_data["query"] = q
        input_data.setdefault("cot", {})
        if not input_data.get("messages"):
            input_data["messages"] = [HumanMessage(content=q)]

    try:
        return _original_graph_invoke(input_data, *args, **kwargs)
    except Exception as e:
        log.error(f"[INVOKE-WRAPPER] Unhandled error in graph.invoke: {e}")
        msg = (
            "An internal error occurred while running the RAG graph, "
            "likely due to a timeout or failure from the upstream LLM/embedding service. "
            "Please retry later or adjust your LLM / embedding provider configuration."
        )
        return {
            "messages": [HumanMessage(content=q), AIMessage(content=msg)],
            "answer": msg,
            "query": q,
            "intent": "error",
            "docs": [],
            "eval": {
                "intent": "error",
                "error": str(e),
                "skip_reason": "invoke_wrapper_error",
            },
            "trace": ["invoke_wrapper_FATAL"],
        }


graph.invoke = _wrapped_graph_invoke

__all__ = [
    "graph",
    "chat_with_greeting_precheck",
    "hybrid_search_async",
]
