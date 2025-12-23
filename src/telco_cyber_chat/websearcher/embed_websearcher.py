"""
embed_webscraper.py

Embed LlamaIndex TextNodes using a REMOTE BGE-M3 (dense+sparse) service exposed via LangServe.

Depends on:
  - telco_cyber_chat.embed_loader.get_hybrid_embeddings (aiohttp -> /embed/invoke)
  - numpy

Env:
  - BGE_EMBEDDING_URL  (recommended: https://<ngrok-domain>/embed  OR /embed/invoke)
  - BGE_EMBEDDING_TIMEOUT (seconds, default in embed_loader)

Goal:
  - keep heavy embedding model OUT of LangSmith deployment
  - embed nodes using your Colab/ngrok service safely (concurrency + retries)
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from llama_index.core.schema import TextNode

from telco_cyber_chat.embed_loader import get_hybrid_embeddings

logger = logging.getLogger(__name__)

SparseDict = Dict[int, float]


@dataclass
class HybridEmbedding:
    dense: Optional[np.ndarray]
    sparse: Optional[SparseDict]


def _node_text(node: TextNode) -> str:
    return str(getattr(node, "text", "") or "").strip()


def _node_metadata(node: TextNode) -> Dict[str, Any]:
    md = getattr(node, "metadata", None)
    return md if isinstance(md, dict) else {}


def _get_existing_id(node: TextNode) -> Optional[str]:
    v = getattr(node, "id_", None)
    v = str(v).strip() if v is not None else ""
    return v or None


def _set_node_id(node: TextNode, new_id: str) -> None:
    try:
        node.id_ = new_id
    except Exception:
        pass


def _generate_stable_id(node: TextNode) -> str:
    md = _node_metadata(node)
    url = md.get("url")
    if isinstance(url, str) and url.strip():
        raw = f"TextNode|url|{url.strip()}"
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    text = _node_text(node)
    try:
        md_norm = json.dumps(md, sort_keys=True, ensure_ascii=False)
    except Exception:
        md_norm = str(md)

    raw = f"TextNode|text|{text}|meta|{md_norm}"
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def ensure_textnode_ids(nodes: List[TextNode]) -> None:
    for n in nodes:
        if not _get_existing_id(n):
            _set_node_id(n, _generate_stable_id(n))


async def _embed_one(
    node: TextNode,
    sem: asyncio.Semaphore,
    *,
    retry_count: int = 3,
    delay_between_retries: float = 2.0,
    per_call_sleep: float = 0.10,
) -> Tuple[str, HybridEmbedding]:
    nid = _get_existing_id(node) or _generate_stable_id(node)
    _set_node_id(node, nid)

    text = _node_text(node)
    if not text:
        return nid, HybridEmbedding(dense=None, sparse=None)

    async with sem:
        for attempt in range(retry_count):
            try:
                dense, sparse = await get_hybrid_embeddings(text)

                if per_call_sleep:
                    await asyncio.sleep(per_call_sleep)

                return nid, HybridEmbedding(dense=dense, sparse=sparse)

            except Exception as e:
                logger.warning(
                    "Embedding failed (attempt %d/%d) for node=%s: %s",
                    attempt + 1,
                    retry_count,
                    nid,
                    e,
                )
                if attempt < retry_count - 1:
                    await asyncio.sleep(delay_between_retries * (attempt + 1))

        logger.error("All retries failed for node=%s", nid)
        return nid, HybridEmbedding(dense=None, sparse=None)


async def embed_textnodes_hybrid(
    nodes: List[TextNode],
    *,
    concurrency: int = 2,
    batch_size: int = 20,
    retry_count: int = 3,
    progress_callback: Optional[Callable[[int, int], None]] = None,
) -> Dict[str, HybridEmbedding]:
    if not nodes:
        return {}

    ensure_textnode_ids(nodes)

    total = len(nodes)
    out: Dict[str, HybridEmbedding] = {}

    for start in range(0, total, batch_size):
        batch = nodes[start : start + batch_size]
        sem = asyncio.Semaphore(concurrency)

        tasks = [
            _embed_one(
                n,
                sem,
                retry_count=retry_count,
                delay_between_retries=2.0,
                per_call_sleep=0.10,
            )
            for n in batch
        ]

        results = await asyncio.gather(*tasks)
        for nid, emb in results:
            out[nid] = emb

        done = min(start + batch_size, total)
        if progress_callback:
            progress_callback(done, total)

        if done < total:
            await asyncio.sleep(2.0)

    return out


def _run_coro_safely(coro):
    try:
        loop = asyncio.get_running_loop()
        running = loop.is_running()
    except RuntimeError:
        running = False

    if not running:
        return asyncio.run(coro)

    try:
        import nest_asyncio
        nest_asyncio.apply()
        loop = asyncio.get_event_loop()
        return loop.run_until_complete(coro)
    except Exception as e:
        raise RuntimeError(
            "Running inside an active event loop. "
            "Either `pip install nest_asyncio` or call the async function with `await`."
        ) from e


def embed_textnodes_hybrid_sync(
    nodes: List[TextNode],
    *,
    concurrency: int = 2,
    batch_size: int = 20,
    retry_count: int = 3,
    show_progress: bool = True,
) -> Dict[str, HybridEmbedding]:
    if not nodes:
        return {}

    if show_progress:
        try:
            from tqdm import tqdm

            pbar = tqdm(total=len(nodes), desc="Embedding TextNodes")

            def progress_fn(done, total):
                pbar.n = done
                pbar.refresh()

            result = _run_coro_safely(
                embed_textnodes_hybrid(
                    nodes,
                    concurrency=concurrency,
                    batch_size=batch_size,
                    retry_count=retry_count,
                    progress_callback=progress_fn,
                )
            )
            pbar.close()
            return result
        except Exception as e:
            logger.warning("Progress UI fallback: %s", e)

    def simple_progress(done, total):
        if done % 25 == 0 or done == total:
            print(f"Progress: {done}/{total}")

    return _run_coro_safely(
        embed_textnodes_hybrid(
            nodes,
            concurrency=concurrency,
            batch_size=batch_size,
            retry_count=retry_count,
            progress_callback=simple_progress,
        )
    )


def attach_dense_to_nodes(nodes: List[TextNode], results: Dict[str, HybridEmbedding]) -> None:
    """
    If your Qdrant ingest reads node.embedding, this sets it.
    (Sparse remains in results[nid].sparse.)
    """
    for n in nodes:
        nid = _get_existing_id(n)
        if not nid:
            continue
        emb = results.get(nid)
        if emb and emb.dense is not None:
            try:
                n.embedding = emb.dense.astype(np.float32).tolist()
            except Exception:
                pass
