"""智谱 Rerank API：对 Hybrid 召回候选做精排。"""
from __future__ import annotations

import os
from typing import Any

import httpx

from config.env_config import get_zhipuai_api_key
from core.api_throttle import call_with_retry

RERANK_URL = os.getenv(
    "RAG_RERANK_URL", "https://open.bigmodel.cn/api/paas/v4/rerank"
)
RERANK_MODEL = os.getenv("RAG_RERANK_MODEL", "rerank")
RERANK_ENABLED = os.getenv("RAG_RERANK_ENABLED", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
RERANK_TIMEOUT_SEC = float(os.getenv("RAG_RERANK_TIMEOUT_SEC", "60"))


def rerank_enabled() -> bool:
    return RERANK_ENABLED


def rerank_documents(
    query: str,
    documents: list[str],
    *,
    top_n: int | None = None,
) -> list[dict[str, Any]]:
    """返回 [{index, score}, ...] 按相关性降序。"""
    q = (query or "").strip()
    docs = [((d or "").strip()) for d in documents]
    docs = [d for d in docs if d]
    if not q or not docs:
        return []

    payload: dict[str, Any] = {
        "model": RERANK_MODEL,
        "query": q,
        "documents": docs,
    }
    if top_n is not None and top_n > 0:
        payload["top_n"] = min(top_n, len(docs))

    def _call() -> dict[str, Any]:
        with httpx.Client(timeout=RERANK_TIMEOUT_SEC) as client:
            resp = client.post(
                RERANK_URL,
                headers={
                    "Authorization": f"Bearer {get_zhipuai_api_key()}",
                    "Content-Type": "application/json",
                },
                json=payload,
            )
            resp.raise_for_status()
            return resp.json()

    data = call_with_retry(_call, label="zhipu-rerank")
    raw = data.get("results") or data.get("data") or []
    out: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        idx = item.get("index")
        if idx is None:
            continue
        score = item.get("relevance_score")
        if score is None:
            score = item.get("score")
        out.append({"index": int(idx), "score": float(score or 0.0)})
    out.sort(key=lambda x: x["score"], reverse=True)
    return out
