"""Hybrid 检索：向量 + BM25 的 RRF 融合与 Parent 级去重。"""
from __future__ import annotations

import os
from typing import Any

from rag.rag_rerank import rerank_documents, rerank_enabled

RRF_K = int(os.getenv("RAG_RRF_K", "60"))
RERANK_TOP_N = int(os.getenv("RAG_RERANK_TOP_N", "10"))


def _parent_key(hit: dict[str, Any]) -> str:
    coll = hit.get("collection") or ""
    fid = hit.get("file_id") or ""
    parent_idx = hit.get("parent_index")
    if parent_idx is None:
        chunk_idx = hit.get("chunk_index")
        if chunk_idx is not None:
            parent_idx = chunk_idx
        else:
            parent_idx = hit.get("child_index", 0)
    return f"{coll}|{fid}|p{parent_idx}"


def _display_text(hit: dict[str, Any]) -> str:
    parent = (hit.get("parent_text") or "").strip()
    if parent:
        return parent
    return (hit.get("text") or "").strip()


def reciprocal_rank_fusion(
    ranked_lists: list[list[dict[str, Any]]],
    *,
    k: int | None = None,
    limit: int,
) -> list[dict[str, Any]]:
    """多路召回 RRF；同一 parent 只保留一条（取得分最高的 child 元数据）。"""
    rrf_k = k if k is not None else RRF_K
    scores: dict[str, float] = {}
    best_hit: dict[str, dict[str, Any]] = {}

    for results in ranked_lists:
        for rank, hit in enumerate(results):
            if not _display_text(hit):
                continue
            key = _parent_key(hit)
            scores[key] = scores.get(key, 0.0) + 1.0 / (rrf_k + rank + 1)
            prev = best_hit.get(key)
            hit_score = float(hit.get("score", 0.0))
            if prev is None or hit_score > float(prev.get("score", 0.0)):
                best_hit[key] = dict(hit)

    merged: list[dict[str, Any]] = []
    for key, rrf_score in sorted(scores.items(), key=lambda x: x[1], reverse=True):
        hit = dict(best_hit[key])
        hit["rrf_score"] = rrf_score
        hit["score"] = rrf_score
        merged.append(hit)
    return merged[:limit]


def rerank_hits(
    query: str,
    hits: list[dict[str, Any]],
    *,
    top_k: int,
    rerank_candidates: int | None = None,
) -> list[dict[str, Any]]:
    """对 parent 级候选做 Rerank；失败时回退 RRF 分数排序。"""
    if not hits:
        return []
    n_cand = rerank_candidates or RERANK_TOP_N
    candidates = hits[: max(n_cand, top_k)]
    docs = [_display_text(h) for h in candidates]
    if not rerank_enabled() or not docs:
        return candidates[:top_k]

    try:
        ranked = rerank_documents(query, docs, top_n=top_k)
    except Exception:
        return candidates[:top_k]

    if not ranked:
        return candidates[:top_k]

    out: list[dict[str, Any]] = []
    for item in ranked:
        idx = item.get("index")
        if idx is None or idx < 0 or idx >= len(candidates):
            continue
        hit = dict(candidates[idx])
        hit["rerank_score"] = float(item.get("score", 0.0))
        hit["score"] = hit["rerank_score"]
        out.append(hit)
    return out[:top_k] if out else candidates[:top_k]
