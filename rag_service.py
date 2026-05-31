"""RAG 模式路由：前端可选「普通 RAG」或「GraphRAG」。"""
from __future__ import annotations

import os
from typing import Any

import chat_redis
import graphrag
import rag_milvus

DEFAULT_RAG_MODE = os.getenv("DEFAULT_RAG_MODE", "rag").strip().lower()


def normalize_rag_mode(mode: str | None, default: str | None = None) -> str:
    m = (mode or default or DEFAULT_RAG_MODE or "rag").strip().lower()
    if m in ("graphrag", "graph", "graph_rag"):
        return "graphrag"
    return "rag"


def mode_enabled(mode: str) -> bool:
    m = normalize_rag_mode(mode)
    if m == "graphrag":
        return graphrag.graphrag_enabled()
    return rag_milvus.rag_enabled()


def index_document(
    mode: str,
    session_id: str,
    file_id: str,
    source_name: str,
    text: str,
) -> dict[str, Any]:
    m = normalize_rag_mode(mode)
    if m == "graphrag":
        result = graphrag.safe_index_document(session_id, file_id, source_name, text)
    else:
        result = rag_milvus.safe_index_document(session_id, file_id, source_name, text)
    result["mode"] = m
    return result


def _group_file_ids_by_mode(
    session_id: str,
    file_ids: list[str],
    fallback_mode: str,
) -> dict[str, list[str]]:
    groups: dict[str, list[str]] = {"rag": [], "graphrag": []}
    fb = normalize_rag_mode(fallback_mode)
    seen: set[str] = set()
    for meta in chat_redis.get_upload_metas(session_id, file_ids):
        fid = (meta.get("file_id") or "").strip()
        if not fid:
            continue
        seen.add(fid)
        fm = normalize_rag_mode(meta.get("rag_mode"), fb)
        groups[fm].append(fid)
    for fid in file_ids:
        fid = (fid or "").strip()
        if fid and fid not in seen:
            groups[fb].append(fid)
    return groups


def build_rag_context(
    session_id: str,
    query: str,
    *,
    file_ids: list[str] | None = None,
    mode: str | None = None,
    top_k: int | None = None,
) -> str:
    """构建检索上下文；有附件时按各文件上传时的 rag_mode 分组检索。"""
    if file_ids:
        groups = _group_file_ids_by_mode(session_id, file_ids, mode or DEFAULT_RAG_MODE)
        parts: list[str] = []
        if groups["rag"]:
            ctx = rag_milvus.build_rag_context(
                session_id, query, top_k=top_k, file_ids=groups["rag"]
            )
            if ctx:
                parts.append(ctx)
        if groups["graphrag"]:
            ctx = graphrag.build_graphrag_context(
                session_id, query, top_k=top_k, file_ids=groups["graphrag"]
            )
            if ctx:
                parts.append(ctx)
        return "\n\n".join(parts)

    m = normalize_rag_mode(mode)
    if m == "graphrag":
        return graphrag.build_graphrag_context(session_id, query, top_k=top_k)
    return rag_milvus.build_rag_context(session_id, query, top_k=top_k)


def should_omit_attachment_body(
    session_id: str,
    file_ids: list[str],
    mode: str | None = None,
) -> bool:
    if not file_ids:
        return False
    groups = _group_file_ids_by_mode(session_id, file_ids, mode or DEFAULT_RAG_MODE)
    if groups["rag"] and rag_milvus.rag_enabled():
        return True
    if groups["graphrag"] and graphrag.graphrag_enabled():
        return True
    return False


def delete_session_indexes(session_id: str) -> None:
    rag_milvus.safe_delete_session_chunks(session_id)
    graphrag.safe_delete_session_graph(session_id)


def list_session_index(session_id: str, mode: str | None = None) -> list[dict[str, Any]]:
    m = normalize_rag_mode(mode)
    if m == "graphrag":
        return graphrag.list_session_graph_index(session_id)
    return rag_milvus.list_session_rag_index(session_id)


def list_all_session_indexes(session_id: str) -> dict[str, Any]:
    return {
        "rag": rag_milvus.list_session_rag_index(session_id),
        "graphrag": graphrag.list_session_graph_index(session_id),
    }
