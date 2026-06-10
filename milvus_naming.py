"""Milvus 命名规范：一会话一个 Database，集合为 rag_/graphrag_ + 文件名（仅 ASCII）。"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

# Milvus 集合/库名：仅字母、数字、下划线
_ASCII_SLUG_RE = re.compile(r"[^0-9a-zA-Z_]+")
_MILVUS_NAME_RE = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

RAG_COLLECTION_PREFIX = "rag"
GRAPHRAG_COLLECTION_PREFIX = "graphrag"

LEGACY_RAG_COLLECTION_PREFIX = "doc"
LEGACY_GRAPHRAG_COLLECTION_PREFIX = "graph"


def is_valid_milvus_name(name: str) -> bool:
    n = (name or "").strip()
    return bool(n and len(n) <= 255 and _MILVUS_NAME_RE.match(n))


def _ascii_slug(value: str, *, max_len: int = 48, prefix_if_digit: str = "d") -> str:
    """转为 Milvus 合法标识：仅保留 ASCII 字母数字下划线；纯中文等则回退为哈希。"""
    raw = (value or "").strip() or "x"
    ascii_only = _ASCII_SLUG_RE.sub("_", raw).strip("_")
    if len(ascii_only) >= 2:
        s = ascii_only[:max_len].strip("_") or "x"
    else:
        digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]
        s = f"h_{digest}"
    if s[0].isdigit():
        s = f"{prefix_if_digit}_{s}"
    return s[:max_len] or "x"


def norm_session_id(session_id: str) -> str:
    return (session_id or "default").strip() or "default"


def session_database_name(session_id: str) -> str:
    """一会话一个 Milvus Database（仅 ASCII）。"""
    return _ascii_slug(norm_session_id(session_id), max_len=64, prefix_if_digit="s")


def _filename_slug(source_name: str, file_id: str = "") -> str:
    stem = Path((source_name or "").strip()).stem
    if not stem:
        stem = (file_id or "").strip() or "document"
    return _ascii_slug(stem, max_len=40, prefix_if_digit="f")


def _file_id_suffix(file_id: str) -> str:
    fid = (file_id or "").strip()
    if not fid:
        return ""
    return _ascii_slug(fid, max_len=12, prefix_if_digit="f")


def rag_collection_name(source_name: str, file_id: str = "") -> str:
    """普通 RAG 集合：rag_{文件名slug}_{file_id}（全 ASCII）。"""
    fn = _filename_slug(source_name, file_id)
    fid = _file_id_suffix(file_id)
    if fid:
        return f"{RAG_COLLECTION_PREFIX}_{fn}_{fid}"
    return f"{RAG_COLLECTION_PREFIX}_{fn}"


def graphrag_collection_name(source_name: str, file_id: str = "") -> str:
    """GraphRAG 集合：graphrag_{文件名slug}_{file_id}（全 ASCII）。"""
    fn = _filename_slug(source_name, file_id)
    fid = _file_id_suffix(file_id)
    if fid:
        return f"{GRAPHRAG_COLLECTION_PREFIX}_{fn}_{fid}"
    return f"{GRAPHRAG_COLLECTION_PREFIX}_{fn}"


def is_rag_collection(name: str) -> bool:
    return (name or "").startswith(f"{RAG_COLLECTION_PREFIX}_")


def is_graphrag_collection(name: str) -> bool:
    return (name or "").startswith(f"{GRAPHRAG_COLLECTION_PREFIX}_")


def is_legacy_rag_collection(name: str) -> bool:
    return (name or "").startswith(f"{LEGACY_RAG_COLLECTION_PREFIX}_")


def is_legacy_graphrag_collection(name: str) -> bool:
    return (name or "").startswith(f"{LEGACY_GRAPHRAG_COLLECTION_PREFIX}_")


def _stored_or_compute(
    stored: str | None,
    compute_fn,
) -> str:
    name = (stored or "").strip()
    if name and is_valid_milvus_name(name):
        return name
    return compute_fn()


def resolve_rag_collection(
    session_id: str,
    file_id: str,
    source_name: str | None = None,
) -> str:
    """根据文件名与 file_id 解析 RAG 集合名；可回退读取上传元数据。"""
    fid = (file_id or "").strip()

    def _compute() -> str:
        if source_name:
            return rag_collection_name(source_name, fid)
        return rag_collection_name(fid or "document", fid)

    try:
        import chat_store

        meta = chat_store.get_upload_meta(session_id, fid) if fid else None
        if meta:
            stored = (meta.get("milvus_collection") or "").strip()
            if stored:
                return _stored_or_compute(
                    stored,
                    lambda: rag_collection_name(meta.get("name") or source_name or fid, fid),
                )
            name = (meta.get("name") or "").strip()
            if name:
                return rag_collection_name(name, fid)
    except Exception:
        pass
    return _compute()


def resolve_graphrag_collection(
    session_id: str,
    file_id: str,
    source_name: str | None = None,
) -> str:
    """根据文件名与 file_id 解析 GraphRAG 集合名。"""
    fid = (file_id or "").strip()

    def _compute() -> str:
        if source_name:
            return graphrag_collection_name(source_name, fid)
        return graphrag_collection_name(fid or "document", fid)

    try:
        import chat_store

        meta = chat_store.get_upload_meta(session_id, fid) if fid else None
        if meta:
            stored = (meta.get("milvus_collection") or "").strip()
            if stored:
                return _stored_or_compute(
                    stored,
                    lambda: graphrag_collection_name(meta.get("name") or source_name or fid, fid),
                )
            name = (meta.get("name") or "").strip()
            if name:
                return graphrag_collection_name(name, fid)
    except Exception:
        pass
    return _compute()
