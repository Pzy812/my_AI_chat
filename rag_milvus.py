"""RAG：分块 → 智谱 Embedding → Milvus 分层存储。

层级结构（在 Attu 中可见）：
  Database  ``rag_{session_id}``   — 一个会话一层
  Collection ``doc_{file_id}``     — 一篇文档一个集合（仅该文档的向量块）

旧版单集合 ``ai_chat_rag`` 不再写入，可在 Attu 中手动删除。
"""
from __future__ import annotations

import os
import re
import time
from typing import Any

import env_config  # noqa: F401 — 加载 .env
from env_config import get_zhipuai_api_key

from api_throttle import call_with_retry
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pymilvus import MilvusClient
from zai import ZhipuAiClient as ZhipuAI # ✅ 新版官方标准

MILVUS_URI = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
MILVUS_DB_PREFIX = os.getenv("MILVUS_DB_PREFIX", "rag")
MILVUS_DOC_PREFIX = os.getenv("MILVUS_DOC_PREFIX", "doc")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "embedding-3")
EMBEDDING_DIM = int(os.getenv("RAG_EMBEDDING_DIM", "1024"))
CHUNK_SIZE = int(os.getenv("RAG_CHUNK_SIZE", "600"))
CHUNK_OVERLAP = int(os.getenv("RAG_CHUNK_OVERLAP", "100"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
EMBED_BATCH_SIZE = int(os.getenv("RAG_EMBED_BATCH_SIZE", "8"))
EMBED_BATCH_DELAY_SEC = float(os.getenv("RAG_EMBED_BATCH_DELAY_SEC", "0"))
RAG_ENABLED = os.getenv("RAG_ENABLED", "1").strip().lower() not in ("0", "false", "no")

_milvus_root: MilvusClient | None = None
_zhipu: ZhipuAI | None = None
_slug_re = re.compile(r"[^0-9a-zA-Z_]+")


def rag_enabled() -> bool:
    return RAG_ENABLED


def _slug(value: str, *, max_len: int = 56, prefix_if_digit: str = "d") -> str:
    s = _slug_re.sub("_", (value or "x").strip())[:max_len].strip("_") or "x"
    if s[0].isdigit():
        s = f"{prefix_if_digit}_{s}"
    return s


def _norm_session_id(session_id: str) -> str:
    return (session_id or "default").strip() or "default"


def session_database_name(session_id: str) -> str:
    """会话级 Milvus Database，例如 rag_default。"""
    sid = _slug(_norm_session_id(session_id), max_len=48, prefix_if_digit="s")
    return f"{MILVUS_DB_PREFIX}_{sid}"


def document_collection_name(file_id: str) -> str:
    """文档级 Collection，例如 doc_8a0f679ce441。"""
    fid = _slug((file_id or "").strip(), max_len=56, prefix_if_digit="f")
    return f"{MILVUS_DOC_PREFIX}_{fid}"


def _zhipu_client() -> ZhipuAI:
    global _zhipu
    if _zhipu is None:
        _zhipu = ZhipuAI(api_key=get_zhipuai_api_key())
    return _zhipu


def _root_client() -> MilvusClient:
    global _milvus_root
    if _milvus_root is None:
        _milvus_root = MilvusClient(uri=MILVUS_URI)
    return _milvus_root


def _ensure_session_database(db_name: str) -> None:
    root = _root_client()
    existing = set(root.list_databases())
    if db_name not in existing:
        root.create_database(db_name)


def _client_for_database(db_name: str) -> MilvusClient:
    _ensure_session_database(db_name)
    return MilvusClient(uri=MILVUS_URI, db_name=db_name)


def _ensure_doc_collection(client: MilvusClient, collection_name: str) -> None:
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        dimension=EMBEDDING_DIM,
        metric_type="COSINE",
        auto_id=True,
        enable_dynamic_field=True,
    )


def list_document_collections(session_id: str) -> list[str]:
    """列出某会话下所有文档集合名。"""
    db_name = session_database_name(session_id)
    _ensure_session_database(db_name)
    client = _client_for_database(db_name)
    prefix = f"{MILVUS_DOC_PREFIX}_"
    return sorted(c for c in client.list_collections() if c.startswith(prefix))


def split_text(text: str) -> list[str]:
    body = (text or "").strip()
    if not body:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", ".", " ", ""],
    )
    return [c.strip() for c in splitter.split_text(body) if c.strip()]


def embed_texts(texts: list[str]) -> list[list[float]]:
    if not texts:
        return []
    client = _zhipu_client()
    vectors: list[list[float]] = []
    batches = list(range(0, len(texts), EMBED_BATCH_SIZE))
    for bi, i in enumerate(batches):
        batch = texts[i : i + EMBED_BATCH_SIZE]

        def _embed_batch() -> list[list[float]]:
            resp = client.embeddings.create(
                model=EMBEDDING_MODEL,
                input=batch,
                dimensions=EMBEDDING_DIM,
            )
            return [item.embedding for item in resp.data]

        vectors.extend(call_with_retry(_embed_batch, label="zhipu-embed"))
        if bi + 1 < len(batches) and EMBED_BATCH_DELAY_SEC > 0:
            time.sleep(EMBED_BATCH_DELAY_SEC)
    return vectors


def delete_file_chunks(session_id: str, file_id: str) -> None:
    if not rag_enabled():
        return
    fid = (file_id or "").strip()
    if not fid:
        return
    db_name = session_database_name(session_id)
    coll_name = document_collection_name(fid)
    _ensure_session_database(db_name)
    client = _client_for_database(db_name)
    if client.has_collection(coll_name):
        client.drop_collection(coll_name)


def delete_session_chunks(session_id: str) -> None:
    if not rag_enabled():
        return
    db_name = session_database_name(session_id)
    root = _root_client()
    if db_name in set(root.list_databases()):
        root.drop_database(db_name)


def index_document(
    session_id: str,
    file_id: str,
    source_name: str,
    text: str,
) -> dict[str, Any]:
    """一篇文档 → 独立 Milvus Collection（位于会话 Database 下）。"""
    if not rag_enabled():
        return {"enabled": False, "chunks": 0}
    sid = _norm_session_id(session_id)
    fid = (file_id or "").strip()
    if not fid:
        return {"enabled": True, "chunks": 0, "error": "缺少 file_id"}

    chunks = split_text(text)
    if not chunks:
        delete_file_chunks(sid, fid)
        return {
            "enabled": True,
            "chunks": 0,
            "database": session_database_name(sid),
            "collection": document_collection_name(fid),
        }

    db_name = session_database_name(sid)
    coll_name = document_collection_name(fid)
    client = _client_for_database(db_name)
    _ensure_doc_collection(client, coll_name)

    vectors = embed_texts(chunks)
    source = (source_name or fid)[:200]
    rows = [
        {
            "vector": vec,
            "text": chunk,
            "chunk_index": idx,
            "source": source,
            "file_id": fid,
            "session_id": sid,
        }
        for idx, (chunk, vec) in enumerate(zip(chunks, vectors))
    ]
    for i in range(0, len(rows), 100):
        client.insert(collection_name=coll_name, data=rows[i : i + 100])

    return {
        "enabled": True,
        "chunks": len(chunks),
        "database": db_name,
        "collection": coll_name,
        "hierarchy": f"{db_name} / {coll_name}",
    }


def _search_one_collection(
    client: MilvusClient,
    collection_name: str,
    query_vec: list[float],
    limit: int,
    *,
    source_name: str = "",
    file_id: str = "",
) -> list[dict[str, Any]]:
    if not client.has_collection(collection_name):
        return []
    raw = client.search(
        collection_name=collection_name,
        data=[query_vec],
        limit=limit,
        output_fields=["text", "source", "file_id", "chunk_index"],
    )
    hits: list[dict[str, Any]] = []
    for item in raw[0] if raw else []:
        entity = item.get("entity") or {}
        hits.append(
            {
                "text": entity.get("text") or "",
                "source": entity.get("source") or source_name,
                "file_id": entity.get("file_id") or file_id,
                "chunk_index": entity.get("chunk_index"),
                "collection": collection_name,
                "score": float(item.get("distance", 0.0)),
            }
        )
    return hits


def _file_id_from_collection(coll_name: str) -> str:
    prefix = f"{MILVUS_DOC_PREFIX}_"
    if coll_name.startswith(prefix):
        return coll_name[len(prefix) :]
    return coll_name


def search_similar(
    session_id: str,
    query: str,
    top_k: int | None = None,
    file_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    if not rag_enabled():
        return []
    q = (query or "").strip()
    if not q:
        return []
    sid = _norm_session_id(session_id)
    limit = top_k or RAG_TOP_K
    query_vec = embed_texts([q])[0]

    db_name = session_database_name(sid)
    _ensure_session_database(db_name)
    client = _client_for_database(db_name)

    if file_ids:
        targets = [document_collection_name(fid) for fid in file_ids if str(fid).strip()]
    else:
        targets = list_document_collections(sid)

    if not targets:
        return []

    per_coll = max(limit, 3)
    merged: list[dict[str, Any]] = []
    for coll in targets:
        merged.extend(
            _search_one_collection(
                client,
                coll,
                query_vec,
                per_coll,
                file_id=_file_id_from_collection(coll),
            )
        )
    merged.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return merged[:limit]


def build_rag_context(
    session_id: str,
    query: str,
    top_k: int | None = None,
    file_ids: list[str] | None = None,
) -> str:
    try:
        hits = search_similar(session_id, query, top_k=top_k, file_ids=file_ids)
    except Exception as e:
        return f"（知识库检索暂不可用：{e}）"
    if not hits:
        return ""
    db_name = session_database_name(session_id)
    parts: list[str] = [
        f"【知识库检索结果】库：{db_name}；以下为向量相似度检索到的文档片段，请优先依据作答："
    ]
    for i, hit in enumerate(hits, 1):
        src = hit.get("source") or hit.get("file_id") or "未知来源"
        coll = hit.get("collection") or ""
        score = hit.get("score", 0.0)
        body = (hit.get("text") or "").strip()
        if not body:
            continue
        loc = f"{coll}" if coll else src
        parts.append(f"\n[片段 {i}] {src} @ {loc}（相关度 {score:.3f}）\n{body}")
    return "\n".join(parts) if len(parts) > 1 else ""


def list_session_rag_index(session_id: str) -> list[dict[str, Any]]:
    """列出会话下已索引的文档（供调试或 API）。"""
    sid = _norm_session_id(session_id)
    db_name = session_database_name(sid)
    if db_name not in set(_root_client().list_databases()):
        return []
    client = _client_for_database(db_name)
    out: list[dict[str, Any]] = []
    for coll in list_document_collections(sid):
        try:
            n = client.query(collection_name=coll, filter="chunk_index >= 0", output_fields=["source"], limit=1)
            source = (n[0].get("source") if n else "") or coll
        except Exception:
            source = coll
        out.append({"database": db_name, "collection": coll, "source": source})
    return out


def safe_index_document(
    session_id: str,
    file_id: str,
    source_name: str,
    text: str,
) -> dict[str, Any]:
    try:
        return index_document(session_id, file_id, source_name, text)
    except Exception as e:
        return {"enabled": True, "chunks": 0, "error": str(e)}


def safe_delete_session_chunks(session_id: str) -> None:
    try:
        delete_session_chunks(session_id)
    except Exception:
        pass
