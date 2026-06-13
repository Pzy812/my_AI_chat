"""RAG：Parent-Child 分块 → 智谱 Embedding → Milvus + BM25 Hybrid → Rerank。

层级结构（在 Attu 中可见）：
  Database   ``{session_id}``              — 一会话一库
  Collection ``rag_{文件名}_{file_id}``   — 普通 RAG 文档向量（child 向量 + parent 文本）

检索流程：
  1. 向量检索 child 块 + BM25 检索 child 块
  2. RRF 融合并按 parent 去重
  3. 智谱 Rerank 对 parent 文本精排
  4. 返回 top_k 个 parent 级上下文

GraphRAG 向量集合 ``graphrag_{文件名}_{file_id}`` 与 RAG 共用同一会话库（见 graphrag.py）。
"""
from __future__ import annotations

import os
import time
from typing import Any

import config.env_config  # noqa: F401 — 加载 .env
from config.env_config import get_zhipuai_api_key

from core.api_throttle import call_with_retry
from langchain_text_splitters import RecursiveCharacterTextSplitter
from rag.milvus_naming import (
    LEGACY_RAG_COLLECTION_PREFIX,
    RAG_COLLECTION_PREFIX,
    is_legacy_rag_collection,
    is_rag_collection,
    norm_session_id,
    rag_collection_name,
    resolve_rag_collection,
    session_database_name,
)
from pymilvus import MilvusClient
from zai import ZhipuAiClient as ZhipuAI  # ✅ 新版官方标准

import rag.rag_bm25 as rag_bm25
from rag.rag_hybrid import reciprocal_rank_fusion, rerank_hits

MILVUS_URI = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
EMBEDDING_MODEL = os.getenv("RAG_EMBEDDING_MODEL", "embedding-3")
EMBEDDING_DIM = int(os.getenv("RAG_EMBEDDING_DIM", "1024"))
CHILD_CHUNK_SIZE = int(os.getenv("RAG_CHILD_CHUNK_SIZE", os.getenv("RAG_CHUNK_SIZE", "600")))
CHILD_CHUNK_OVERLAP = int(
    os.getenv("RAG_CHILD_CHUNK_OVERLAP", os.getenv("RAG_CHUNK_OVERLAP", "100"))
)
PARENT_CHUNK_SIZE = int(os.getenv("RAG_PARENT_CHUNK_SIZE", "1800"))
RAG_TOP_K = int(os.getenv("RAG_TOP_K", "5"))
RAG_FETCH_K = int(os.getenv("RAG_FETCH_K", "30"))
RAG_HYBRID_ENABLED = os.getenv("RAG_HYBRID_ENABLED", "1").strip().lower() not in (
    "0",
    "false",
    "no",
)
EMBED_BATCH_SIZE = int(os.getenv("RAG_EMBED_BATCH_SIZE", "8"))
EMBED_BATCH_DELAY_SEC = float(os.getenv("RAG_EMBED_BATCH_DELAY_SEC", "0"))
RAG_ENABLED = os.getenv("RAG_ENABLED", "1").strip().lower() not in ("0", "false", "no")

# 兼容 graphrag 等旧调用方
CHUNK_SIZE = CHILD_CHUNK_SIZE
CHUNK_OVERLAP = CHILD_CHUNK_OVERLAP

_milvus_root: MilvusClient | None = None
_zhipu: ZhipuAI | None = None

_PARENT_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", " ", ""]
_CHILD_SEPARATORS = ["\n\n", "\n", "。", "！", "？", ".", " ", ""]


def rag_enabled() -> bool:
    return RAG_ENABLED


def hybrid_enabled() -> bool:
    return RAG_HYBRID_ENABLED


def document_collection_name(file_id: str, source_name: str = "") -> str:
    """文档级 Collection（兼容旧调用方，优先使用文件名）。"""
    return rag_collection_name(source_name or file_id, file_id)


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
    """列出某会话下所有 RAG 文档集合名（含旧版 doc_*）。"""
    db_name = session_database_name(session_id)
    _ensure_session_database(db_name)
    client = _client_for_database(db_name)
    legacy_prefix = f"{LEGACY_RAG_COLLECTION_PREFIX}_"
    return sorted(
        c
        for c in client.list_collections()
        if is_rag_collection(c) or c.startswith(legacy_prefix)
    )


def split_text(text: str) -> list[str]:
    """扁平分块（GraphRAG 等模块仍使用）。"""
    body = (text or "").strip()
    if not body:
        return []
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=_CHILD_SEPARATORS,
    )
    return [c.strip() for c in splitter.split_text(body) if c.strip()]


def split_text_parent_child(text: str) -> list[dict[str, Any]]:
    """Parent-Child 分块：parent 供 LLM 上下文，child 供向量/BM25 检索。"""
    body = (text or "").strip()
    if not body:
        return []

    parent_splitter = RecursiveCharacterTextSplitter(
        chunk_size=PARENT_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=_PARENT_SEPARATORS,
    )
    child_splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHILD_CHUNK_SIZE,
        chunk_overlap=CHILD_CHUNK_OVERLAP,
        separators=_CHILD_SEPARATORS,
    )

    records: list[dict[str, Any]] = []
    parents = [p.strip() for p in parent_splitter.split_text(body) if p.strip()]
    for parent_index, parent in enumerate(parents):
        children = [c.strip() for c in child_splitter.split_text(parent) if c.strip()]
        if not children:
            children = [parent]
        for child_index, child in enumerate(children):
            records.append(
                {
                    "child_text": child,
                    "parent_text": parent,
                    "parent_index": parent_index,
                    "child_index": child_index,
                }
            )
    return records


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


def delete_file_chunks(
    session_id: str,
    file_id: str,
    source_name: str | None = None,
) -> None:
    if not rag_enabled():
        return
    fid = (file_id or "").strip()
    if not fid:
        return
    db_name = session_database_name(session_id)
    coll_name = resolve_rag_collection(session_id, fid, source_name)
    _ensure_session_database(db_name)
    client = _client_for_database(db_name)
    if client.has_collection(coll_name):
        client.drop_collection(coll_name)
    legacy = f"{LEGACY_RAG_COLLECTION_PREFIX}_{fid}"
    if client.has_collection(legacy):
        client.drop_collection(legacy)
    rag_bm25.delete_bm25_index(session_id, fid)


def _maybe_drop_empty_session_database(session_id: str) -> None:
    db_name = session_database_name(session_id)
    root = _root_client()
    if db_name not in set(root.list_databases()):
        return
    client = _client_for_database(db_name)
    if not client.list_collections():
        root.drop_database(db_name)


def delete_session_chunks(session_id: str) -> None:
    """删除会话下所有 RAG 集合（不删除 graphrag_* 集合）。"""
    if not rag_enabled():
        return
    db_name = session_database_name(session_id)
    root = _root_client()
    if db_name not in set(root.list_databases()):
        rag_bm25.delete_session_bm25_indexes(session_id)
        return
    client = _client_for_database(db_name)
    legacy_prefix = f"{LEGACY_RAG_COLLECTION_PREFIX}_"
    for coll in list(client.list_collections()):
        if is_rag_collection(coll) or coll.startswith(legacy_prefix):
            client.drop_collection(coll)
    rag_bm25.delete_session_bm25_indexes(session_id)
    _maybe_drop_empty_session_database(session_id)


def index_document(
    session_id: str,
    file_id: str,
    source_name: str,
    text: str,
) -> dict[str, Any]:
    """一篇文档 → 独立 Milvus Collection（child 向量 + parent 文本）+ BM25 侧索引。"""
    if not rag_enabled():
        return {"enabled": False, "chunks": 0}
    sid = norm_session_id(session_id)
    fid = (file_id or "").strip()
    if not fid:
        return {"enabled": True, "chunks": 0, "error": "缺少 file_id"}

    records = split_text_parent_child(text)
    coll_name = rag_collection_name(source_name, fid)
    if not records:
        delete_file_chunks(sid, fid, source_name)
        return {
            "enabled": True,
            "chunks": 0,
            "parents": 0,
            "database": session_database_name(sid),
            "collection": coll_name,
        }

    db_name = session_database_name(sid)
    client = _client_for_database(db_name)
    _ensure_doc_collection(client, coll_name)

    child_texts = [r["child_text"] for r in records]
    vectors = embed_texts(child_texts)
    source = (source_name or fid)[:200]
    rows = [
        {
            "vector": vec,
            "text": rec["child_text"],
            "parent_text": rec["parent_text"],
            "parent_index": rec["parent_index"],
            "child_index": rec["child_index"],
            "chunk_index": rec["child_index"],
            "source": source,
            "file_id": fid,
            "session_id": sid,
        }
        for idx, (rec, vec) in enumerate(zip(records, vectors))
    ]
    for i in range(0, len(rows), 100):
        client.insert(collection_name=coll_name, data=rows[i : i + 100])

    rag_bm25.save_bm25_index(sid, fid, records)
    parent_count = len({r["parent_index"] for r in records})

    return {
        "enabled": True,
        "chunks": len(records),
        "parents": parent_count,
        "database": db_name,
        "collection": coll_name,
        "hierarchy": f"{db_name} / {coll_name}",
        "retrieval": "hybrid_parent_child",
    }


def _search_one_collection_vector(
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
        output_fields=[
            "text",
            "parent_text",
            "parent_index",
            "child_index",
            "source",
            "file_id",
            "chunk_index",
        ],
    )
    hits: list[dict[str, Any]] = []
    for item in raw[0] if raw else []:
        entity = item.get("entity") or {}
        hits.append(
            {
                "text": entity.get("text") or "",
                "parent_text": entity.get("parent_text") or "",
                "parent_index": entity.get("parent_index"),
                "child_index": entity.get("child_index"),
                "chunk_index": entity.get("chunk_index"),
                "source": entity.get("source") or source_name,
                "file_id": entity.get("file_id") or file_id,
                "collection": collection_name,
                "score": float(item.get("distance", 0.0)),
                "retriever": "vector",
            }
        )
    return hits


def _file_id_from_collection(coll_name: str) -> str:
    if is_legacy_rag_collection(coll_name):
        return coll_name[len(f"{LEGACY_RAG_COLLECTION_PREFIX}_") :]
    if is_rag_collection(coll_name):
        parts = coll_name.split("_")
        if len(parts) >= 2:
            return parts[-1]
    return coll_name


def _search_one_collection_hybrid(
    session_id: str,
    client: MilvusClient,
    collection_name: str,
    query: str,
    query_vec: list[float],
    limit: int,
    *,
    source_name: str = "",
    file_id: str = "",
) -> list[dict[str, Any]]:
    fid = file_id or _file_id_from_collection(collection_name)
    vector_hits = _search_one_collection_vector(
        client,
        collection_name,
        query_vec,
        limit,
        source_name=source_name,
        file_id=fid,
    )
    if not hybrid_enabled():
        return vector_hits

    bm25_hits = rag_bm25.search_bm25(
        session_id,
        fid,
        query,
        limit=limit,
        source_name=source_name,
        collection=collection_name,
    )
    if not bm25_hits:
        return vector_hits
    if not vector_hits:
        return bm25_hits

    return reciprocal_rank_fusion(
        [vector_hits, bm25_hits],
        limit=limit,
    )


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
    sid = norm_session_id(session_id)
    final_k = top_k or RAG_TOP_K
    fetch_k = max(RAG_FETCH_K, final_k * 3)
    query_vec = embed_texts([q])[0]

    db_name = session_database_name(sid)
    _ensure_session_database(db_name)
    client = _client_for_database(db_name)

    if file_ids:
        targets = [
            resolve_rag_collection(sid, str(fid).strip())
            for fid in file_ids
            if str(fid).strip()
        ]
    else:
        targets = list_document_collections(sid)

    if not targets:
        return []

    per_coll = max(fetch_k, 3)
    ranked_lists: list[list[dict[str, Any]]] = []
    for coll in targets:
        hits = _search_one_collection_hybrid(
            sid,
            client,
            coll,
            q,
            query_vec,
            per_coll,
            file_id=_file_id_from_collection(coll),
        )
        if hits:
            ranked_lists.append(hits)

    if not ranked_lists:
        return []

    if len(ranked_lists) == 1:
        merged = ranked_lists[0][:fetch_k]
    else:
        merged = reciprocal_rank_fusion(ranked_lists, limit=fetch_k)

    return rerank_hits(q, merged, top_k=final_k)


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
    mode_hint = "Hybrid(BM25+向量)+Rerank+Parent-Child" if hybrid_enabled() else "向量"
    parts: list[str] = [
        f"【知识库检索结果】库：{db_name}；模式：{mode_hint}；以下为检索到的文档片段，请优先依据作答："
    ]
    for i, hit in enumerate(hits, 1):
        src = hit.get("source") or hit.get("file_id") or "未知来源"
        coll = hit.get("collection") or ""
        score = hit.get("score", 0.0)
        body = (hit.get("parent_text") or hit.get("text") or "").strip()
        if not body:
            continue
        loc = f"{coll}" if coll else src
        score_label = "精排" if hit.get("rerank_score") is not None else "相关度"
        parts.append(f"\n[片段 {i}] {src} @ {loc}（{score_label} {score:.3f}）\n{body}")
    return "\n".join(parts) if len(parts) > 1 else ""


def list_session_rag_index(session_id: str) -> list[dict[str, Any]]:
    """列出会话下已索引的文档（供调试或 API）。"""
    sid = norm_session_id(session_id)
    db_name = session_database_name(sid)
    if db_name not in set(_root_client().list_databases()):
        return []
    client = _client_for_database(db_name)
    out: list[dict[str, Any]] = []
    for coll in list_document_collections(sid):
        try:
            n = client.query(
                collection_name=coll,
                filter="chunk_index >= 0",
                output_fields=["source"],
                limit=1,
            )
            source = (n[0].get("source") if n else "") or coll
        except Exception:
            source = coll
        fid = _file_id_from_collection(coll)
        out.append(
            {
                "database": db_name,
                "collection": coll,
                "source": source,
                "bm25_index": rag_bm25.bm25_index_exists(sid, fid),
                "retrieval": "hybrid_parent_child",
            }
        )
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
