"""GraphRAG：分块 → LLM 抽取实体/关系 → Neo4j → 向量入 Milvus → 混合检索。

流程概览
--------
索引（上传文档）:
  1. split_text 拆分正文
  2. 对每个块调用 LLM 抽取 entities + relations
  3. 合并去重后清空 Neo4j 单库并写入当前文档图谱（社区版仅保留最新一份）
  4. 为实体/关系文本生成 embedding，写入 Milvus graph_{file_id} 集合

检索（用户提问）:
  1. 问题 embedding → Milvus 向量检索（实体 + 关系）
  2. 命中实体名 → Neo4j 图遍历扩展邻域（1~N 跳）
  3. 拼装「向量命中 + 图结构」上下文注入大模型
"""
from __future__ import annotations

import json
import logging
import os
import re
import time
from typing import Any

import env_config  # noqa: F401

from langchain_core.messages import HumanMessage, SystemMessage
from pymilvus import MilvusClient

import neo4j_store
from api_throttle import call_with_retry
from llm_zhipu import make_chat_llm
from rag_milvus import embed_texts, split_text

logger = logging.getLogger("graphrag")

MILVUS_URI = os.getenv("MILVUS_URI", "http://127.0.0.1:19530")
MILVUS_DB_PREFIX = os.getenv("MILVUS_DB_PREFIX", "graphrag")
MILVUS_GRAPH_PREFIX = os.getenv("MILVUS_GRAPH_PREFIX", "graph")
EMBEDDING_DIM = int(os.getenv("RAG_EMBEDDING_DIM", "1024"))
GRAPHRAG_TOP_K = int(os.getenv("GRAPHRAG_TOP_K", "8"))
GRAPHRAG_GRAPH_HOPS = int(os.getenv("GRAPHRAG_GRAPH_HOPS", "2"))
GRAPHRAG_ENABLED = os.getenv(
    "GRAPHRAG_ENABLED", os.getenv("RAG_ENABLED", "1")
).strip().lower() not in ("0", "false", "no")
EXTRACT_MODEL = os.getenv("GRAPHRAG_EXTRACT_MODEL", "glm-4-flash")
EXTRACT_MAX_CHUNKS = int(os.getenv("GRAPHRAG_EXTRACT_MAX_CHUNKS", "16"))
EXTRACT_BATCH_SIZE = int(os.getenv("GRAPHRAG_EXTRACT_BATCH_SIZE", "3"))
EXTRACT_BATCH_DELAY_SEC = float(os.getenv("GRAPHRAG_EXTRACT_BATCH_DELAY_SEC", "0"))

_milvus_root: MilvusClient | None = None
_slug_re = re.compile(r"[^0-9a-zA-Z_]+")

EXTRACT_SYSTEM = """你是知识图谱抽取助手。从给定文本片段中抽取实体与关系，严格输出 JSON，不要 markdown 代码块：
{
  "entities": [
    {"name": "实体名称", "type": "Person|Organization|Location|Concept|Event|Product|Other", "description": "一句话描述"}
  ],
  "relations": [
    {"source": "源实体名", "target": "目标实体名", "relation": "RELATION_TYPE", "description": "关系说明"}
  ]
}
规则：只抽取文本中明确出现的信息；实体名简洁且与原文一致；relation 用英文大写下划线；无内容则返回空数组。"""


def graphrag_enabled() -> bool:
    return GRAPHRAG_ENABLED


def _slug(value: str, *, max_len: int = 56, prefix_if_digit: str = "d") -> str:
    s = _slug_re.sub("_", (value or "x").strip())[:max_len].strip("_") or "x"
    if s[0].isdigit():
        s = f"{prefix_if_digit}_{s}"
    return s


def _norm_session_id(session_id: str) -> str:
    return (session_id or "default").strip() or "default"


def session_database_name(session_id: str) -> str:
    sid = _slug(_norm_session_id(session_id), max_len=48, prefix_if_digit="s")
    return f"{MILVUS_DB_PREFIX}_{sid}"


def graph_collection_name(file_id: str) -> str:
    fid = _slug((file_id or "").strip(), max_len=56, prefix_if_digit="f")
    return f"{MILVUS_GRAPH_PREFIX}_{fid}"


def _root_client() -> MilvusClient:
    global _milvus_root
    if _milvus_root is None:
        _milvus_root = MilvusClient(uri=MILVUS_URI)
    return _milvus_root


def _ensure_session_database(db_name: str) -> None:
    root = _root_client()
    if db_name not in set(root.list_databases()):
        root.create_database(db_name)


def _client_for_database(db_name: str) -> MilvusClient:
    _ensure_session_database(db_name)
    return MilvusClient(uri=MILVUS_URI, db_name=db_name)


def _ensure_graph_collection(client: MilvusClient, collection_name: str) -> None:
    if client.has_collection(collection_name):
        client.drop_collection(collection_name)
    client.create_collection(
        collection_name=collection_name,
        dimension=EMBEDDING_DIM,
        metric_type="COSINE",
        auto_id=True,
        enable_dynamic_field=True,
    )


def list_graph_collections(session_id: str) -> list[str]:
    db_name = session_database_name(session_id)
    _ensure_session_database(db_name)
    client = _client_for_database(db_name)
    prefix = f"{MILVUS_GRAPH_PREFIX}_"
    return sorted(c for c in client.list_collections() if c.startswith(prefix))


def _parse_extract_json(raw: str) -> dict[str, list]:
    text = (raw or "").strip()
    if not text:
        return {"entities": [], "relations": []}
    m = re.search(r"\{[\s\S]*\}", text)
    if m:
        text = m.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {"entities": [], "relations": []}
    entities = data.get("entities") if isinstance(data.get("entities"), list) else []
    relations = data.get("relations") if isinstance(data.get("relations"), list) else []
    return {"entities": entities, "relations": relations}


def _extract_from_chunks(chunks: list[str]) -> dict[str, list]:
    """对一个或多个文本块抽取实体/关系（合并请求以减少 429）。"""
    if not chunks:
        return {"entities": [], "relations": []}

    llm = make_chat_llm(model=EXTRACT_MODEL, temperature=0.0)
    if len(chunks) == 1:
        body = f"文本片段：\n{chunks[0][:4000]}"
    else:
        parts = [f"【片段 {i + 1}】\n{c[:3000]}" for i, c in enumerate(chunks)]
        body = "以下多个片段来自同一文档，请合并抽取实体与关系：\n\n" + "\n\n".join(parts)

    def _invoke() -> dict[str, list]:
        resp = llm.invoke(
            [
                SystemMessage(content=EXTRACT_SYSTEM),
                HumanMessage(content=body),
            ]
        )
        content = resp.content if isinstance(resp.content, str) else str(resp.content)
        return _parse_extract_json(content)

    return call_with_retry(_invoke, label="graphrag-extract", throttle=False)


def _merge_extractions(chunks_results: list[dict[str, list]]) -> tuple[list[dict], list[dict]]:
    entity_map: dict[str, dict] = {}
    relations: list[dict] = []
    seen_rel: set[tuple[str, str, str]] = set()

    for block in chunks_results:
        for e in block.get("entities") or []:
            name = (e.get("name") or "").strip()
            if not name:
                continue
            key = name.lower()
            if key not in entity_map:
                entity_map[key] = {
                    "name": name,
                    "type": (e.get("type") or "Other").strip(),
                    "description": (e.get("description") or "").strip(),
                }
            elif e.get("description") and not entity_map[key].get("description"):
                entity_map[key]["description"] = e.get("description", "").strip()

        for r in block.get("relations") or []:
            src = (r.get("source") or "").strip()
            tgt = (r.get("target") or "").strip()
            rel = (r.get("relation") or "RELATED").strip()
            if not src or not tgt:
                continue
            sig = (src.lower(), tgt.lower(), rel.upper())
            if sig in seen_rel:
                continue
            seen_rel.add(sig)
            relations.append(
                {
                    "source": src,
                    "target": tgt,
                    "relation": rel,
                    "description": (r.get("description") or "").strip(),
                }
            )
    return list(entity_map.values()), relations


def _entity_embed_text(entity: dict) -> str:
    name = entity.get("name") or ""
    etype = entity.get("type") or ""
    desc = entity.get("description") or ""
    return f"实体：{name} | 类型：{etype} | {desc}".strip()


def _relation_embed_text(relation: dict) -> str:
    src = relation.get("source") or ""
    tgt = relation.get("target") or ""
    rel = relation.get("relation") or ""
    desc = relation.get("description") or ""
    return f"关系：{src} -[{rel}]-> {tgt} | {desc}".strip()


def _index_vectors_to_milvus(
    session_id: str,
    file_id: str,
    source_name: str,
    entities: list[dict],
    relations: list[dict],
) -> int:
    sid = _norm_session_id(session_id)
    fid = (file_id or "").strip()
    db_name = session_database_name(sid)
    coll_name = graph_collection_name(fid)
    client = _client_for_database(db_name)
    _ensure_graph_collection(client, coll_name)

    texts: list[str] = []
    rows_meta: list[dict[str, Any]] = []

    for e in entities:
        t = _entity_embed_text(e)
        texts.append(t)
        rows_meta.append(
            {
                "item_type": "entity",
                "text": t,
                "name": e.get("name") or "",
                "entity_type": e.get("type") or "",
                "source": source_name[:200],
                "file_id": fid,
                "session_id": sid,
            }
        )
    for r in relations:
        t = _relation_embed_text(r)
        texts.append(t)
        rows_meta.append(
            {
                "item_type": "relation",
                "text": t,
                "name": f"{r.get('source')}-{r.get('relation')}-{r.get('target')}",
                "relation": r.get("relation") or "",
                "rel_source": r.get("source") or "",
                "rel_target": r.get("target") or "",
                "source": source_name[:200],
                "file_id": fid,
                "session_id": sid,
            }
        )

    if not texts:
        return 0

    vectors = embed_texts(texts)
    rows = [{**meta, "vector": vec} for meta, vec in zip(rows_meta, vectors)]
    for i in range(0, len(rows), 100):
        client.insert(collection_name=coll_name, data=rows[i : i + 100])
    return len(rows)


def delete_file_graph(session_id: str, file_id: str) -> None:
    if not graphrag_enabled():
        return
    sid = _norm_session_id(session_id)
    fid = (file_id or "").strip()
    if not fid:
        return
    try:
        neo4j_store.delete_document_module(sid, fid)
    except Exception:
        pass
    db_name = session_database_name(sid)
    coll_name = graph_collection_name(fid)
    _ensure_session_database(db_name)
    client = _client_for_database(db_name)
    if client.has_collection(coll_name):
        client.drop_collection(coll_name)


def delete_session_graph(session_id: str) -> None:
    if not graphrag_enabled():
        return
    sid = _norm_session_id(session_id)
    try:
        neo4j_store.delete_session_graph(sid)
    except Exception:
        pass
    db_name = session_database_name(sid)
    root = _root_client()
    if db_name in set(root.list_databases()):
        root.drop_database(db_name)


def index_document(
    session_id: str,
    file_id: str,
    source_name: str,
    text: str,
) -> dict[str, Any]:
    """一篇文档 → Neo4j 模块 + Milvus 图向量集合。"""
    if not graphrag_enabled():
        return {"enabled": False, "entities": 0, "relations": 0}

    sid = _norm_session_id(session_id)
    fid = (file_id or "").strip()
    if not fid:
        return {"enabled": True, "entities": 0, "relations": 0, "error": "缺少 file_id"}

    chunks = split_text(text)
    if not chunks:
        delete_file_graph(sid, fid)
        return {
            "enabled": True,
            "entities": 0,
            "relations": 0,
            "chunks": 0,
            "database": session_database_name(sid),
            "collection": graph_collection_name(fid),
            "mode": "graphrag",
        }

    extract_chunks = chunks[:EXTRACT_MAX_CHUNKS]
    batch_size = max(1, EXTRACT_BATCH_SIZE)
    chunk_results: list[dict[str, list]] = []
    batch_ranges = list(range(0, len(extract_chunks), batch_size))
    for bi, start in enumerate(batch_ranges):
        batch = extract_chunks[start : start + batch_size]
        try:
            chunk_results.append(_extract_from_chunks(batch))
        except Exception as e:
            logger.warning("GraphRAG 抽取失败（batch %s）：%s", bi + 1, e)
            chunk_results.append({"entities": [], "relations": []})
        if bi + 1 < len(batch_ranges) and EXTRACT_BATCH_DELAY_SEC > 0:
            time.sleep(EXTRACT_BATCH_DELAY_SEC)

    entities, relations = _merge_extractions(chunk_results)

    graph_stats = {"entities": 0, "relations": 0}
    try:
        graph_stats = neo4j_store.store_document_graph(
            sid, fid, source_name, entities, relations
        )
    except Exception as e:
        return {
            "enabled": True,
            "entities": len(entities),
            "relations": len(relations),
            "chunks": len(chunks),
            "error": f"Neo4j 写入失败: {e}",
            "mode": "graphrag",
        }

    vector_count = 0
    try:
        vector_count = _index_vectors_to_milvus(
            sid, fid, source_name, entities, relations
        )
    except Exception as e:
        return {
            "enabled": True,
            **graph_stats,
            "chunks": len(chunks),
            "vectors": 0,
            "error": f"Milvus 写入失败: {e}",
            "database": session_database_name(sid),
            "collection": graph_collection_name(fid),
            "mode": "graphrag",
        }

    db_name = session_database_name(sid)
    coll_name = graph_collection_name(fid)
    neo4j_db = graph_stats.get("neo4j_database") or neo4j_store.NEO4J_DEFAULT_DB
    storage_mode = graph_stats.get("storage_mode") or "community_replace"
    return {
        "enabled": True,
        "entities": graph_stats.get("entities", len(entities)),
        "relations": graph_stats.get("relations", len(relations)),
        "vectors": vector_count,
        "chunks": len(chunks),
        "extracted_chunks": len(extract_chunks),
        "database": db_name,
        "collection": coll_name,
        "neo4j_database": neo4j_db,
        "storage_mode": storage_mode,
        "neo4j_active_file_id": graph_stats.get("active_file_id") or fid,
        "milvus_attu_hint": f"Attu → 数据库 `{db_name}` → 集合 `{coll_name}`",
        "neo4j_note": "社区版 Neo4j 仅保留最近一次上传的图谱；历史文档向量仍在 Milvus",
        "hierarchy": f"Neo4j({neo4j_db}/最新文档) + Milvus({db_name}/{coll_name})",
        "mode": "graphrag",
    }


def _search_milvus_graph(
    session_id: str,
    query_vec: list[float],
    limit: int,
    file_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    sid = _norm_session_id(session_id)
    db_name = session_database_name(sid)
    _ensure_session_database(db_name)
    client = _client_for_database(db_name)

    if file_ids:
        targets = [graph_collection_name(fid) for fid in file_ids if str(fid).strip()]
    else:
        targets = list_graph_collections(sid)

    hits: list[dict[str, Any]] = []
    per_coll = max(limit, 4)
    for coll in targets:
        if not client.has_collection(coll):
            continue
        raw = client.search(
            collection_name=coll,
            data=[query_vec],
            limit=per_coll,
            output_fields=[
                "text",
                "item_type",
                "name",
                "entity_type",
                "relation",
                "rel_source",
                "rel_target",
                "source",
                "file_id",
            ],
        )
        for item in raw[0] if raw else []:
            entity = item.get("entity") or {}
            hits.append(
                {
                    "text": entity.get("text") or "",
                    "item_type": entity.get("item_type") or "",
                    "name": entity.get("name") or "",
                    "entity_type": entity.get("entity_type") or "",
                    "relation": entity.get("relation") or "",
                    "rel_source": entity.get("rel_source") or "",
                    "rel_target": entity.get("rel_target") or "",
                    "source": entity.get("source") or "",
                    "file_id": entity.get("file_id") or "",
                    "collection": coll,
                    "score": float(item.get("distance", 0.0)),
                }
            )
    hits.sort(key=lambda x: x.get("score", 0.0), reverse=True)
    return hits[:limit]


def hybrid_search(
    session_id: str,
    query: str,
    top_k: int | None = None,
    file_ids: list[str] | None = None,
) -> dict[str, Any]:
    """混合检索：Milvus 向量 + Neo4j 图扩展。"""
    if not graphrag_enabled():
        return {"vector_hits": [], "graph_triples": [], "seed_entities": []}

    q = (query or "").strip()
    if not q:
        return {"vector_hits": [], "graph_triples": [], "seed_entities": []}

    sid = _norm_session_id(session_id)
    limit = top_k or GRAPHRAG_TOP_K
    query_vec = embed_texts([q])[0]
    vector_hits = _search_milvus_graph(sid, query_vec, limit, file_ids=file_ids)

    seed_names: list[str] = []
    for hit in vector_hits:
        if hit.get("item_type") == "entity" and hit.get("name"):
            seed_names.append(hit["name"])
        elif hit.get("item_type") == "relation":
            if hit.get("rel_source"):
                seed_names.append(hit["rel_source"])
            if hit.get("rel_target"):
                seed_names.append(hit["rel_target"])

    seed_names = list(dict.fromkeys(seed_names))[:10]

    graph_triples: list[dict] = []
    if seed_names:
        try:
            graph_triples = neo4j_store.expand_graph_from_entities(
                sid,
                seed_names,
                file_ids=file_ids,
                max_hops=GRAPHRAG_GRAPH_HOPS,
                limit=limit * 3,
            )
        except Exception:
            graph_triples = []

    return {
        "vector_hits": vector_hits,
        "graph_triples": graph_triples,
        "seed_entities": seed_names,
    }


def build_graphrag_context(
    session_id: str,
    query: str,
    top_k: int | None = None,
    file_ids: list[str] | None = None,
) -> str:
    try:
        result = hybrid_search(session_id, query, top_k=top_k, file_ids=file_ids)
    except Exception as e:
        return f"（GraphRAG 检索暂不可用：{e}）"

    vector_hits = result.get("vector_hits") or []
    graph_triples = result.get("graph_triples") or []
    if not vector_hits and not graph_triples:
        return ""

    db_name = session_database_name(session_id)
    parts: list[str] = [
        f"【GraphRAG 混合检索结果】Milvus 库：{db_name}；Neo4j 按文档独立库检索",
        "以下为向量相似度命中的实体/关系，以及对应文档 Neo4j 库内的图扩展三元组，请优先依据作答：",
    ]

    if vector_hits:
        parts.append("\n--- 向量检索命中 ---")
        for i, hit in enumerate(vector_hits, 1):
            body = (hit.get("text") or "").strip()
            if not body:
                continue
            score = hit.get("score", 0.0)
            itype = hit.get("item_type") or "item"
            src = hit.get("source") or hit.get("file_id") or ""
            parts.append(f"\n[{i}] ({itype}) {src} 相关度 {score:.3f}\n{body}")

    if graph_triples:
        parts.append("\n--- 知识图谱扩展 ---")
        for i, t in enumerate(graph_triples, 1):
            src = t.get("source") or "?"
            rel = t.get("relation") or "RELATED"
            tgt = t.get("target") or "?"
            rel_desc = (t.get("rel_desc") or "").strip()
            neo4j_db = t.get("neo4j_database") or ""
            line = f"{src} -[{rel}]-> {tgt}"
            if rel_desc:
                line += f"（{rel_desc}）"
            if neo4j_db:
                line += f" @ {neo4j_db}"
            parts.append(f"\n[图 {i}] {line}")

    return "\n".join(parts) if len(parts) > 2 else ""


def list_session_graph_index(session_id: str) -> list[dict[str, Any]]:
    sid = _norm_session_id(session_id)
    out: list[dict[str, Any]] = []
    db_name = session_database_name(sid)

    neo_modules: dict[str, dict] = {}
    try:
        for m in neo4j_store.list_session_modules(sid):
            fid = m.get("file_id") or ""
            neo_modules[fid] = m
    except Exception:
        pass

    if db_name not in set(_root_client().list_databases()):
        return [
            {
                "database": db_name,
                "collection": "",
                "file_id": fid,
                "source": mod.get("name") or fid,
                "entity_count": mod.get("entity_count", 0),
                "mode": "graphrag",
            }
            for fid, mod in neo_modules.items()
        ]

    client = _client_for_database(db_name)
    for coll in list_graph_collections(sid):
        fid = coll[len(f"{MILVUS_GRAPH_PREFIX}_") :]
        mod = neo_modules.get(fid, {})
        try:
            n = client.query(
                collection_name=coll,
                filter='item_type != ""',
                output_fields=["source"],
                limit=1,
            )
            source = (n[0].get("source") if n else "") or mod.get("name") or coll
        except Exception:
            source = mod.get("name") or coll
        out.append(
            {
                "database": db_name,
                "collection": coll,
                "file_id": fid,
                "source": source,
                "entity_count": mod.get("entity_count", 0),
                "neo4j_database": mod.get("neo4j_database") or neo4j_store.document_database_name(sid, fid),
                "mode": "graphrag",
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
        return {"enabled": True, "entities": 0, "relations": 0, "error": str(e), "mode": "graphrag"}


def safe_delete_session_graph(session_id: str) -> None:
    try:
        delete_session_graph(session_id)
    except Exception:
        pass
