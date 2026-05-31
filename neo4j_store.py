"""Neo4j 社区版单库模式：每次上传 GraphRAG 文档时清空 neo4j 库并重建知识图谱。

Neo4j Community 仅支持一个用户库（neo4j），因此：
  - 新文档上传 → MATCH (n) DETACH DELETE n → 写入当前文档实体/关系
  - Neo4j 中始终只保留「最近一次上传」的图谱
  - 历史文档检索仍由 Milvus 向量库承担（按 file_id 分 collection）
"""
from __future__ import annotations

import logging
import os
from typing import Any

import env_config  # noqa: F401
from neo4j import GraphDatabase

logger = logging.getLogger("neo4j_store")

NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "12345678")
NEO4J_DEFAULT_DB = os.getenv("NEO4J_DEFAULT_DB", "neo4j")

_driver = None


def get_driver():
    global _driver
    if _driver is None:
        _driver = GraphDatabase.driver(
            NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
    return _driver


def close_driver() -> None:
    global _driver
    if _driver is not None:
        _driver.close()
        _driver = None


def _norm_session_id(session_id: str) -> str:
    return (session_id or "default").strip() or "default"


def _session():
    return get_driver().session(database=NEO4J_DEFAULT_DB)


def document_database_name(session_id: str, file_id: str) -> str:
    """社区版固定返回 neo4j（逻辑标识，便于 API 与前端展示）。"""
    return NEO4J_DEFAULT_DB


def clear_all_graph_data() -> None:
    """清空 neo4j 库内全部节点与关系（含旧版 DocumentModule / 演示数据等）。"""
    with _session() as session:
        session.run("MATCH (n) DETACH DELETE n")
    logger.info("已清空 Neo4j 库 %s 内全部图数据", NEO4J_DEFAULT_DB)


def storage_info(session_id: str, file_id: str) -> dict[str, Any]:
    sid = _norm_session_id(session_id)
    fid = (file_id or "").strip()
    return {
        "storage_mode": "community_replace",
        "neo4j_database": NEO4J_DEFAULT_DB,
        "neo4j_browser_hint": (
            f"Browser 选择库 `{NEO4J_DEFAULT_DB}`，执行 "
            f"MATCH (d:Document {{file_id:'{fid}', session_id:'{sid}'}})-[:CONTAINS]->(e) RETURN d,e"
        ),
        "note": "社区版单库：Neo4j 仅保留最近一次上传的图谱；历史文档见 Milvus",
    }


def delete_document_module(session_id: str, file_id: str) -> None:
    clear_all_graph_data()


def delete_session_graph(session_id: str) -> None:
    clear_all_graph_data()


def store_document_graph(
    session_id: str,
    file_id: str,
    source_name: str,
    entities: list[dict[str, Any]],
    relations: list[dict[str, Any]],
) -> dict[str, Any]:
    """清空 neo4j 后写入当前文档的知识图谱。"""
    sid = _norm_session_id(session_id)
    fid = (file_id or "").strip()
    if not fid:
        return {
            "entities": 0,
            "relations": 0,
            "neo4j_database": NEO4J_DEFAULT_DB,
            "storage_mode": "community_replace",
        }

    ent_rows = []
    for e in entities:
        name = (e.get("name") or "").strip()
        if not name:
            continue
        ent_rows.append(
            {
                "name": name[:200],
                "type": (e.get("type") or "Unknown")[:80],
                "description": (e.get("description") or "")[:500],
            }
        )

    rel_rows = []
    for r in relations:
        src = (r.get("source") or "").strip()
        tgt = (r.get("target") or "").strip()
        if not src or not tgt:
            continue
        rel_rows.append(
            {
                "source": src[:200],
                "target": tgt[:200],
                "relation": (r.get("relation") or "RELATED")[:80],
                "description": (r.get("description") or "")[:500],
            }
        )

    clear_all_graph_data()

    with _session() as session:
        session.run(
            """
            CREATE (d:Document {
                file_id: $fid,
                session_id: $sid,
                name: $name,
                updated_at: datetime()
            })
            """,
            sid=sid,
            fid=fid,
            name=(source_name or fid)[:200],
        )
        if ent_rows:
            session.run(
                """
                MATCH (d:Document {file_id: $fid, session_id: $sid})
                UNWIND $rows AS row
                CREATE (e:Entity {
                    name: row.name,
                    type: row.type,
                    description: row.description,
                    file_id: $fid,
                    session_id: $sid
                })
                CREATE (d)-[:CONTAINS]->(e)
                """,
                sid=sid,
                fid=fid,
                rows=ent_rows,
            )
        if rel_rows:
            session.run(
                """
                UNWIND $rows AS row
                MATCH (a:Entity {name: row.source, file_id: $fid, session_id: $sid})
                MATCH (b:Entity {name: row.target, file_id: $fid, session_id: $sid})
                CREATE (a)-[:RELATED {
                    relation: row.relation,
                    description: row.description
                }]->(b)
                """,
                sid=sid,
                fid=fid,
                rows=rel_rows,
            )

    logger.info(
        "Neo4j 已重建图谱 session=%s file=%s entities=%s relations=%s",
        sid,
        fid,
        len(ent_rows),
        len(rel_rows),
    )
    return {
        "entities": len(ent_rows),
        "relations": len(rel_rows),
        "neo4j_database": NEO4J_DEFAULT_DB,
        "storage_mode": "community_replace",
        "active_file_id": fid,
        "active_session_id": sid,
    }


def get_active_document() -> dict[str, Any] | None:
    """返回 Neo4j 中当前唯一图谱对应的文档元信息。"""
    with _session() as session:
        row = session.run(
            """
            MATCH (d:Document)
            OPTIONAL MATCH (d)-[:CONTAINS]->(e:Entity)
            RETURN d.file_id AS file_id, d.session_id AS session_id, d.name AS name,
                   count(e) AS entity_count
            ORDER BY d.updated_at DESC
            LIMIT 1
            """
        ).single()
        if not row:
            return None
        return dict(row)


def list_session_modules(session_id: str) -> list[dict[str, Any]]:
    """社区版 Neo4j 仅一份活跃图谱；若属于该 session 则返回，否则为空。"""
    active = get_active_document()
    sid = _norm_session_id(session_id)
    if not active or active.get("session_id") != sid:
        return []
    return [
        {
            "file_id": active.get("file_id") or "",
            "name": active.get("name") or "",
            "entity_count": int(active.get("entity_count") or 0),
            "neo4j_database": NEO4J_DEFAULT_DB,
            "storage_mode": "community_replace",
        }
    ]


def _expand_in_graph(
    entity_names: list[str],
    *,
    session_id: str = "",
    file_id: str = "",
    max_hops: int = 2,
    limit: int = 40,
) -> list[dict[str, Any]]:
    names = [n.strip() for n in entity_names if n and n.strip()]
    if not names:
        return []
    hops = max(1, min(max_hops, 3))

    if file_id and session_id:
        cypher = f"""
        MATCH (seed:Entity {{file_id: $fid, session_id: $sid}})
        WHERE seed.name IN $names
        MATCH path = (seed)-[:RELATED*1..{hops}]-(other:Entity {{file_id: $fid, session_id: $sid}})
        UNWIND relationships(path) AS rel
        WITH startNode(rel) AS a, rel, endNode(rel) AS b
        RETURN DISTINCT
            a.name AS source, a.type AS source_type, a.description AS source_desc,
            rel.relation AS relation, rel.description AS rel_desc,
            b.name AS target, b.type AS target_type, b.description AS target_desc
        LIMIT $limit
        """
        params: dict[str, Any] = {
            "names": names,
            "fid": file_id,
            "sid": session_id,
            "limit": limit,
        }
    else:
        cypher = f"""
        MATCH (seed:Entity)
        WHERE seed.name IN $names
        MATCH path = (seed)-[:RELATED*1..{hops}]-(other:Entity)
        UNWIND relationships(path) AS rel
        WITH startNode(rel) AS a, rel, endNode(rel) AS b
        RETURN DISTINCT
            a.name AS source, a.type AS source_type, a.description AS source_desc,
            rel.relation AS relation, rel.description AS rel_desc,
            b.name AS target, b.type AS target_type, b.description AS target_desc
        LIMIT $limit
        """
        params = {"names": names, "limit": limit}

    with _session() as session:
        rows = [dict(rec) for rec in session.run(cypher, **params)]
        for r in rows:
            r["neo4j_database"] = NEO4J_DEFAULT_DB
        return rows


def expand_graph_from_entities(
    session_id: str,
    entity_names: list[str],
    *,
    file_ids: list[str] | None = None,
    max_hops: int = 2,
    limit: int = 40,
) -> list[dict[str, Any]]:
    """在 neo4j 单库中扩展；仅当活跃文档与请求的 file_id 一致时才有结果。"""
    active = get_active_document()
    if not active:
        return []

    sid = _norm_session_id(session_id)
    active_fid = (active.get("file_id") or "").strip()
    active_sid = (active.get("session_id") or "").strip()

    fids = [f.strip() for f in (file_ids or []) if f and str(f).strip()]
    if fids and active_fid not in fids:
        return []
    if active_sid != sid:
        return []

    try:
        return _expand_in_graph(
            entity_names,
            session_id=active_sid,
            file_id=active_fid,
            max_hops=max_hops,
            limit=limit,
        )
    except Exception:
        return []


def get_entities_by_names(
    session_id: str,
    entity_names: list[str],
    *,
    file_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    sid = _norm_session_id(session_id)
    names = [n.strip() for n in entity_names if n and n.strip()]
    if not names:
        return []

    active = get_active_document()
    if not active or active.get("session_id") != sid:
        return []

    active_fid = (active.get("file_id") or "").strip()
    fids = [f.strip() for f in (file_ids or []) if f and str(f).strip()]
    if fids and active_fid not in fids:
        return []

    with _session() as session:
        res = session.run(
            """
            MATCH (d:Document {session_id: $sid, file_id: $fid})-[:CONTAINS]->(e:Entity)
            WHERE e.name IN $names
            RETURN d.file_id AS file_id, e.name AS name, e.type AS type,
                   e.description AS description
            """,
            sid=sid,
            fid=active_fid,
            names=names,
        )
        out = []
        for rec in res:
            row = dict(rec)
            row["neo4j_database"] = NEO4J_DEFAULT_DB
            out.append(row)
        return out
