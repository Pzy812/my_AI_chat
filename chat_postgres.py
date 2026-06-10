"""PostgreSQL 权威存储：会话、消息、上传元数据。"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timedelta, timezone
from typing import Any

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

from app_config import (
    CHAT_AGENT_CONTEXT_MESSAGES,
    CHAT_HISTORY_MAX_MESSAGES,
    POSTGRES_URI,
    UPLOAD_META_TTL_SEC,
)

logger = logging.getLogger("ai_chat.pg")

_pool: ConnectionPool | None = None
_schema_ready = False


def enabled() -> bool:
    return bool(POSTGRES_URI)


def _norm_session_id(session_id: str) -> str:
    return (session_id or "default").strip() or "default"


def _get_pool() -> ConnectionPool:
    global _pool
    if _pool is None:
        if not POSTGRES_URI:
            raise RuntimeError("未配置 POSTGRES_URI，无法使用 PostgreSQL 聊天存储")
        _pool = ConnectionPool(
            conninfo=POSTGRES_URI,
            min_size=1,
            max_size=10,
            kwargs={"row_factory": dict_row},
        )
    return _pool


def _msg_row_to_dict(row: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"role": row["role"], "content": row["content"] or ""}
    extras = row.get("extras") or {}
    if isinstance(extras, str):
        try:
            extras = json.loads(extras)
        except Exception:
            extras = {}
    if extras.get("mcp_attachments"):
        out["mcp_attachments"] = extras["mcp_attachments"]
    if extras.get("user_uploads"):
        out["user_uploads"] = extras["user_uploads"]
    return out


def _extras_from_message(
    mcp_attachments: list[dict[str, Any]] | None,
    user_uploads: list[dict[str, Any]] | None,
) -> dict[str, Any]:
    extras: dict[str, Any] = {}
    if mcp_attachments:
        extras["mcp_attachments"] = mcp_attachments
    if user_uploads:
        extras["user_uploads"] = user_uploads
    return extras


def init_schema() -> None:
    """创建业务表（与 LangGraph checkpoint 表互不干扰）。"""
    global _schema_ready
    if not enabled():
        return
    if _schema_ready:
        return
    ddl = """
    CREATE TABLE IF NOT EXISTS chat_sessions (
        session_id TEXT PRIMARY KEY,
        title TEXT NOT NULL DEFAULT '',
        updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
        extras JSONB NOT NULL DEFAULT '{}'::jsonb
    );
    CREATE TABLE IF NOT EXISTS chat_messages (
        id BIGSERIAL PRIMARY KEY,
        session_id TEXT NOT NULL REFERENCES chat_sessions(session_id) ON DELETE CASCADE,
        role TEXT NOT NULL,
        content TEXT NOT NULL DEFAULT '',
        extras JSONB NOT NULL DEFAULT '{}'::jsonb,
        created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_chat_messages_session_created
        ON chat_messages (session_id, created_at);
    CREATE TABLE IF NOT EXISTS chat_upload_meta (
        session_id TEXT NOT NULL,
        file_id TEXT NOT NULL,
        meta JSONB NOT NULL,
        expires_at TIMESTAMPTZ,
        PRIMARY KEY (session_id, file_id)
    );
    """
    with _get_pool().connection() as conn:
        conn.execute(ddl)
        conn.execute(
            "ALTER TABLE chat_sessions ADD COLUMN IF NOT EXISTS extras JSONB NOT NULL DEFAULT '{}'::jsonb"
        )
        conn.commit()
    _schema_ready = True
    logger.info("PostgreSQL 聊天表已就绪")


def touch_session(session_id: str, title: str | None = None) -> None:
    sid = _norm_session_id(session_id)
    with _get_pool().connection() as conn:
        conn.execute(
            """
            INSERT INTO chat_sessions (session_id, title, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (session_id) DO UPDATE SET
                title = CASE WHEN %s <> '' THEN EXCLUDED.title ELSE chat_sessions.title END,
                updated_at = NOW()
            """,
            (sid, (title or sid).strip() or sid, (title or "").strip()),
        )
        conn.commit()


def save_message(
    session_id: str,
    role: str,
    content: str,
    mcp_attachments: list[dict[str, Any]] | None = None,
    user_uploads: list[dict[str, Any]] | None = None,
) -> None:
    sid = _norm_session_id(session_id)
    extras = _extras_from_message(mcp_attachments, user_uploads)
    touch_session(sid)
    with _get_pool().connection() as conn:
        conn.execute(
            """
            INSERT INTO chat_messages (session_id, role, content, extras)
            VALUES (%s, %s, %s, %s::jsonb)
            """,
            (sid, role, content, json.dumps(extras, ensure_ascii=False)),
        )
        conn.execute(
            "UPDATE chat_sessions SET updated_at = NOW() WHERE session_id = %s",
            (sid,),
        )
        conn.commit()


def fetch_messages(
    session_id: str,
    *,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """按时间正序返回最近 limit 条消息。"""
    sid = _norm_session_id(session_id)
    cap = limit or CHAT_HISTORY_MAX_MESSAGES
    sql = """
        SELECT role, content, extras
        FROM (
            SELECT role, content, extras, created_at, id
            FROM chat_messages
            WHERE session_id = %s
            ORDER BY created_at DESC, id DESC
            LIMIT %s
        ) sub
        ORDER BY sub.created_at ASC, sub.id ASC
    """
    with _get_pool().connection() as conn:
        rows = conn.execute(sql, (sid, cap)).fetchall()
    return [_msg_row_to_dict(r) for r in rows]


def fetch_messages_range(
    session_id: str,
    offset: int,
    limit: int,
) -> list[dict[str, Any]]:
    """按时间正序返回会话内 [offset, offset+limit) 条消息（用于摘要增量读取）。"""
    sid = _norm_session_id(session_id)
    off = max(0, offset)
    lim = max(0, limit)
    if lim <= 0:
        return []
    sql = """
        SELECT role, content, extras
        FROM chat_messages
        WHERE session_id = %s
        ORDER BY created_at ASC, id ASC
        OFFSET %s LIMIT %s
    """
    with _get_pool().connection() as conn:
        rows = conn.execute(sql, (sid, off, lim)).fetchall()
    return [_msg_row_to_dict(r) for r in rows]


def count_messages(session_id: str) -> int:
    sid = _norm_session_id(session_id)
    with _get_pool().connection() as conn:
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM chat_messages WHERE session_id = %s",
            (sid,),
        ).fetchone()
    return int(row["n"]) if row else 0


def get_session_summary_meta(session_id: str) -> dict[str, Any]:
    sid = _norm_session_id(session_id)
    with _get_pool().connection() as conn:
        row = conn.execute(
            "SELECT extras FROM chat_sessions WHERE session_id = %s",
            (sid,),
        ).fetchone()
    extras = (row or {}).get("extras") or {}
    if isinstance(extras, str):
        try:
            extras = json.loads(extras)
        except Exception:
            extras = {}
    if not isinstance(extras, dict):
        extras = {}
    count = extras.get("summary_message_count")
    if count is None:
        count = extras.get("summary_through_index")
    return {
        "context_summary": extras.get("context_summary") or "",
        "summary_message_count": int(count or 0),
    }


def set_session_summary_meta(
    session_id: str,
    *,
    context_summary: str,
    summary_through_index: int,
) -> None:
    sid = _norm_session_id(session_id)
    count = max(0, summary_through_index)
    patch = {
        "context_summary": context_summary,
        "summary_message_count": count,
        "summary_through_index": count,
    }
    with _get_pool().connection() as conn:
        conn.execute(
            """
            INSERT INTO chat_sessions (session_id, title, updated_at, extras)
            VALUES (%s, %s, NOW(), %s::jsonb)
            ON CONFLICT (session_id) DO UPDATE SET
                extras = COALESCE(chat_sessions.extras, '{}'::jsonb) || EXCLUDED.extras,
                updated_at = NOW()
            """,
            (sid, sid, json.dumps(patch, ensure_ascii=False)),
        )
        conn.commit()


def clear_session(session_id: str) -> None:
    sid = _norm_session_id(session_id)
    with _get_pool().connection() as conn:
        conn.execute("DELETE FROM chat_sessions WHERE session_id = %s", (sid,))
        conn.commit()


def set_session_title(session_id: str, title: str) -> None:
    sid = _norm_session_id(session_id)
    t = (title or sid).strip() or sid
    with _get_pool().connection() as conn:
        conn.execute(
            """
            INSERT INTO chat_sessions (session_id, title, updated_at)
            VALUES (%s, %s, NOW())
            ON CONFLICT (session_id) DO UPDATE SET title = EXCLUDED.title, updated_at = NOW()
            """,
            (sid, t),
        )
        conn.commit()


def list_sessions(limit: int = 80) -> list[dict[str, Any]]:
    cap = max(1, min(limit, 200))
    with _get_pool().connection() as conn:
        rows = conn.execute(
            """
            SELECT s.session_id, s.title, s.updated_at,
                   (SELECT COUNT(*) FROM chat_messages m WHERE m.session_id = s.session_id) AS message_count
            FROM chat_sessions s
            ORDER BY s.updated_at DESC
            LIMIT %s
            """,
            (cap,),
        ).fetchall()
    out: list[dict[str, Any]] = []
    for row in rows:
        updated = row.get("updated_at")
        out.append(
            {
                "id": row["session_id"],
                "title": row.get("title") or row["session_id"],
                "updated": updated.isoformat() if isinstance(updated, datetime) else str(updated or ""),
                "message_count": int(row.get("message_count") or 0),
            }
        )
    return out


def save_upload_meta(session_id: str, file_id: str, meta: dict[str, Any]) -> None:
    sid = _norm_session_id(session_id)
    expires_at = datetime.now(timezone.utc) + timedelta(seconds=UPLOAD_META_TTL_SEC)
    with _get_pool().connection() as conn:
        conn.execute(
            """
            INSERT INTO chat_upload_meta (session_id, file_id, meta, expires_at)
            VALUES (%s, %s, %s::jsonb, %s)
            ON CONFLICT (session_id, file_id) DO UPDATE SET
                meta = EXCLUDED.meta,
                expires_at = EXCLUDED.expires_at
            """,
            (sid, file_id, json.dumps(meta, ensure_ascii=False), expires_at),
        )
        conn.commit()


def get_upload_meta(session_id: str, file_id: str) -> dict[str, Any] | None:
    sid = _norm_session_id(session_id)
    with _get_pool().connection() as conn:
        row = conn.execute(
            """
            SELECT meta FROM chat_upload_meta
            WHERE session_id = %s AND file_id = %s
              AND (expires_at IS NULL OR expires_at > NOW())
            """,
            (sid, file_id),
        ).fetchone()
    if not row:
        return None
    meta = row["meta"]
    if isinstance(meta, dict):
        return meta
    try:
        return json.loads(meta)
    except Exception:
        return None


def get_upload_metas(session_id: str, file_ids: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for fid in file_ids:
        m = get_upload_meta(session_id, fid)
        if m:
            out.append(m)
    return out


def get_recent_messages(session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
    cap = limit or CHAT_AGENT_CONTEXT_MESSAGES
    return fetch_messages(session_id, limit=cap)


def close_pool() -> None:
    global _pool, _schema_ready
    if _pool is not None:
        _pool.close()
        _pool = None
    _schema_ready = False
