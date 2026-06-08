"""聊天存储门面：Postgres 权威写入，Redis 热缓存读取回填。"""
from __future__ import annotations

import logging
from typing import Any

import chat_postgres
import chat_redis
from app_config import CHAT_AGENT_CONTEXT_MESSAGES, CHAT_HISTORY_MAX_MESSAGES, POSTGRES_URI

logger = logging.getLogger("ai_chat.store")


def init_chat_store() -> None:
    """应用启动时初始化 Postgres 表结构。"""
    if POSTGRES_URI:
        try:
            chat_postgres.init_schema()
            logger.info("聊天存储：Postgres 权威 + Redis 缓存")
        except Exception as e:
            logger.warning("Postgres 聊天表初始化失败，将降级为仅 Redis：%s", e)
    else:
        logger.info("未配置 POSTGRES_URI，聊天存储使用仅 Redis 模式")


def _pg_ok() -> bool:
    return bool(POSTGRES_URI) and chat_postgres.enabled()


def save_message(
    session_id: str,
    role: str,
    content: str,
    mcp_attachments: list[dict[str, Any]] | None = None,
    user_uploads: list[dict[str, Any]] | None = None,
) -> None:
    """先写 Postgres，再更新 Redis 缓存。"""
    if _pg_ok():
        chat_postgres.save_message(
            session_id, role, content,
            mcp_attachments=mcp_attachments,
            user_uploads=user_uploads,
        )
    else:
        chat_redis.save_message(
            session_id, role, content,
            mcp_attachments=mcp_attachments,
            user_uploads=user_uploads,
        )
        return
    chat_redis.cache_append_message(
        session_id, role, content,
        mcp_attachments=mcp_attachments,
        user_uploads=user_uploads,
    )
    chat_redis.cache_touch_session(session_id)


def get_all_messages(session_id: str, *, limit: int | None = None) -> list[dict[str, Any]]:
    """读取全量/较长历史：Redis 命中 → 否则 Postgres → 回填 Redis。"""
    cap = limit or CHAT_HISTORY_MAX_MESSAGES
    if chat_redis.cache_messages_exists(session_id):
        msgs = chat_redis.cache_get_messages(session_id)
        return msgs[-cap:] if len(msgs) > cap else msgs
    if _pg_ok():
        msgs = chat_postgres.fetch_messages(session_id, limit=cap)
        chat_redis.cache_set_messages(
            session_id, msgs[-chat_redis.REDIS_CHAT_MAX_MESSAGES :]
        )
        chat_redis.cache_touch_session(session_id)
        return msgs
    return chat_redis.get_all_messages(session_id)


def get_recent_messages(session_id: str, limit: int | None = None) -> list[dict[str, Any]]:
    """Agent 上下文：仅最近 N 条。"""
    cap = limit or CHAT_AGENT_CONTEXT_MESSAGES
    if chat_redis.cache_messages_exists(session_id):
        msgs = chat_redis.cache_get_messages(session_id)
        return msgs[-cap:] if len(msgs) > cap else msgs
    if _pg_ok():
        msgs = chat_postgres.get_recent_messages(session_id, limit=cap)
        chat_redis.cache_set_messages(session_id, msgs)
        chat_redis.cache_touch_session(session_id)
        return msgs
    msgs = chat_redis.get_all_messages(session_id)
    return msgs[-cap:] if len(msgs) > cap else msgs


def clear_session(session_id: str) -> None:
    if _pg_ok():
        chat_postgres.clear_session(session_id)
    chat_redis.clear_session(session_id)
    _clear_agent_checkpoint(session_id)


def _clear_agent_checkpoint(session_id: str) -> None:
    import asyncio

    from agent_checkpointer import reset_agent_thread

    try:
        asyncio.get_running_loop()
        return
    except RuntimeError:
        pass
    try:
        asyncio.run(reset_agent_thread(session_id))
    except Exception:
        pass


def set_session_title(session_id: str, title: str) -> None:
    if _pg_ok():
        chat_postgres.set_session_title(session_id, title)
    chat_redis.set_session_title(session_id, title)


def list_sessions(limit: int = 80) -> list[dict[str, Any]]:
    if _pg_ok():
        sessions = chat_postgres.list_sessions(limit)
        chat_redis.cache_sync_sessions(sessions)
        return sessions
    return chat_redis.list_sessions(limit)


def save_upload_meta(session_id: str, file_id: str, meta: dict[str, Any]) -> None:
    if _pg_ok():
        chat_postgres.save_upload_meta(session_id, file_id, meta)
    chat_redis.save_upload_meta(session_id, file_id, meta)


def get_upload_meta(session_id: str, file_id: str) -> dict[str, Any] | None:
    cached = chat_redis.get_upload_meta(session_id, file_id)
    if cached is not None:
        return cached
    if _pg_ok():
        meta = chat_postgres.get_upload_meta(session_id, file_id)
        if meta:
            chat_redis.save_upload_meta(session_id, file_id, meta)
        return meta
    return None


def get_upload_metas(session_id: str, file_ids: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for fid in file_ids:
        m = get_upload_meta(session_id, fid)
        if m:
            out.append(m)
    return out
