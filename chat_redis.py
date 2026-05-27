"""
对话记忆：按 session_id 分 key；并维护最近会话索引（ZSET）与标题（HASH）。
"""
from __future__ import annotations

import json
import os
import time
from typing import Any

import redis

REDIS_HOST = os.getenv("REDIS_HOST", "127.0.0.1")
REDIS_PORT = int(os.getenv("REDIS_PORT", "6379"))
REDIS_DB = int(os.getenv("REDIS_DB", "0"))
REDIS_CHAT_PREFIX = os.getenv("REDIS_CHAT_PREFIX", "ai_chat_web:")
REDIS_SESSION_ZSET = os.getenv("REDIS_SESSION_ZSET", "ai_chat_web:sessions")
REDIS_SESSION_META_PREFIX = os.getenv("REDIS_SESSION_META_PREFIX", "ai_chat_web:meta:")
REDIS_CHAT_MAX_MESSAGES = int(os.getenv("REDIS_CHAT_MAX_MESSAGES", "80"))
REDIS_UPLOAD_META_PREFIX = os.getenv("REDIS_UPLOAD_META_PREFIX", "ai_chat_web:upload:")
UPLOAD_META_TTL_SEC = int(os.getenv("UPLOAD_META_TTL_SEC", str(7 * 24 * 3600)))


def get_redis() -> redis.Redis:
    # socket_connect_timeout：容器内 Redis 未就绪时更快失败重试，便于排查
    return redis.Redis(
        host=REDIS_HOST,
        port=REDIS_PORT,
        db=REDIS_DB,
        decode_responses=True,
        socket_connect_timeout=10,
        socket_timeout=30,
    )


def _key(session_id: str) -> str:
    sid = (session_id or "default").strip() or "default"
    return f"{REDIS_CHAT_PREFIX}{sid}"


def touch_session(session_id: str, title: str | None = None) -> None:
    r = get_redis()
    sid = (session_id or "default").strip() or "default"
    now = time.time()
    r.zadd(REDIS_SESSION_ZSET, {sid: now})
    meta_key = f"{REDIS_SESSION_META_PREFIX}{sid}"
    if title:
        r.hset(meta_key, mapping={"title": title, "updated": str(now)})
    else:
        if not r.exists(meta_key):
            r.hset(meta_key, mapping={"title": sid, "updated": str(now)})
        else:
            r.hset(meta_key, "updated", str(now))


def _upload_meta_key(session_id: str, file_id: str) -> str:
    sid = (session_id or "default").strip() or "default"
    return f"{REDIS_UPLOAD_META_PREFIX}{sid}:{file_id}"


def save_upload_meta(session_id: str, file_id: str, meta: dict[str, Any]) -> None:
    get_redis().setex(
        _upload_meta_key(session_id, file_id),
        UPLOAD_META_TTL_SEC,
        json.dumps(meta, ensure_ascii=False),
    )


def get_upload_meta(session_id: str, file_id: str) -> dict[str, Any] | None:
    raw = get_redis().get(_upload_meta_key(session_id, file_id))
    if not raw:
        return None
    try:
        return json.loads(raw)
    except Exception:
        return None


def get_upload_metas(session_id: str, file_ids: list[str]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for fid in file_ids:
        m = get_upload_meta(session_id, fid)
        if m:
            out.append(m)
    return out


def save_message(
    session_id: str,
    role: str,
    content: str,
    mcp_attachments: list[dict[str, Any]] | None = None,
    user_uploads: list[dict[str, Any]] | None = None,
) -> None:
    msg: dict[str, Any] = {"role": role, "content": content}
    if mcp_attachments:
        msg["mcp_attachments"] = mcp_attachments
    if user_uploads:
        msg["user_uploads"] = user_uploads
    r = get_redis()
    key = _key(session_id)
    r.rpush(key, json.dumps(msg, ensure_ascii=False))
    n = r.llen(key)
    if n > REDIS_CHAT_MAX_MESSAGES:
        r.ltrim(key, -REDIS_CHAT_MAX_MESSAGES, -1)
    touch_session(session_id)


def get_all_messages(session_id: str) -> list[dict[str, Any]]:
    r = get_redis()
    raw = r.lrange(_key(session_id), 0, -1)
    messages: list[dict[str, Any]] = []
    for item in raw:
        try:
            messages.append(json.loads(item))
        except Exception:
            continue
    return messages


def clear_session(session_id: str) -> None:
    r = get_redis()
    sid = (session_id or "default").strip() or "default"
    r.delete(_key(sid))
    r.zrem(REDIS_SESSION_ZSET, sid)
    r.delete(f"{REDIS_SESSION_META_PREFIX}{sid}")


def set_session_title(session_id: str, title: str) -> None:
    sid = (session_id or "default").strip() or "default"
    r = get_redis()
    now = time.time()
    r.hset(
        f"{REDIS_SESSION_META_PREFIX}{sid}",
        mapping={"title": title.strip() or sid, "updated": str(now)},
    )
    r.zadd(REDIS_SESSION_ZSET, {sid: now})


def list_sessions(limit: int = 80) -> list[dict[str, Any]]:
    r = get_redis()
    ids = r.zrevrange(REDIS_SESSION_ZSET, 0, max(0, limit - 1))
    out: list[dict[str, Any]] = []
    for sid in ids:
        h = r.hgetall(f"{REDIS_SESSION_META_PREFIX}{sid}") or {}
        out.append(
            {
                "id": sid,
                "title": h.get("title") or sid,
                "updated": h.get("updated", ""),
                "message_count": r.llen(_key(sid)),
            }
        )
    return out
