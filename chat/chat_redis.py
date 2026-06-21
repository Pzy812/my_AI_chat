"""
Redis 热缓存：消息 List、会话 ZSET/ HASH、上传 meta TTL。
业务写入请走 chat_store；本模块提供 cache_* 供 chat_store 调用。
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


def _norm_sid(session_id: str) -> str:
    return (session_id or "default").strip() or "default"


def _pack_message(
    role: str,
    content: str,
    mcp_attachments: list[dict[str, Any]] | None = None,
    user_uploads: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    msg: dict[str, Any] = {"role": role, "content": content}
    if mcp_attachments:
        msg["mcp_attachments"] = mcp_attachments
    if user_uploads:
        msg["user_uploads"] = user_uploads
    return msg


def _parse_messages(raw: list[str]) -> list[dict[str, Any]]:
    messages: list[dict[str, Any]] = []
    for item in raw:
        try:
            messages.append(json.loads(item))
        except Exception:
            continue
    return messages


# --- 消息缓存 ---


def cache_messages_exists(session_id: str) -> bool:
    return bool(get_redis().exists(_key(session_id)))


def cache_get_messages(session_id: str) -> list[dict[str, Any]]:
    raw = get_redis().lrange(_key(session_id), 0, -1)
    return _parse_messages(raw)


def cache_set_messages(session_id: str, messages: list[dict[str, Any]]) -> None:
    sid = _norm_sid(session_id)
    r = get_redis()
    key = _key(sid)
    pipe = r.pipeline()
    pipe.delete(key)
    for msg in messages[-REDIS_CHAT_MAX_MESSAGES:]:
        pipe.rpush(key, json.dumps(msg, ensure_ascii=False))
    pipe.execute()


def cache_append_message(
    session_id: str,
    role: str,
    content: str,
    mcp_attachments: list[dict[str, Any]] | None = None,
    user_uploads: list[dict[str, Any]] | None = None,
) -> None:
    msg = _pack_message(role, content, mcp_attachments, user_uploads)
    r = get_redis()
    key = _key(session_id)
    r.rpush(key, json.dumps(msg, ensure_ascii=False))
    n = r.llen(key)
    if n > REDIS_CHAT_MAX_MESSAGES:
        r.ltrim(key, -REDIS_CHAT_MAX_MESSAGES, -1)


def cache_touch_session(session_id: str, title: str | None = None) -> None:
    touch_session(session_id, title)


# --- 会话索引缓存 ---


def touch_session(session_id: str, title: str | None = None) -> None:
    r = get_redis()
    sid = _norm_sid(session_id)
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


def cache_sync_sessions(sessions: list[dict[str, Any]]) -> None:
    """将 Postgres 会话列表回填 Redis ZSET/HASH。"""
    r = get_redis()
    for s in sessions:
        sid = s.get("id") or "default"
        updated = s.get("updated") or ""
        try:
            score = float(updated) if updated.replace(".", "", 1).isdigit() else time.time()
        except Exception:
            score = time.time()
        if isinstance(updated, str) and "T" in updated:
            try:
                from datetime import datetime

                score = datetime.fromisoformat(updated.replace("Z", "+00:00")).timestamp()
            except Exception:
                score = time.time()
        r.zadd(REDIS_SESSION_ZSET, {sid: score})
        r.hset(
            f"{REDIS_SESSION_META_PREFIX}{sid}",
            mapping={"title": s.get("title") or sid, "updated": str(updated)},
        )


def _upload_meta_key(session_id: str, file_id: str) -> str:
    return f"{REDIS_UPLOAD_META_PREFIX}{_norm_sid(session_id)}:{file_id}"


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


# --- 仅 Redis 模式（未配置 Postgres 时 chat_store 降级调用） ---


def save_message(
    session_id: str,
    role: str,
    content: str,
    mcp_attachments: list[dict[str, Any]] | None = None,
    user_uploads: list[dict[str, Any]] | None = None,
) -> None:
    cache_append_message(
        session_id, role, content,
        mcp_attachments=mcp_attachments,
        user_uploads=user_uploads,
    )
    touch_session(session_id)


def get_all_messages(session_id: str) -> list[dict[str, Any]]:
    return cache_get_messages(session_id)


def clear_session(session_id: str) -> None:
    r = get_redis()
    sid = _norm_sid(session_id)
    r.delete(_key(sid))
    r.zrem(REDIS_SESSION_ZSET, sid)
    r.delete(f"{REDIS_SESSION_META_PREFIX}{sid}")


def get_session_summary_meta(session_id: str) -> dict[str, Any]:
    sid = _norm_sid(session_id)
    h = get_redis().hgetall(f"{REDIS_SESSION_META_PREFIX}{sid}") or {}
    count = h.get("summary_message_count") or h.get("summary_through_index") or 0
    return {
        "context_summary": h.get("context_summary") or "",
        "summary_message_count": int(count),
    }


def set_session_summary_meta(
    session_id: str,
    *,
    context_summary: str,
    summary_through_index: int,
) -> None:
    sid = _norm_sid(session_id)
    r = get_redis()
    meta_key = f"{REDIS_SESSION_META_PREFIX}{sid}"
    count = str(max(0, summary_through_index))
    r.hset(
        meta_key,
        mapping={
            "context_summary": context_summary,
            "summary_message_count": count,
            "summary_through_index": count,
        },
    )


def clear_session_summary_meta(session_id: str) -> None:
    sid = _norm_sid(session_id)
    r = get_redis()
    meta_key = f"{REDIS_SESSION_META_PREFIX}{sid}"
    r.hdel(meta_key, "context_summary", "summary_message_count", "summary_through_index")


def count_messages(session_id: str) -> int:
    return get_redis().llen(_key(session_id))


def fetch_messages_range(session_id: str, offset: int, limit: int) -> list[dict[str, Any]]:
    off = max(0, offset)
    lim = max(0, limit)
    if lim <= 0:
        return []
    raw = get_redis().lrange(_key(session_id), off, off + lim - 1)
    return _parse_messages(raw)


def get_session_meta(session_id: str) -> dict[str, Any]:
    sid = _norm_sid(session_id)
    h = get_redis().hgetall(f"{REDIS_SESSION_META_PREFIX}{sid}") or {}
    return {
        "title": h.get("title") or sid,
        "title_manual": h.get("title_manual") in ("1", "true", "True"),
        "auto_title_done": h.get("auto_title_done") in ("1", "true", "True"),
    }


def set_session_title(session_id: str, title: str, *, manual: bool = True) -> None:
    sid = _norm_sid(session_id)
    r = get_redis()
    now = time.time()
    mapping: dict[str, str] = {
        "title": (title or sid).strip() or sid,
        "updated": str(now),
    }
    if manual:
        mapping["title_manual"] = "1"
    else:
        mapping["auto_title_done"] = "1"
        mapping["title_manual"] = "0"
    r.hset(f"{REDIS_SESSION_META_PREFIX}{sid}", mapping=mapping)
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
