"""根据会话前几轮对话自动生成侧边栏标题。"""
from __future__ import annotations

import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from llm.llm_zhipu import make_summary_llm

logger = logging.getLogger("ai_chat.session_title")

TITLE_SYSTEM = (
    "你是对话标题助手。根据用户与助手的前几轮对话，生成一条简洁的中文会话标题。\n"
    "要求：\n"
    "1. 8～24 字，概括核心主题；\n"
    "2. 不要引号、书名号、句号；不要写「对话」「会话」「聊天」等词；\n"
    "3. 只输出标题本身，一行，无其它说明。"
)

_MAX_MSG_CHARS = 400
_MAX_MESSAGES = 6


def _strip_attachment_block(text: str) -> str:
    s = (text or "").strip()
    idx = s.find("\n\n--- 附件")
    if idx >= 0:
        s = s[:idx].strip()
    return s


def _clip(text: str, cap: int) -> str:
    s = (text or "").strip()
    return s if len(s) <= cap else s[:cap] + "…"


def format_messages_for_title(messages: list[dict]) -> str:
    parts: list[str] = []
    for m in messages[:_MAX_MESSAGES]:
        role = m.get("role") or "unknown"
        label = "用户" if role == "user" else "助手" if role == "assistant" else role
        body = _clip(_strip_attachment_block(str(m.get("content") or "")), _MAX_MSG_CHARS)
        if not body:
            continue
        parts.append(f"{label}：{body}")
    return "\n\n".join(parts)


def _clean_title(raw: str) -> str:
    s = (raw or "").strip()
    s = re.sub(r'^[\s「『"\']+|[\s」』"\']+$', "", s)
    s = re.sub(r"[。．.!！?？…]+$", "", s).strip()
    if len(s) > 48:
        s = s[:48].rstrip()
    return s


async def generate_title_from_messages(messages: list[dict]) -> str:
    block = format_messages_for_title(messages)
    if not block.strip():
        return ""
    llm = make_summary_llm()
    resp = await llm.ainvoke(
        [
            SystemMessage(content=TITLE_SYSTEM),
            HumanMessage(content=f"【对话片段】\n{block}\n\n请输出会话标题："),
        ]
    )
    text = resp.content or ""
    if isinstance(text, list):
        chunks: list[str] = []
        for item in text:
            if isinstance(item, dict) and item.get("type") == "text":
                chunks.append(str(item.get("text") or ""))
        text = "\n".join(chunks)
    return _clean_title(str(text))


async def maybe_auto_title_session(session_id: str) -> None:
    import chat.chat_store as chat_store

    sid = (session_id or "default").strip() or "default"
    meta = chat_store.get_session_meta(sid)
    if meta.get("title_manual") or meta.get("auto_title_done"):
        return
    current = (meta.get("title") or sid).strip()
    if current and current not in (sid, "default"):
        return
    msgs = chat_store.fetch_messages_range(sid, 0, _MAX_MESSAGES)
    if len(msgs) < 2:
        return
    has_user = any(m.get("role") == "user" for m in msgs)
    has_asst = any(m.get("role") == "assistant" for m in msgs)
    if not (has_user and has_asst):
        return
    try:
        title = await generate_title_from_messages(msgs)
    except Exception as e:
        logger.warning("自动生成会话标题失败 session=%s: %s", sid, e)
        return
    if not title:
        return
    chat_store.set_session_title(sid, title, manual=False)
    logger.info("会话标题已自动生成 session=%s title=%s", sid, title)


def schedule_session_auto_title(session_id: str) -> None:
    from core.async_runner import schedule_async

    schedule_async(maybe_auto_title_session(session_id))
