"""会话历史自动摘要：超过 N 轮时用智谱小模型压缩较早对话。"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from app_config import (
    CHAT_SUMMARY_ENABLED,
    CHAT_SUMMARY_KEEP_ROUNDS,
    CHAT_SUMMARY_MAX_INPUT_CHARS,
    CHAT_SUMMARY_MAX_MSG_CHARS,
    CHAT_SUMMARY_ROUNDS,
)
from chat_helpers import dict_history_to_lc_messages
from llm_zhipu import make_summary_llm

logger = logging.getLogger("ai_chat.summary")

SUMMARY_SYSTEM = (
    "你是对话摘要助手。将用户与助手的历史对话压缩为简洁中文摘要，供后续大模型理解上下文。\n"
    "要求：\n"
    "1. 保留用户目标、已做决定、关键事实、数字、文件名、待办与约束；\n"
    "2. 省略寒暄与重复；不要编造未出现的信息；\n"
    "3. 若已有摘要，在其基础上合并新内容，输出一份更新后的完整摘要（非增量片段）；\n"
    "4. 控制在 800 字以内，可用条目列表。"
)


def count_user_rounds(messages: list[dict[str, Any]]) -> int:
    return sum(1 for m in messages if m.get("role") == "user")


def split_messages_by_rounds(
    messages: list[dict[str, Any]],
    keep_rounds: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """按用户轮次切分：较早部分 vs 保留原文的最近 keep_rounds 轮。"""
    if keep_rounds <= 0:
        return [], list(messages)
    user_indices = [i for i, m in enumerate(messages) if m.get("role") == "user"]
    if len(user_indices) <= keep_rounds:
        return [], list(messages)
    split_at = user_indices[-keep_rounds]
    return list(messages[:split_at]), list(messages[split_at:])


def _clip_text(text: str, cap: int) -> str:
    s = (text or "").strip()
    if len(s) <= cap:
        return s
    return s[:cap] + "\n…(已截断)"


def format_messages_for_summary(messages: list[dict[str, Any]]) -> str:
    parts: list[str] = []
    total = 0
    cap = CHAT_SUMMARY_MAX_INPUT_CHARS
    per_msg = CHAT_SUMMARY_MAX_MSG_CHARS
    for m in messages:
        role = m.get("role") or "unknown"
        label = "用户" if role == "user" else "助手" if role == "assistant" else role
        body = _clip_text(str(m.get("content") or ""), per_msg)
        if not body:
            continue
        line = f"{label}：{body}"
        if total + len(line) > cap:
            remain = cap - total
            if remain > 80:
                parts.append(line[:remain] + "\n…(后续对话因长度限制未写入摘要输入)")
            break
        parts.append(line)
        total += len(line) + 1
    return "\n\n".join(parts)


def _summary_user_prompt(existing_summary: str, new_block: str) -> str:
    chunks: list[str] = []
    if (existing_summary or "").strip():
        chunks.append(f"【已有摘要】\n{existing_summary.strip()}")
    chunks.append(f"【待合并的对话】\n{new_block}")
    chunks.append("请输出更新后的完整对话摘要。")
    return "\n\n".join(chunks)


async def summarize_dialogue(
    messages: list[dict[str, Any]],
    existing_summary: str = "",
) -> str:
    block = format_messages_for_summary(messages)
    if not block.strip():
        return (existing_summary or "").strip()
    llm = make_summary_llm()
    resp = await llm.ainvoke(
        [
            SystemMessage(content=SUMMARY_SYSTEM),
            HumanMessage(content=_summary_user_prompt(existing_summary, block)),
        ]
    )
    text = (resp.content or "").strip()
    if isinstance(text, list):
        parts = []
        for block in text:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
        text = "\n".join(parts).strip()
    if not text:
        logger.warning("摘要模型返回空，沿用已有摘要")
        return (existing_summary or "").strip()
    return text


def lc_messages_with_summary(
    summary: str,
    recent_messages: list[dict[str, Any]],
) -> list:
    """将摘要注入为一条模拟对话，再拼接最近原文。"""
    if not (summary or "").strip():
        return dict_history_to_lc_messages(recent_messages)
    prefix = [
        HumanMessage(
            content=(
                "【此前对话摘要（系统自动生成，供理解上下文）】\n"
                f"{summary.strip()}\n\n"
                "请结合以上摘要与后续完整对话回答，无需向用户重复说明摘要本身。"
            )
        ),
        AIMessage(content="好的，我已了解此前对话要点。"),
    ]
    return prefix + dict_history_to_lc_messages(recent_messages)


async def build_agent_lc_messages(
    session_id: str,
    messages: list[dict[str, Any]],
) -> list:
    """
    根据消息列表构建发给 Agent 的 LangChain 消息。
    超过 CHAT_SUMMARY_ROUNDS 轮时，对更早对话做小模型摘要（摘要持久化，支持增量合并）。
    """
    if not CHAT_SUMMARY_ENABLED:
        return dict_history_to_lc_messages(messages)

    rounds = count_user_rounds(messages)
    if rounds <= CHAT_SUMMARY_ROUNDS:
        return dict_history_to_lc_messages(messages)

    _, recent = split_messages_by_rounds(messages, CHAT_SUMMARY_KEEP_ROUNDS)
    if not recent:
        return dict_history_to_lc_messages(messages)

    import chat_store

    total = chat_store.count_messages(session_id)
    recent_count = len(recent)
    summarize_end = max(0, total - recent_count)

    meta = chat_store.get_summary_meta(session_id)
    stored_summary = (meta.get("context_summary") or "").strip()
    covered = int(meta.get("summary_message_count") or 0)

    if covered > summarize_end:
        covered = 0
        stored_summary = ""

    if covered >= summarize_end and stored_summary:
        summary = stored_summary
    else:
        to_summarize = chat_store.fetch_messages_range(
            session_id, covered, summarize_end - covered
        )
        try:
            summary = await summarize_dialogue(
                to_summarize,
                stored_summary if covered > 0 else "",
            )
        except Exception as e:
            logger.warning(
                "会话摘要失败 session=%s: %s，降级为仅最近原文",
                session_id,
                e,
            )
            return dict_history_to_lc_messages(recent)
        chat_store.save_summary_meta(
            session_id,
            context_summary=summary,
            summary_through_index=summarize_end,
        )

    logger.info(
        "会话摘要已注入 session=%s rounds=%s summarize_end=%s recent_msgs=%s summary_len=%s",
        session_id,
        rounds,
        summarize_end,
        len(recent),
        len(summary),
    )
    return lc_messages_with_summary(summary, recent)


async def prepare_agent_lc_messages(session_id: str) -> list:
    """读取最近消息并应用摘要策略，返回 Agent 用 LangChain 消息列表。"""
    import chat_store

    rows = chat_store.get_recent_messages(session_id)
    return await build_agent_lc_messages(session_id, rows)
