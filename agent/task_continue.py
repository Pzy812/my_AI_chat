"""检测 Agent 是否在外发/交付步骤前过早结束，并构造续跑提示。"""
from __future__ import annotations

import re

from langchain_core.messages import AIMessage, ToolMessage

from agent.task_state import DELIVER_KEYWORDS
from chat.chat_helpers import messages_have_pending_tool_calls

DELIVER_ACTION_TOOLS: frozenset[str] = frozenset(
    {
        "send_wechat_message",
        "send_wechat_files",
        "send_email",
        "export_to_excel",
    }
)

_DELIVER_SUCCESS_RE = re.compile(
    r"邮件已发送|消息已发送|已发送到|微信.*已发送|Excel\s*已保存|✅",
    re.IGNORECASE,
)

_PROMISE_DELIVER_RE = re.compile(
    r"(将|马上|现在|接下来|准备).{0,12}(发送|发给|发邮件|发微信|发给您|发给你|"
    r"整理成邮件|导出|交付)|"
    r"(send_email|send_wechat)",
    re.IGNORECASE,
)

MAX_DELIVER_CONTINUATIONS = 1

# thread_id → 已成功执行的外发工具名
_deliver_done: dict[str, set[str]] = {}


def _tool_content_str(content) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                parts.append(str(block.get("text") or block.get("content") or ""))
            else:
                parts.append(str(block))
        return "\n".join(parts)
    return str(content)


def is_deliver_success_text(text: str) -> bool:
    s = (text or "").strip()
    if not s or s.startswith("⏸️") or s.startswith("⛔") or s.startswith("ℹ️"):
        return False
    return bool(_DELIVER_SUCCESS_RE.search(s))


def _is_successful_deliver_tool(name: str, content_str: str) -> bool:
    text = (content_str or "").strip()
    if not text or text.startswith("⏸️") or text.startswith("⛔") or text.startswith("ℹ️"):
        return False
    if name in DELIVER_ACTION_TOOLS:
        return is_deliver_success_text(text) or bool(text)
    if is_deliver_success_text(text):
        return True
    return False


def _detect_completed_deliver_tools(messages: list) -> set[str]:
    done: set[str] = set()
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        text = _tool_content_str(m.content)
        name = (m.name or "").strip()
        if name in DELIVER_ACTION_TOOLS and _is_successful_deliver_tool(name, text):
            done.add(name)
            continue
        if not name and is_deliver_success_text(text):
            if "邮件" in text:
                done.add("send_email")
            elif "微信" in text:
                done.add("send_wechat_message")
    return done


def sync_deliver_completion_flags(thread_id: str, messages: list) -> None:
    done = _detect_completed_deliver_tools(messages)
    if not done:
        return
    tid = (thread_id or "default").strip() or "default"
    prev = _deliver_done.get(tid, set())
    _deliver_done[tid] = prev | done


def mark_deliver_tool_done(thread_id: str, tool_name: str) -> None:
    if tool_name not in DELIVER_ACTION_TOOLS:
        return
    tid = (thread_id or "default").strip() or "default"
    prev = _deliver_done.get(tid, set())
    _deliver_done[tid] = prev | {tool_name}


def is_deliver_tool_done(thread_id: str, tool_name: str) -> bool:
    tid = (thread_id or "default").strip() or "default"
    return tool_name in _deliver_done.get(tid, set())


def clear_deliver_flags(thread_id: str) -> None:
    tid = (thread_id or "default").strip() or "default"
    _deliver_done.pop(tid, None)


def deliver_duplicate_block_message(tool_name: str) -> str:
    from agent.hitl_config import hitl_tool_label

    label = hitl_tool_label(tool_name)
    return (
        f"ℹ️ {label}已在本次对话中成功执行，请勿重复调用。"
        "请直接向用户总结已完成的内容，不要再发起外发。"
    )


def user_goal_requires_deliver(user_goal: str) -> bool:
    text = (user_goal or "").strip()
    if not text:
        return False
    if any(k in text for k in DELIVER_KEYWORDS):
        return True
    return bool(re.search(r"[@＠]\w+|@\w+\.\w+", text))


def deliver_tools_used(messages: list) -> bool:
    return bool(_detect_completed_deliver_tools(messages))


def assistant_promised_deliver(reply: str) -> bool:
    text = (reply or "").strip()
    if not text:
        return False
    return bool(_PROMISE_DELIVER_RE.search(text))


def build_deliver_nudge(user_goal: str, reply: str) -> str:
    goal_preview = (user_goal or "").strip()[:500]
    return (
        "【系统提醒】用户的原始目标包含外发/交付要求，但你尚未调用相应工具完成交付。\n"
        f"原始目标摘要：{goal_preview}\n"
        "请不要再仅用文字说明「将要发送/整理后发送」，必须立即调用工具：\n"
        "· 发邮件 → send_email\n"
        "· 发微信 → send_wechat_message 或 send_wechat_files\n"
        "· 导出 Excel → export_to_excel\n"
        "若正文已在上一轮回复中整理好，请把完整内容作为工具参数直接调用。"
        + (f"\n（你刚才的回复末尾：{reply[-200:]}）" if reply else "")
    )


def should_continue_deliver(
    state: dict,
    messages: list,
    reply: str | None,
) -> str | None:
    """若 Agent 口头承诺外发但未调用工具，返回续跑提示（已成功外发则不再续跑）。"""
    if not reply or not reply.strip():
        return None
    if messages_have_pending_tool_calls(messages):
        return None
    if deliver_tools_used(messages):
        return None
    goal = (state.get("user_goal") or "").strip()
    if not user_goal_requires_deliver(goal):
        return None
    last_ai = _last_ai_message(messages)
    if last_ai and last_ai.tool_calls:
        return None
    if not assistant_promised_deliver(reply):
        return None
    return build_deliver_nudge(goal, reply)


def _last_ai_message(messages: list) -> AIMessage | None:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return m
    return None
