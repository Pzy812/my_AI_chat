"""检测 Agent 是否在外发/交付步骤前过早结束，并构造续跑提示。"""
from __future__ import annotations

import re
from threading import Lock

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from agent.task_state import DELIVER_KEYWORDS
from chat.chat_helpers import messages_have_pending_tool_calls

_NON_USER_HUMAN_PREFIXES = ("【系统", "【此前对话摘要")

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

# thread_id → 当前用户轮次序号 / 该轮已成功执行的外发工具名
_deliver_turn_serial: dict[str, int] = {}
_deliver_done: dict[str, set[str]] = {}
_deliver_inflight: dict[str, set[str]] = {}
_deliver_lock = Lock()


def _human_message_text(msg: HumanMessage) -> str:
    content = msg.content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts).strip()
    return str(content or "").strip()


def is_real_user_message(msg: BaseMessage) -> bool:
    """排除系统自动续跑等注入的 HumanMessage。"""
    if not isinstance(msg, HumanMessage):
        return False
    text = _human_message_text(msg).strip()
    if not text:
        return False
    return not any(text.startswith(prefix) for prefix in _NON_USER_HUMAN_PREFIXES)


def count_real_user_messages(messages: list) -> int:
    return sum(1 for m in (messages or []) if is_real_user_message(m))


def messages_in_current_user_turn(messages: list) -> list:
    """当前用户轮次内的消息（自最近一条真实用户消息起，含续跑 nudge 之后的内容）。"""
    if not messages:
        return []
    last_user_idx = -1
    for i, m in enumerate(messages):
        if is_real_user_message(m):
            last_user_idx = i
    if last_user_idx < 0:
        return list(messages)
    return list(messages[last_user_idx:])


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
        return is_deliver_success_text(text)
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
    """仅根据当前用户轮次内的 ToolMessage 同步外发完成标记（跨轮次可再次外发）。"""
    tid = (thread_id or "default").strip() or "default"
    serial = count_real_user_messages(messages)
    turn_messages = messages_in_current_user_turn(messages)
    done = _detect_completed_deliver_tools(turn_messages)
    prev_serial = _deliver_turn_serial.get(tid)
    if prev_serial != serial:
        _deliver_turn_serial[tid] = serial
        _deliver_done[tid] = done
        return
    _deliver_done[tid] = done


def get_deliver_done_tools(thread_id: str) -> set[str]:
    tid = (thread_id or "default").strip() or "default"
    return set(_deliver_done.get(tid, set()))


def mark_deliver_tool_done(thread_id: str, tool_name: str) -> None:
    if tool_name not in DELIVER_ACTION_TOOLS:
        return
    tid = (thread_id or "default").strip() or "default"
    with _deliver_lock:
        prev = _deliver_done.get(tid, set())
        _deliver_done[tid] = prev | {tool_name}
        _deliver_inflight.setdefault(tid, set()).discard(tool_name)
    try:
        from agent.harness import patch_deliver_done_in_run_context

        patch_deliver_done_in_run_context(tid, tool_name)
    except Exception:
        pass


def is_deliver_tool_done(thread_id: str, tool_name: str) -> bool:
    tid = (thread_id or "default").strip() or "default"
    try:
        from agent.harness import get_run_context_deliver_done

        ctx_done = get_run_context_deliver_done(tid)
        if ctx_done is not None:
            return tool_name in ctx_done
    except Exception:
        pass
    return tool_name in _deliver_done.get(tid, set())


def reserve_deliver_tool(thread_id: str, tool_name: str) -> bool:
    """Atomically reserve one side effect so parallel tool calls cannot duplicate it."""
    if tool_name not in DELIVER_ACTION_TOOLS:
        return True
    tid = (thread_id or "default").strip() or "default"
    with _deliver_lock:
        if tool_name in _deliver_done.get(tid, set()):
            return False
        inflight = _deliver_inflight.setdefault(tid, set())
        if tool_name in inflight:
            return False
        inflight.add(tool_name)
        return True


def release_deliver_tool(thread_id: str, tool_name: str) -> None:
    """Release a failed/cancelled reservation so a later legitimate retry can run."""
    if tool_name not in DELIVER_ACTION_TOOLS:
        return
    tid = (thread_id or "default").strip() or "default"
    with _deliver_lock:
        _deliver_inflight.setdefault(tid, set()).discard(tool_name)


def clear_deliver_flags(thread_id: str) -> None:
    tid = (thread_id or "default").strip() or "default"
    with _deliver_lock:
        _deliver_done.pop(tid, None)
        _deliver_turn_serial.pop(tid, None)
        _deliver_inflight.pop(tid, None)
    try:
        from agent.harness import clear_run_context_deliver_state

        clear_run_context_deliver_state(tid)
    except Exception:
        pass


def deliver_duplicate_block_message(tool_name: str) -> str:
    from agent.hitl_config import hitl_tool_label

    label = hitl_tool_label(tool_name)
    return (
        f"⛔ {label}在本轮用户请求中已成功执行过，本次重复调用未实际执行。"
        "请不要再重复调用该外发工具，直接向用户总结已发送的内容。"
    )


def user_goal_requires_deliver(user_goal: str) -> bool:
    text = (user_goal or "").strip()
    if not text:
        return False
    if any(k in text for k in DELIVER_KEYWORDS):
        return True
    return bool(re.search(r"[@＠]\w+|@\w+\.\w+", text))


_GATHER_INTENT_KEYWORDS = (
    "搜索",
    "联网",
    "查询",
    "查找",
    "查一下",
    "读取",
    "读文件",
    "目录",
    "文件夹",
    "附件",
    "论文",
    "新闻",
    "收集",
    "汇总",
    "整理",
    "表格",
    "天气",
    "股价",
    "汇率",
    "web_search",
    "list_local",
    "glob_local",
    "read_local",
)


def goal_requires_gather(user_goal: str, *, file_count: int = 0) -> bool:
    """用户目标是否必须先做信息收集（搜索/读文件等）再外发。"""
    if file_count > 0:
        return True
    text = (user_goal or "").strip()
    if not text:
        return False
    return any(k in text for k in _GATHER_INTENT_KEYWORDS)


def deliver_tools_used(messages: list) -> bool:
    return bool(_detect_completed_deliver_tools(messages_in_current_user_turn(messages)))


def required_deliver_tools(user_goal: str) -> set[str]:
    text = (user_goal or "").strip().lower()
    required: set[str] = set()
    if re.search(r"[@＠]\w+|@\w+\.\w+", text) or any(
        k in text for k in ("邮件", "邮箱", "email")
    ):
        required.add("send_email")
    if "微信" in text:
        required.add("send_wechat_message")
    if any(k in text for k in ("excel", "xlsx", "导出表格")):
        required.add("export_to_excel")
    return required


def deliver_goal_satisfied(user_goal: str, messages: list) -> bool:
    required = required_deliver_tools(user_goal)
    if not required:
        return False
    completed = _detect_completed_deliver_tools(messages_in_current_user_turn(messages))
    return required.issubset(completed)


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
    goal = (state.get("user_goal") or "").strip()
    if deliver_goal_satisfied(goal, messages):
        return None
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
