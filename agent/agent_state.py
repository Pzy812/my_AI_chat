"""LangGraph Agent 运行结束态：HITL interrupt / 未完成 tool_calls 判定。"""
from __future__ import annotations

import asyncio
import logging
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage

from chat.chat_helpers import last_assistant_text, messages_have_pending_tool_calls
from agent.hitl_config import HITL_TOOL_NAMES, hitl_payload_for_tool
from agent.hitl_tools import normalize_interrupts

logger = logging.getLogger("ai_chat.agent_state")


def _answered_tool_call_ids(msgs: list) -> set[str]:
    answered: set[str] = set()
    for m in msgs:
        if isinstance(m, ToolMessage):
            tid = getattr(m, "tool_call_id", None)
            if tid:
                answered.add(str(tid))
    return answered


def synthetic_hitl_from_messages(msgs: list) -> list[dict]:
    """仅对尚未产生 ToolMessage 的 HITL 工具调用构造待确认项。"""
    if not messages_have_pending_tool_calls(msgs):
        return []
    answered = _answered_tool_call_ids(msgs)
    for m in reversed(msgs):
        if not isinstance(m, AIMessage) or not m.tool_calls:
            continue
        out: list[dict] = []
        for tc in m.tool_calls:
            tid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if tid and str(tid) in answered:
                continue
            name = tc.get("name") if isinstance(tc, dict) else getattr(tc, "name", "")
            if name not in HITL_TOOL_NAMES:
                continue
            args = tc.get("args") if isinstance(tc, dict) else getattr(tc, "args", None)
            if not isinstance(args, dict):
                args = {}
            payload = hitl_payload_for_tool(name, args)
            out.append(
                {
                    "tool": payload["tool"],
                    "label": payload["label"],
                    "args": payload["args"],
                    "summary": payload["summary"],
                }
            )
        if out:
            return out
    return []


async def read_agent_snapshot(agent, config: dict) -> tuple[dict[str, Any], list, list[dict] | None]:
    """优先 aget_state 读取 checkpoint（含 __interrupt__）。"""
    try:
        snap = await asyncio.wait_for(agent.aget_state(config), timeout=15.0)
    except Exception as e:
        logger.warning("aget_state 失败/超时: %s", e)
        return {}, [], None
    values: dict[str, Any] = dict(snap.values) if snap and snap.values else {}
    msgs = list(values.get("messages") or [])
    pending = messages_have_pending_tool_calls(msgs)

    hitl = normalize_interrupts(values.get("__interrupt__"))
    if hitl and not pending:
        # 工具已执行完毕但 checkpoint 仍残留 interrupt 元数据时，不应再次要求确认
        logger.debug("忽略已完成 run 的残留 __interrupt__")
        hitl = []
    elif not hitl and pending:
        hitl = synthetic_hitl_from_messages(msgs)
    return values, msgs, hitl or None


async def finalize_agent_run(
    agent,
    config: dict,
    *,
    collected: dict[str, Any] | None = None,
) -> tuple[str | None, list, list[dict] | None]:
    """返回 (reply|None, messages, hitl_pending|None)。禁止在 tool_calls 未完成时假完成。"""
    _values, msgs, hitl = await read_agent_snapshot(agent, config)
    if not msgs and collected:
        msgs = list(collected.get("messages") or [])
        if not hitl and messages_have_pending_tool_calls(msgs):
            hitl = synthetic_hitl_from_messages(msgs) or None
    if hitl:
        return None, msgs, hitl
    if messages_have_pending_tool_calls(msgs):
        logger.warning("Agent 存在未完成 tool_calls，不返回假完成回复")
        return None, msgs, None
    return last_assistant_text(msgs), msgs, None
