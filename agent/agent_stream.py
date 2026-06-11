"""LangGraph Agent 流式事件：astream_events → Thought / Action / Observation。"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, SystemMessage
from langgraph.types import Command
from mcp.client.streamable_http import streamable_http_client
from mcp import ClientSession

from agent.agent_checkpointer import get_checkpointer, reset_agent_thread
from agent.agent_service import (
    CHAT_OFFLINE_PROMPT_SUFFIX,
    chat_agent_prompt_with_rag,
    langchain_tools_from_mcp_session,
    log_llm_system_prompt,
    prompt_debug_payload,
    _create_agent,
)
from agent.agent_state import finalize_agent_run
from config.app_config import MCP_URL, logger
from chat.chat_helpers import (
    build_tool_debug_from_messages,
    extract_mcp_attachments_from_messages,
    last_assistant_text,
)
from agent.hitl_tools import normalize_interrupts
from llm.llm_zhipu import make_chat_llm
import app_mcp.mcp_lifecycle as mcp_lifecycle

_STREAM_CONTENT_CAP = 12_000


def _clip(text: str, cap: int = _STREAM_CONTENT_CAP) -> str:
    s = (text or "").strip()
    if len(s) <= cap:
        return s
    return s[:cap] + f"\n…(已截断，共 {len(s)} 字符)"


def _message_content_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(parts)
    return str(content)


def _tool_call_entries(message: AIMessage) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for tc in message.tool_calls or []:
        if isinstance(tc, dict):
            out.append({"name": tc.get("name") or "", "args": tc.get("args") or {}})
        else:
            out.append(
                {
                    "name": getattr(tc, "name", "") or "",
                    "args": getattr(tc, "args", None) or {},
                }
            )
    return out


def _parse_astream_event(event: dict[str, Any]) -> list[dict[str, Any]]:
    """将 LangGraph astream_events(v2) 单条事件转为 0~N 个前端 step。"""
    kind = event.get("event") or ""
    data = event.get("data") or {}

    if kind == "on_chat_model_stream":
        chunk = data.get("chunk")
        if chunk is None:
            return []
        text = _message_content_text(getattr(chunk, "content", None))
        if not text:
            return []
        return [{"type": "step_delta", "phase": "thought", "content": text}]

    if kind == "on_chat_model_end":
        output = data.get("output")
        if not isinstance(output, AIMessage):
            return []
        steps: list[dict[str, Any]] = []
        text = _message_content_text(output.content)
        if text.strip():
            steps.append(
                {
                    "type": "step",
                    "phase": "thought",
                    "content": _clip(text),
                }
            )
        for tc in _tool_call_entries(output):
            name = tc["name"]
            args = tc["args"]
            try:
                args_preview = json.dumps(args, ensure_ascii=False)
            except Exception:
                args_preview = str(args)
            steps.append(
                {
                    "type": "step",
                    "phase": "action",
                    "tool": name,
                    "args": args,
                    "content": _clip(f"调用工具 {name}({args_preview})"),
                }
            )
        return steps

    if kind == "on_tool_start":
        name = event.get("name") or data.get("name") or ""
        inputs = data.get("input") or {}
        try:
            args_preview = json.dumps(inputs, ensure_ascii=False)
        except Exception:
            args_preview = str(inputs)
        return [
            {
                "type": "step",
                "phase": "action",
                "tool": name,
                "args": inputs if isinstance(inputs, dict) else {"input": inputs},
                "content": _clip(f"执行 {name}({args_preview})"),
            }
        ]

    if kind == "on_tool_end":
        name = event.get("name") or ""
        output = data.get("output")
        if hasattr(output, "content"):
            content = _message_content_text(output.content)
        else:
            content = _message_content_text(output)
        return [
            {
                "type": "step",
                "phase": "observation",
                "tool": name,
                "content": _clip(content or "(空)"),
            }
        ]

    if kind == "on_chain_start" and event.get("name") == "agent":
        return [{"type": "step", "phase": "status", "content": "Agent 开始推理…"}]

    return []


async def _prepare_invoke(
    *,
    session_id: str,
    lc_messages: list | None,
    resume_action: str | None,
    fresh_thread: bool,
) -> tuple[dict, Any]:
    checkpointer = get_checkpointer()
    config = {"configurable": {"thread_id": session_id}} if checkpointer else {}
    if checkpointer and fresh_thread and resume_action is None:
        await reset_agent_thread(session_id)
    if resume_action is not None:
        return config, Command(resume={"action": resume_action})
    return config, {"messages": lc_messages or []}


def _state_from_chain_event(event: dict[str, Any]) -> dict[str, Any] | None:
    """从 on_chain_end 提取 LangGraph 状态，避免流结束后 aget_state 卡住。"""
    if event.get("event") != "on_chain_end":
        return None
    output = (event.get("data") or {}).get("output")
    if isinstance(output, dict) and (
        "messages" in output or "__interrupt__" in output
    ):
        return output
    return None


async def _finalize_agent(
    agent,
    config: dict,
    *,
    collected: dict[str, Any] | None = None,
) -> tuple[str | None, list, list[dict] | None]:
    return await finalize_agent_run(agent, config, collected=collected)


async def _iter_react_agent_events(
    agent,
    input_data: Any,
    config: dict,
) -> AsyncIterator[dict[str, Any]]:
    last_state: dict[str, Any] = {}
    agent_started = False
    async for event in agent.astream_events(input_data, config=config, version="v2"):
        for step in _parse_astream_event(event):
            if step.get("phase") == "status" and "Agent 开始推理" in (step.get("content") or ""):
                if agent_started:
                    continue
                agent_started = True
            yield step
        chunk_state = _state_from_chain_event(event)
        if chunk_state:
            last_state = chunk_state
    reply, msgs, hitl = await _finalize_agent(
        agent, config, collected=last_state or None
    )
    yield {"type": "_agent_result", "reply": reply, "messages": msgs, "hitl": hitl}


async def _iter_llm_only_events(
    lc_messages: list,
    rag_context: str | None,
    *,
    session_id: str,
    log_prompt: bool,
) -> AsyncIterator[dict[str, Any]]:
    yield {
        "type": "step",
        "phase": "status",
        "content": "MCP 不可用，使用纯 LLM 模式（无工具调用）…",
    }
    llm = make_chat_llm()
    prompt = chat_agent_prompt_with_rag(rag_context) + CHAT_OFFLINE_PROMPT_SUFFIX
    if log_prompt:
        log_llm_system_prompt(
            "llm_only_offline_stream",
            prompt,
            session_id=session_id,
            rag_context=rag_context,
        )
    msgs = list(lc_messages)
    full_parts: list[str] = []
    async for chunk in llm.astream([SystemMessage(content=prompt)] + msgs):
        text = _message_content_text(getattr(chunk, "content", None))
        if text:
            full_parts.append(text)
            yield {"type": "step_delta", "phase": "thought", "content": text}
    reply = "".join(full_parts).strip()
    if reply:
        yield {"type": "step", "phase": "thought", "content": _clip(reply)}
    yield {
        "type": "_agent_result",
        "reply": reply or None,
        "messages": msgs + [AIMessage(content=reply)] if reply else msgs,
        "hitl": None,
    }


async def stream_agent_with_history(
    lc_messages: list,
    rag_context: str | None = None,
    *,
    session_id: str = "",
    log_prompt: bool = False,
) -> AsyncIterator[dict[str, Any]]:
    """流式 ReAct；末尾 yield _agent_result（由路由转为 done / hitl_pending）。"""
    from core.app_utils import format_error

    if not mcp_lifecycle.ensure_mcp_server_started():
        async for ev in _iter_llm_only_events(
            lc_messages, rag_context, session_id=session_id, log_prompt=log_prompt
        ):
            yield ev
        return

    try:
        if log_prompt:
            log_llm_system_prompt(
                "react_agent_stream",
                chat_agent_prompt_with_rag(rag_context),
                session_id=session_id,
                rag_context=rag_context,
            )
        llm = make_chat_llm()
        async with streamable_http_client(MCP_URL) as (r, w, _):
            async with ClientSession(r, w) as session:
                tools = await langchain_tools_from_mcp_session(session)
                agent = await _create_agent(llm, tools, rag_context)
                config, input_data = await _prepare_invoke(
                    session_id=session_id,
                    lc_messages=lc_messages,
                    resume_action=None,
                    fresh_thread=True,
                )
                async for ev in _iter_react_agent_events(agent, input_data, config):
                    yield ev
    except BaseException as e:
        logger.warning(
            "Agent 流式调用失败，降级纯 LLM：%s",
            format_error(e),
        )
        yield {
            "type": "step",
            "phase": "status",
            "content": f"Agent+MCP 失败，已降级：{format_error(e)}",
        }
        async for ev in _iter_llm_only_events(
            lc_messages, rag_context, session_id=session_id, log_prompt=log_prompt
        ):
            yield ev


async def stream_agent_hitl_resume(
    session_id: str,
    action: str,
    *,
    rag_context: str | None = None,
    log_prompt: bool = False,
) -> AsyncIterator[dict[str, Any]]:
    if not mcp_lifecycle.ensure_mcp_server_started():
        raise RuntimeError("MCP 未就绪，无法恢复 HITL 会话")
    from agent.agent_service import hitl_available

    if not hitl_available():
        raise RuntimeError("HITL 未启用或未配置 Postgres Checkpointer")

    if log_prompt:
        log_llm_system_prompt(
            "react_agent_hitl_resume_stream",
            chat_agent_prompt_with_rag(rag_context),
            session_id=session_id,
            rag_context=rag_context,
        )
    llm = make_chat_llm()
    async with streamable_http_client(MCP_URL) as (r, w, _):
        async with ClientSession(r, w) as session:
            tools = await langchain_tools_from_mcp_session(session)
            agent = await _create_agent(llm, tools, rag_context)
            config, input_data = await _prepare_invoke(
                session_id=session_id,
                lc_messages=None,
                resume_action=action,
                fresh_thread=False,
            )
            yield {
                "type": "step",
                "phase": "status",
                "content": "已" + ("确认" if action == "approve" else "取消") + "，继续 Agent…",
            }
            async for ev in _iter_react_agent_events(agent, input_data, config):
                yield ev


def build_stream_done_payload(
    *,
    session_id: str,
    reply: str,
    msgs: list,
    rag_context: str,
    rag_mode: str,
    include_tool_debug: bool,
    agent_system_prompt: str,
) -> dict[str, Any]:
    attachments = extract_mcp_attachments_from_messages(msgs)
    out: dict[str, Any] = {
        "type": "done",
        "code": 0,
        "status": "completed",
        "msg": reply,
        "session_id": session_id,
    }
    if attachments:
        out["mcp_attachments"] = attachments
    if rag_context:
        out["rag_used"] = True
        out["rag_mode"] = rag_mode
        if rag_mode == "graphrag":
            out["graphrag_used"] = True
    if include_tool_debug:
        out["tool_debug"] = build_tool_debug_from_messages(msgs)
        out["prompt_debug"] = prompt_debug_payload(agent_system_prompt, rag_context)
    return out


def build_stream_hitl_payload(
    *,
    session_id: str,
    hitl_pending: list[dict],
    rag_mode: str,
) -> dict[str, Any]:
    pending = hitl_pending[0] if hitl_pending else {}
    from agent.agent_service import hitl_available

    return {
        "type": "hitl_pending",
        "code": 0,
        "status": "hitl_pending",
        "session_id": session_id,
        "rag_mode": rag_mode,
        "hitl": {
            "pending": hitl_pending,
            "tool": pending.get("tool"),
            "label": pending.get("label"),
            "summary": pending.get("summary"),
            "args": pending.get("args"),
            "hitl_enabled": hitl_available(),
        },
        "msg": "等待您确认是否执行敏感操作",
    }
