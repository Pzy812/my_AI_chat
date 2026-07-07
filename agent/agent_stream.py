"""LangGraph Agent 流式事件：astream_events → Thought / Action / Observation。"""
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import AsyncIterator
from typing import Any

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from mcp import ClientSession

from app_mcp.mcp_http_client import (
    is_mcp_transport_error,
    open_ephemeral_mcp_session,
    reconnect_mcp_session,
    recover_mcp_server_async,
)

from agent.harness import (
    build_stuck_give_up_nudge,
    format_reanchor_summary,
    get_abandoned_tools,
    is_phase_gate_abandon_message,
    merge_task_state,
    persist_task_harness_meta,
    prepare_agent_invoke,
    sync_run_context_from_values,
    task_harness_event_payload,
)
from agent.planner import format_plan_for_display
from config.app_config import logger
from agent.agent_service import (
    CHAT_OFFLINE_PROMPT_SUFFIX,
    chat_agent_prompt_with_rag,
    langchain_tools_from_mcp_session,
    log_llm_system_prompt,
    prompt_debug_payload,
    _create_agent,
)
from agent.agent_state import finalize_agent_run
from agent.task_checklist import MAX_TASK_CONTINUATIONS, should_continue_task
from chat.chat_helpers import (
    build_tool_debug_from_messages,
    extract_mcp_attachments_from_messages,
    last_assistant_text,
)
from agent.hitl_tools import normalize_interrupts
from llm.model_config import make_llm_from_config
import app_mcp.mcp_lifecycle as mcp_lifecycle

_STREAM_CONTENT_CAP = 12_000
# 连续收到「工具已放弃」类 observation 后强制停止重试
_MAX_CONSECUTIVE_PHASE_ABANDON = 2


_STREAM_THINK_TAG_RE = re.compile(
    r"</?(?:think|redacted_reasoning)[^>]*>",
    re.IGNORECASE,
)


def _sanitize_stream_text(text: str) -> str:
    s = _STREAM_THINK_TAG_RE.sub("", text or "")
    return s.strip()


def _clip(text: str, cap: int = _STREAM_CONTENT_CAP) -> str:
    s = _sanitize_stream_text(text)
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
        text = _sanitize_stream_text(_message_content_text(getattr(chunk, "content", None)))
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
    file_count: int = 0,
    rag_mode: str | None = None,
) -> tuple[dict, Any]:
    config, input_data, _ = await prepare_agent_invoke(
        session_id=session_id,
        lc_messages=lc_messages,
        resume_action=resume_action,
        fresh_thread=fresh_thread,
        file_count=file_count,
        rag_mode=rag_mode,
        stream=True,
    )
    return config, input_data


def _state_from_chain_event(event: dict[str, Any]) -> dict[str, Any] | None:
    """从 on_chain_end 提取 LangGraph 状态，避免流结束后 aget_state 卡住。"""
    if event.get("event") != "on_chain_end":
        return None
    output = (event.get("data") or {}).get("output")
    if isinstance(output, dict) and (
        "messages" in output
        or "__interrupt__" in output
        or "user_goal" in output
        or "harness_enabled" in output
    ):
        return output
    return None


def _harness_state_signature(payload: dict[str, Any]) -> str:
    return json.dumps(
        {
            "user_goal": payload.get("user_goal"),
            "plan": payload.get("plan"),
            "plan_index": payload.get("plan_index"),
            "task_phase": payload.get("task_phase"),
            "harness_enabled": payload.get("harness_enabled"),
            "completed_steps": payload.get("completed_steps"),
            "step_checklist": payload.get("step_checklist"),
        },
        ensure_ascii=False,
        sort_keys=True,
    )


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
    *,
    pre_events: list[dict[str, Any]] | None = None,
    resume_action: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    from observability.langsmith_session import (
        agent_turn_trace,
        extract_last_user_text,
        finish_agent_turn,
    )

    for ev in pre_events or []:
        yield ev
    task_baseline: dict[str, Any] = {}
    if isinstance(input_data, dict):
        task_baseline = {
            k: input_data.get(k)
            for k in (
                "user_goal",
                "plan",
                "plan_index",
                "task_phase",
                "harness_enabled",
                "completed_steps",
                "step_checklist",
                "task_status",
            )
            if k in input_data
        }
    current_input = input_data
    continuations = 0
    reply: str | None = None
    msgs: list = []
    hitl: list[dict] | None = None
    last_state: dict[str, Any] = dict(task_baseline)
    stuck_recovery_used = False
    root_run_id: str | None = None
    thread_id = str((config.get("configurable") or {}).get("thread_id") or "default")
    _root_chain_names = frozenset({"LangGraph", "agent", "RunnableSequence"})
    _configured_run_name = (config.get("run_name") or "").strip()
    user_preview = ""
    if isinstance(input_data, dict):
        user_preview = extract_last_user_text(input_data.get("messages"))
    is_resume = resume_action is not None

    with agent_turn_trace(
        thread_id,
        user_input=user_preview,
        is_resume=is_resume,
    ) as turn_run:
        while True:
            last_state = dict(task_baseline) if continuations == 0 else last_state
            last_harness_sig: str | None = None
            agent_started = False
            stuck_give_up_nudge: str | None = None
            consecutive_phase_abandon = 0
            async for event in agent.astream_events(current_input, config=config, version="v2"):
                if event.get("event") == "on_chain_start":
                    ev_run_id = event.get("run_id")
                    ev_name = (event.get("name") or "").strip()
                    if ev_run_id:
                        rid_s = str(ev_run_id)
                        if (
                            ev_name in _root_chain_names
                            or ev_name.startswith("turn-")
                            or ev_name.startswith("chat:")
                            or ev_name.startswith("agent:")
                            or (_configured_run_name and ev_name == _configured_run_name)
                        ):
                            root_run_id = rid_s
                        elif root_run_id is None:
                            parent_ids = event.get("parent_ids") or []
                            if not parent_ids:
                                root_run_id = rid_s
                if event.get("event") == "on_tool_end":
                    output = (event.get("data") or {}).get("output")
                    if hasattr(output, "content"):
                        obs_text = _message_content_text(output.content)
                    else:
                        obs_text = _message_content_text(output)
                    if is_phase_gate_abandon_message(obs_text):
                        consecutive_phase_abandon += 1
                    else:
                        consecutive_phase_abandon = 0
                    if consecutive_phase_abandon >= _MAX_CONSECUTIVE_PHASE_ABANDON:
                        abandoned = get_abandoned_tools(thread_id)
                        stuck_give_up_nudge = build_stuck_give_up_nudge(abandoned)
                        yield {
                            "type": "step",
                            "phase": "status",
                            "content": (
                                "检测到工具因阶段限制反复失败，停止重试，"
                                "改为直接向用户说明…"
                            ),
                        }
                        break
                chunk_state = _state_from_chain_event(event)
                if chunk_state:
                    merged = merge_task_state(last_state, chunk_state)
                    last_state = merged
                    if merged.get("user_goal") or merged.get("harness_enabled"):
                        harness_ev = task_harness_event_payload(merged)
                        sig = _harness_state_signature(harness_ev)
                        if sig != last_harness_sig:
                            last_harness_sig = sig
                            persist_task_harness_meta(
                                str((config.get("configurable") or {}).get("thread_id") or "default"),
                                merged,
                            )
                            yield harness_ev
                for step in _parse_astream_event(event):
                    if step.get("phase") == "thought":
                        step = {**step, "content": _sanitize_stream_text(step.get("content") or "")}
                        if not step["content"]:
                            continue
                    if step.get("phase") == "status" and "Agent 开始推理" in (step.get("content") or ""):
                        if agent_started:
                            continue
                        agent_started = True
                    yield step
            if stuck_give_up_nudge:
                if stuck_recovery_used:
                    break
                stuck_recovery_used = True
                current_input = {"messages": [HumanMessage(content=stuck_give_up_nudge)]}
                continue
            reply, msgs, hitl = await _finalize_agent(
                agent, config, collected=last_state or None
            )
            if hitl:
                break
            nudge = should_continue_task(last_state, msgs, reply)
            if not nudge or continuations >= MAX_TASK_CONTINUATIONS:
                break
            continuations += 1
            if last_state.get("harness_enabled"):
                plan = list(last_state.get("plan") or [])
                plan_index = int(last_state.get("plan_index") or 0)
                from agent.harness import compute_task_phase

                last_state = {
                    **last_state,
                    "task_phase": compute_task_phase(
                        plan, plan_index, harness_enabled=True
                    ),
                }
                from agent.harness import sync_run_context_from_values

                thread_id = str((config.get("configurable") or {}).get("thread_id") or "default")
                sync_run_context_from_values(thread_id, last_state)
            plan = list(last_state.get("plan") or [])
            plan_index = int(last_state.get("plan_index") or 0)
            yield {
                "type": "step",
                "phase": "status",
                "content": (
                    f"检测到任务尚未全部完成（{min(plan_index + 1, len(plan))}/{len(plan)}），"
                    "Agent 自动续跑…"
                    if plan
                    else "检测到外发/交付步骤尚未执行，Agent 继续运行…"
                ),
            }
            current_input = {"messages": [HumanMessage(content=nudge)]}

        if last_state.get("user_goal") or last_state.get("harness_enabled"):
            persist_task_harness_meta(
                str((config.get("configurable") or {}).get("thread_id") or "default"),
                last_state,
            )
            yield task_harness_event_payload(last_state)
        finish_agent_turn(
            thread_id,
            turn_run=turn_run,
            reply=reply,
            hitl_pending=bool(hitl),
        )
        from observability.langsmith_trace import finalize_trace_for_session

        langsmith = finalize_trace_for_session(
            thread_id,
            event_root_run_id=root_run_id,
        )
        yield {
            "type": "_agent_result",
            "reply": reply,
            "messages": msgs,
            "hitl": hitl,
            "langsmith": langsmith,
        }


async def _iter_llm_only_events(
    lc_messages: list,
    rag_context: str | None,
    *,
    session_id: str,
    log_prompt: bool,
    llm_config: dict | None = None,
) -> AsyncIterator[dict[str, Any]]:
    yield {
        "type": "step",
        "phase": "status",
        "content": "MCP 不可用，使用纯 LLM 模式（无工具调用）…",
    }
    llm = make_llm_from_config(llm_config)
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


async def _ensure_mcp_ready() -> bool:
    return await mcp_lifecycle.ensure_mcp_server_started_async()


def _plan_pre_events(input_data: dict[str, Any]) -> list[dict[str, Any]]:
    pre_events: list[dict[str, Any]] = [task_harness_event_payload(input_data)]
    if input_data.get("harness_enabled"):
        plan = input_data.get("plan") or []
        if plan:
            pre_events.append(
                {
                    "type": "step",
                    "phase": "status",
                    "content": format_plan_for_display(
                        plan, plan_index=int(input_data.get("plan_index") or 0)
                    ),
                }
            )
    return pre_events


async def _hitl_resume_pre_events(
    agent,
    config: dict,
    session_id: str,
    action: str,
) -> list[dict[str, Any]]:
    pre_events: list[dict[str, Any]] = [
        {
            "type": "step",
            "phase": "status",
            "content": "已" + ("确认" if action == "approve" else "取消") + "，继续 Agent…",
        }
    ]
    try:
        snap = await agent.aget_state(config)
        values = dict(snap.values) if snap and snap.values else {}
        if values.get("harness_enabled"):
            sync_run_context_from_values(session_id, values)
        pre_events.append(task_harness_event_payload(values))
        if values.get("harness_enabled"):
            pre_events.append(
                {
                    "type": "step",
                    "phase": "status",
                    "content": format_reanchor_summary(values),
                }
            )
    except Exception:
        pass
    return pre_events


async def _stream_react_on_session(
    session: ClientSession,
    *,
    session_id: str,
    lc_messages: list | None,
    resume_action: str | None,
    fresh_thread: bool,
    file_count: int,
    rag_context: str | None,
    rag_mode: str | None,
    llm_config: dict | None,
    hitl_enabled: bool,
) -> AsyncIterator[dict[str, Any]]:
    llm = make_llm_from_config(llm_config)
    tools = await langchain_tools_from_mcp_session(session, hitl_enabled=hitl_enabled)
    agent = await _create_agent(llm, tools, rag_context)
    config, input_data = await _prepare_invoke(
        session_id=session_id,
        lc_messages=lc_messages,
        resume_action=resume_action,
        fresh_thread=fresh_thread,
        file_count=file_count,
        rag_mode=rag_mode,
    )
    if resume_action is not None:
        pre_events = await _hitl_resume_pre_events(agent, config, session_id, resume_action)
    elif isinstance(input_data, dict):
        pre_events = _plan_pre_events(input_data)
    else:
        pre_events = []
    async for ev in _iter_react_agent_events(
        agent, input_data, config, pre_events=pre_events, resume_action=resume_action
    ):
        yield ev


async def _stream_react_with_mcp_retry(
    *,
    session_id: str,
    lc_messages: list | None,
    resume_action: str | None,
    fresh_thread: bool,
    file_count: int,
    rag_context: str | None,
    rag_mode: str | None,
    llm_config: dict | None,
    hitl_enabled: bool,
) -> AsyncIterator[dict[str, Any]]:
    """流式 Agent；MCP SSE 异常时重连客户端，必要时重启 MCP 服务。"""
    last_err: BaseException | None = None
    max_attempts = 3
    for attempt in range(max_attempts):
        try:
            async with open_ephemeral_mcp_session() as session:
                async for ev in _stream_react_on_session(
                    session,
                    session_id=session_id,
                    lc_messages=lc_messages,
                    resume_action=resume_action,
                    fresh_thread=fresh_thread,
                    file_count=file_count,
                    rag_context=rag_context,
                    rag_mode=rag_mode,
                    llm_config=llm_config,
                    hitl_enabled=hitl_enabled,
                ):
                    yield ev
            return
        except BaseException as e:
            last_err = e
            if attempt + 1 >= max_attempts or not is_mcp_transport_error(e):
                raise
            logger.warning(
                "Agent 流式 MCP 失败，准备重试 (%s/%s): %s",
                attempt + 1,
                max_attempts,
                e,
            )
            yield {
                "type": "step",
                "phase": "status",
                "content": (
                    "MCP 连接异常，正在重试…"
                    if attempt == 0
                    else "MCP 仍不可用，正在重启 MCP 服务并重试…"
                ),
            }
            if attempt >= 1 and not await recover_mcp_server_async():
                raise RuntimeError(
                    "MCP 重启失败，请手动运行 python mcp_server.py"
                ) from e
    if last_err is not None:
        raise last_err


async def stream_agent_with_history(
    lc_messages: list,
    rag_context: str | None = None,
    *,
    session_id: str = "",
    log_prompt: bool = False,
    file_count: int = 0,
    llm_config: dict | None = None,
    hitl_enabled: bool = True,
    rag_mode: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """流式 ReAct；末尾 yield _agent_result（由路由转为 done / hitl_pending）。"""
    from core.app_utils import format_error

    if not await _ensure_mcp_ready():
        async for ev in _iter_llm_only_events(
            lc_messages,
            rag_context,
            session_id=session_id,
            log_prompt=log_prompt,
            llm_config=llm_config,
        ):
            yield ev
        return

    react_gen = _stream_react_with_mcp_retry(
        session_id=session_id,
        lc_messages=lc_messages,
        resume_action=None,
        fresh_thread=True,
        file_count=file_count,
        rag_context=rag_context,
        rag_mode=rag_mode,
        llm_config=llm_config,
        hitl_enabled=hitl_enabled,
    )
    try:
        if log_prompt:
            log_llm_system_prompt(
                "react_agent_stream",
                chat_agent_prompt_with_rag(rag_context),
                session_id=session_id,
                rag_context=rag_context,
            )
        async for ev in react_gen:
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
            lc_messages,
            rag_context,
            session_id=session_id,
            log_prompt=log_prompt,
            llm_config=llm_config,
        ):
            yield ev
    finally:
        try:
            await react_gen.aclose()
        except BaseException:
            pass


async def stream_agent_hitl_resume(
    session_id: str,
    action: str,
    *,
    rag_context: str | None = None,
    log_prompt: bool = False,
    llm_config: dict | None = None,
    hitl_enabled: bool = True,
    rag_mode: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    if not await _ensure_mcp_ready():
        raise RuntimeError("MCP 未就绪，无法恢复 HITL 会话")
    from agent.agent_service import effective_hitl_enabled

    if not effective_hitl_enabled(hitl_enabled):
        raise RuntimeError("Human-in-the-Loop 未启用或已在设置中关闭")

    if log_prompt:
        log_llm_system_prompt(
            "react_agent_hitl_resume_stream",
            chat_agent_prompt_with_rag(rag_context),
            session_id=session_id,
            rag_context=rag_context,
        )
    try:
        async for ev in _stream_react_with_mcp_retry(
            session_id=session_id,
            lc_messages=None,
            resume_action=action,
            fresh_thread=False,
            file_count=0,
            rag_context=rag_context,
            rag_mode=rag_mode,
            llm_config=llm_config,
            hitl_enabled=hitl_enabled,
        ):
            yield ev
    except BaseException as e:
        from core.app_utils import format_error

        logger.error("HITL 流式恢复失败: %s", format_error(e))
        yield {
            "type": "_agent_result",
            "reply": None,
            "messages": [],
            "hitl": None,
            "_error": f"HITL 恢复失败：{format_error(e)}（请刷新后重试，或重启 app.py）",
        }


def build_stream_done_payload(
    *,
    session_id: str,
    reply: str,
    msgs: list,
    rag_context: str,
    rag_mode: str,
    include_tool_debug: bool,
    agent_system_prompt: str,
    langsmith_trace: dict | None = None,
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
    if langsmith_trace:
        out["langsmith"] = langsmith_trace
    return out


def build_stream_hitl_payload(
    *,
    session_id: str,
    hitl_pending: list[dict],
    rag_mode: str,
    langsmith_trace: dict | None = None,
) -> dict[str, Any]:
    pending = hitl_pending[0] if hitl_pending else {}
    from agent.agent_service import hitl_available

    out: dict[str, Any] = {
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
    if langsmith_trace:
        out["langsmith"] = langsmith_trace
    return out
