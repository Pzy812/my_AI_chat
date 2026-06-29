"""Task Harness：上下文裁剪、重锚定、工具阶段 gate、pre/post hooks。"""
from __future__ import annotations

import logging
from typing import Any

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableConfig
from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import Command

from agent.planner import build_task_plan, needs_task_harness, format_plan_for_display
from agent.task_checklist import (
    build_step_checklist,
    extract_user_goal,
    resolve_user_goal,
)
from agent.task_state import (
    PHASE_LABELS,
    TaskPhase,
    allowed_tools_for_phase,
    default_task_fields,
    infer_phase_from_step,
)
from config.app_config import (
    AGENT_LLM_CONTEXT_MESSAGES,
    AGENT_PHASE_GATE_MAX_RETRIES,
    AGENT_REANCHOR_EVERY_N_TOOLS,
)

logger = logging.getLogger("ai_chat.harness")

# thread_id → 最新 task 字段（供工具 phase gate 读取）
_run_task_context: dict[str, dict[str, Any]] = {}
# thread_id → tool_name → 阶段 gate 拒绝次数；thread_id → 已放弃的工具名
_phase_gate_attempts: dict[str, dict[str, int]] = {}
_abandoned_tools: dict[str, set[str]] = {}

PHASE_GATE_MARKER = "⛔ 工具"
PHASE_GATE_ABANDON_MARKER = "已放弃"

TASK_DISCIPLINE_PROMPT = (
    "\n\n【复杂任务执行纪律】\n"
    "1. 严格按「执行计划」推进，每次工具调用前说明当前在做哪一步。\n"
    "2. 当前阶段仅可使用系统提示中列出的工具；阶段未到时不要调用外发/导出类工具。\n"
    "3. 工具返回后先更新进度，再决定下一步；若发现偏离用户原始目标，停止并说明。\n"
    "4. 未完成全部必要步骤前，不要给出最终结论。\n"
    "5. Gather 阶段若有多个独立搜索主题，优先使用 web_search_batch(queries=[...]) 并行检索。"
)


def count_tool_rounds(messages: list[BaseMessage]) -> int:
    """统计已完成 tool call 的轮数（AIMessage+ToolMessage 配对）。"""
    answered: set[str] = set()
    for m in messages:
        if isinstance(m, ToolMessage) and m.tool_call_id:
            answered.add(str(m.tool_call_id))
    rounds = 0
    for m in messages:
        if not isinstance(m, AIMessage) or not m.tool_calls:
            continue
        ids = []
        for tc in m.tool_calls:
            tid = tc.get("id") if isinstance(tc, dict) else getattr(tc, "id", None)
            if tid:
                ids.append(str(tid))
        if ids and all(i in answered for i in ids):
            rounds += 1
    return rounds


def _thread_id_from_config(config: RunnableConfig | None) -> str:
    if not config:
        return "default"
    return str((config.get("configurable") or {}).get("thread_id") or "default")


def _sync_run_context(thread_id: str, state: dict[str, Any]) -> None:
    _run_task_context[thread_id] = {
        "harness_enabled": bool(state.get("harness_enabled")),
        "task_phase": state.get("task_phase") or "gather",
        "user_goal": state.get("user_goal") or "",
        "plan": list(state.get("plan") or []),
        "plan_index": int(state.get("plan_index") or 0),
    }


def _state_get(state: Any, key: str, default=None):
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def compute_plan_index(plan: list[str], messages: list[BaseMessage]) -> int:
    if not plan:
        return 0
    rounds = count_tool_rounds(messages)
    return min(rounds, len(plan) - 1)


def compute_task_phase(
    plan: list[str],
    plan_index: int,
    *,
    harness_enabled: bool,
) -> TaskPhase:
    if not harness_enabled:
        return "deliver"
    if plan and 0 <= plan_index < len(plan):
        return infer_phase_from_step(plan[plan_index])
    if plan_index > 0:
        return infer_phase_from_step(plan[-1])
    return "gather"


def build_reanchor_text(state: dict[str, Any]) -> str:
    goal = (_state_get(state, "user_goal") or "").strip()
    plan = list(_state_get(state, "plan") or [])
    plan_index = int(_state_get(state, "plan_index") or 0)
    phase = _state_get(state, "task_phase") or "gather"
    completed = list(_state_get(state, "completed_steps") or [])
    checklist = list(_state_get(state, "step_checklist") or [])
    allowed = allowed_tools_for_phase(
        phase, harness_enabled=bool(_state_get(state, "harness_enabled"))
    )

    lines = ["【任务续跑上下文 — 每步推理前必读】"]
    if goal:
        lines.append(f"原始目标：{goal[:2000]}")
    if plan:
        lines.append(format_plan_for_display(plan, plan_index=plan_index))
        if plan_index < len(plan):
            lines.append(f"当前应执行：第 {plan_index + 1} 步 — {plan[plan_index]}")
    else:
        lines.append("（本轮为简单任务，无分步计划）")
    lines.append(f"当前阶段：{PHASE_LABELS.get(phase, phase)}")
    if checklist:
        lines.append("子任务进度：")
        for item in checklist:
            mark = "✓" if item.get("done") else ("→" if item.get("current") else "✗")
            lines.append(f"  {mark} {item.get('index', 0) + 1}. {item.get('step', '')}")
    elif completed:
        lines.append("已完成：" + "；".join(completed[-5:]))
    if _state_get(state, "harness_enabled"):
        lines.append(
            "本阶段允许工具："
            + ", ".join(sorted(allowed))
        )
    lines.append("请先确认当前步骤，再决定是否调用工具；不要偏离原始目标。")
    return "\n".join(lines)


def trim_messages_for_llm(
    messages: list[BaseMessage],
    *,
    keep_recent: int | None = None,
) -> list[BaseMessage]:
    """保留首条用户消息 + 最近若干条，降低 ToolMessage 淹没目标的问题。"""
    cap = keep_recent if keep_recent is not None else AGENT_LLM_CONTEXT_MESSAGES
    if len(messages) <= cap + 1:
        return list(messages)

    first_user: BaseMessage | None = None
    for m in messages:
        if isinstance(m, HumanMessage):
            first_user = m
            break

    tail = list(messages[-cap:])
    if first_user is not None and first_user not in tail:
        return [first_user, *tail]
    return tail


async def build_initial_agent_state(
    lc_messages: list[BaseMessage],
    *,
    session_id: str = "",
    file_count: int = 0,
) -> dict[str, Any]:
    """构造带 Task 字段的 Agent 初始 state。"""
    fields = default_task_fields()
    user_goal, is_continue = resolve_user_goal(lc_messages or [])
    fields["user_goal"] = user_goal
    harness = needs_task_harness(user_goal, file_count=file_count)
    fields["harness_enabled"] = harness

    if is_continue:
        fields["user_goal"] = user_goal
        import chat.chat_store as chat_store

        persisted = chat_store.get_task_harness_meta(session_id)
        if persisted.get("user_goal"):
            fields["user_goal"] = persisted["user_goal"]
        if persisted.get("plan"):
            fields["plan"] = list(persisted["plan"])
            fields["plan_index"] = int(persisted.get("plan_index") or 0)
            fields["task_phase"] = persisted.get("task_phase") or "gather"
            fields["completed_steps"] = list(persisted.get("completed_steps") or [])
            fields["step_checklist"] = list(persisted.get("step_checklist") or [])
            fields["task_status"] = persisted.get("task_status") or "executing"
            fields["harness_enabled"] = bool(persisted.get("harness_enabled", True))
            harness = fields["harness_enabled"]
        elif harness and user_goal:
            fields["task_status"] = "planning"
            plan = await build_task_plan(user_goal)
            fields["plan"] = plan
            fields["plan_index"] = 0
            fields["task_phase"] = compute_task_phase(plan, 0, harness_enabled=True)
            fields["step_checklist"] = build_step_checklist(plan, 0)
            fields["task_status"] = "executing"
    elif harness and user_goal:
        fields["task_status"] = "planning"
        plan = await build_task_plan(user_goal)
        fields["plan"] = plan
        fields["plan_index"] = 0
        fields["task_phase"] = compute_task_phase(plan, 0, harness_enabled=True)
        fields["step_checklist"] = build_step_checklist(plan, 0)
        fields["task_status"] = "executing"
    else:
        fields["task_phase"] = "deliver"

    state: dict[str, Any] = {
        "messages": lc_messages or [],
        **fields,
    }
    _sync_run_context(session_id or "default", state)
    if fields.get("harness_enabled") and fields.get("plan"):
        persist_task_harness_meta(session_id, state)
    return state


def format_reanchor_summary(state: dict[str, Any]) -> str:
    text = build_reanchor_text(state)
    return text.replace("【任务续跑上下文 — 每步推理前必读】", "【HITL 确认后继续 — 任务重锚定】", 1)


def make_pre_model_hook():
    async def pre_model_hook(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        messages = list(_state_get(state, "messages") or [])
        harness_enabled = bool(_state_get(state, "harness_enabled"))
        plan = list(_state_get(state, "plan") or [])
        plan_index = compute_plan_index(plan, messages)
        task_phase = compute_task_phase(
            plan, plan_index, harness_enabled=harness_enabled
        )
        from agent.task_continue import (
            deliver_tools_used,
            sync_deliver_completion_flags,
            user_goal_requires_deliver,
        )

        thread_id = _thread_id_from_config(config)
        sync_abandoned_tools_from_messages(thread_id, messages)
        sync_deliver_completion_flags(thread_id, messages)

        goal = (_state_get(state, "user_goal") or "").strip()
        if (
            harness_enabled
            and user_goal_requires_deliver(goal)
            and not deliver_tools_used(messages)
        ):
            if plan and plan_index >= len(plan) - 1:
                task_phase = "deliver"
            elif plan and infer_phase_from_step(plan[-1]) == "deliver" and plan_index >= len(plan) - 2:
                task_phase = "deliver"
            elif not plan and count_tool_rounds(messages) >= 1:
                task_phase = "deliver"
        tool_rounds = count_tool_rounds(messages)

        updated: dict[str, Any] = {
            "plan_index": plan_index,
            "task_phase": task_phase,
        }
        if plan:
            updated["completed_steps"] = plan[:plan_index]
            updated["step_checklist"] = build_step_checklist(plan, plan_index)
        if plan and plan_index >= len(plan):
            updated["task_status"] = "done"

        base = dict(state) if isinstance(state, dict) else {}
        merged = {**base, **updated}
        _sync_run_context(thread_id, merged)

        trimmed = trim_messages_for_llm(messages)
        reanchor_text = build_reanchor_text(merged)
        abandon_nudge = build_abandon_nudge(get_abandoned_tools(thread_id))
        if abandon_nudge:
            reanchor_text += abandon_nudge
        if deliver_tools_used(messages):
            reanchor_text += (
                "\n\n【外发已完成】邮件/微信/导出已成功执行。"
                "请直接向用户总结结果，勿再次调用 send_email、send_wechat_* 或 export_to_excel。"
            )
        reanchor = SystemMessage(content=reanchor_text)

        # 每 N 轮工具后或 HITL 恢复后强制重锚（首条 Human 之后插入）
        llm_input: list[BaseMessage] = [reanchor]
        inserted = False
        for msg in trimmed:
            llm_input.append(msg)
            if (
                not inserted
                and isinstance(msg, HumanMessage)
                and tool_rounds > 0
                and tool_rounds % max(1, AGENT_REANCHOR_EVERY_N_TOOLS) == 0
            ):
                llm_input.append(SystemMessage(content="【进度检查】请对照计划确认当前步骤与下一步。"))
                inserted = True

        updated["llm_input_messages"] = llm_input
        return updated

    return pre_model_hook


def make_post_model_hook():
    async def post_model_hook(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        messages = list(_state_get(state, "messages") or [])
        plan = list(_state_get(state, "plan") or [])
        plan_index = compute_plan_index(plan, messages)
        harness_enabled = bool(_state_get(state, "harness_enabled"))
        task_phase = compute_task_phase(
            plan, plan_index, harness_enabled=harness_enabled
        )
        updated = {
            "plan_index": plan_index,
            "task_phase": task_phase,
        }
        if plan:
            updated["completed_steps"] = plan[:plan_index]
            updated["step_checklist"] = build_step_checklist(plan, plan_index)
        if plan and plan_index >= len(plan):
            updated["task_status"] = "done"
        thread_id = _thread_id_from_config(config)
        base = dict(state) if isinstance(state, dict) else {}
        merged = {**base, **updated}
        _sync_run_context(thread_id, merged)
        return updated

    return post_model_hook


def is_phase_gate_rejection(content: str) -> bool:
    return PHASE_GATE_MARKER in (content or "")


def is_phase_gate_abandon_message(content: str) -> bool:
    text = content or ""
    return PHASE_GATE_ABANDON_MARKER in text and "请勿再尝试" in text


def count_tool_phase_gate_failures(messages: list[BaseMessage], tool_name: str) -> int:
    count = 0
    for m in messages:
        if isinstance(m, ToolMessage) and (m.name or "") == tool_name:
            if is_phase_gate_rejection(str(m.content or "")):
                count += 1
    return count


def detect_abandoned_tools_from_messages(
    messages: list[BaseMessage],
    *,
    threshold: int | None = None,
) -> set[str]:
    cap = threshold if threshold is not None else AGENT_PHASE_GATE_MAX_RETRIES
    counts: dict[str, int] = {}
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        name = m.name or ""
        if not name or not is_phase_gate_rejection(str(m.content or "")):
            continue
        counts[name] = counts.get(name, 0) + 1
    return {name for name, n in counts.items() if n >= cap}


def get_abandoned_tools(thread_id: str) -> set[str]:
    return set(_abandoned_tools.get(thread_id or "default", set()))


def sync_abandoned_tools_from_messages(thread_id: str, messages: list[BaseMessage]) -> set[str]:
    tid = thread_id or "default"
    counts: dict[str, int] = {}
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        name = m.name or ""
        if not name or not is_phase_gate_rejection(str(m.content or "")):
            continue
        counts[name] = counts.get(name, 0) + 1
    if counts:
        attempts = _phase_gate_attempts.setdefault(tid, {})
        for name, n in counts.items():
            attempts[name] = max(attempts.get(name, 0), n)
    cap = AGENT_PHASE_GATE_MAX_RETRIES
    found = {name for name, n in counts.items() if n >= cap}
    if found:
        bucket = _abandoned_tools.setdefault(tid, set())
        bucket.update(found)
    return get_abandoned_tools(thread_id)


def build_abandon_nudge(abandoned: set[str]) -> str:
    if not abandoned:
        return ""
    tools = "、".join(sorted(abandoned))
    return (
        f"\n\n【系统：工具已放弃】{tools} 已多次因当前任务阶段限制无法调用，请勿再尝试。"
        "请立即向用户输出最终回答：用 Markdown 表格或纯文字展示已收集的信息，"
        "并简短说明上述工具暂不可用。不要再次调用这些工具。"
    )


def build_stuck_give_up_nudge(abandoned: set[str]) -> str:
    tools = "、".join(sorted(abandoned)) if abandoned else "部分格式化/外发工具"
    return (
        "【系统强制终止工具重试】"
        f"{tools} 已连续多次因阶段限制失败。"
        "请不要再调用任何工具，直接用 Markdown 表格或文字向用户展示已有结果，"
        "并说明当前无法使用该工具（例如仍在信息收集阶段、无法生成格式化表格）。"
    )


def _phase_gate_message(tool_name: str, phase: TaskPhase, *, attempt: int = 1) -> str:
    allowed = PHASE_LABELS.get(phase, phase)
    base = (
        f"⛔ 工具 {tool_name} 在当前阶段不可用。"
        f"当前阶段：{allowed}。"
        f"请先完成信息收集/整理，进入对应阶段后再调用。"
    )
    if attempt >= 2:
        base += f"（第 {attempt} 次被拒绝，请勿重复调用。）"
    return base


def _phase_gate_abandon_message(tool_name: str, phase: TaskPhase) -> str:
    allowed = PHASE_LABELS.get(phase, phase)
    return (
        f"⛔ 工具 {tool_name} 已放弃：已连续 {AGENT_PHASE_GATE_MAX_RETRIES} 次"
        f"因阶段限制无法调用（当前阶段：{allowed}）。"
        f"请勿再尝试调用 {tool_name}。"
        "请改用 Markdown 表格或纯文字直接向用户展示已有数据，"
        "并简短说明当前无法使用该格式化工具。"
    )


def _record_phase_gate_block(thread_id: str, tool_name: str, phase: TaskPhase) -> str:
    tid = thread_id or "default"
    if tool_name in _abandoned_tools.get(tid, set()):
        return _phase_gate_abandon_message(tool_name, phase)
    attempts = _phase_gate_attempts.setdefault(tid, {})
    attempts[tool_name] = attempts.get(tool_name, 0) + 1
    n = attempts[tool_name]
    if n >= AGENT_PHASE_GATE_MAX_RETRIES:
        _abandoned_tools.setdefault(tid, set()).add(tool_name)
        return _phase_gate_abandon_message(tool_name, phase)
    return _phase_gate_message(tool_name, phase, attempt=n)


def wrap_tools_with_phase_gate(tools: list[BaseTool]) -> list[BaseTool]:
    """按 task_phase 限制工具；harness 未启用时全部放行。"""
    wrapped: list[BaseTool] = []
    for tool in tools:
        wrapped.append(_wrap_tool_phase(tool))
    return wrapped


def _wrap_tool_phase(tool: BaseTool) -> BaseTool:
    name = tool.name

    async def _gated_coroutine(**kwargs: Any) -> str:
        from langchain_core.runnables import ensure_config

        from agent.task_continue import (
            DELIVER_ACTION_TOOLS,
            deliver_duplicate_block_message,
            is_deliver_tool_done,
        )

        config = ensure_config()
        thread_id = _thread_id_from_config(config)
        if name in DELIVER_ACTION_TOOLS and is_deliver_tool_done(thread_id, name):
            return deliver_duplicate_block_message(name)
        ctx = _run_task_context.get(thread_id, {})
        if not ctx.get("harness_enabled"):
            result = await tool.ainvoke(kwargs)
            return result if isinstance(result, str) else str(result)

        phase: TaskPhase = ctx.get("task_phase") or "gather"
        allowed = allowed_tools_for_phase(phase, harness_enabled=True)
        if name not in allowed:
            return _record_phase_gate_block(thread_id, name, phase)
        result = await tool.ainvoke(kwargs)
        return result if isinstance(result, str) else str(result)

    def _gated_sync(**kwargs: Any) -> str:
        from langchain_core.runnables import ensure_config

        config = ensure_config()
        thread_id = _thread_id_from_config(config)
        ctx = _run_task_context.get(thread_id, {})
        if not ctx.get("harness_enabled"):
            result = tool.invoke(kwargs)
            return result if isinstance(result, str) else str(result)

        phase: TaskPhase = ctx.get("task_phase") or "gather"
        allowed = allowed_tools_for_phase(phase, harness_enabled=True)
        if name not in allowed:
            return _record_phase_gate_block(thread_id, name, phase)
        result = tool.invoke(kwargs)
        return result if isinstance(result, str) else str(result)

    return StructuredTool(
        name=tool.name,
        description=tool.description or tool.name,
        args_schema=tool.args_schema,
        coroutine=_gated_coroutine,
        func=_gated_sync if hasattr(tool, "func") and tool.func else None,
    )


TASK_STATE_KEYS = (
    "user_goal",
    "plan",
    "plan_index",
    "task_phase",
    "harness_enabled",
    "completed_steps",
    "step_checklist",
    "task_status",
)


def merge_task_state(base: dict[str, Any], patch: dict[str, Any]) -> dict[str, Any]:
    """合并 task 字段：LangGraph 中间 checkpoint 常只带 messages，需保留初始 goal/plan。"""
    out = dict(base)
    for key in TASK_STATE_KEYS:
        if key not in patch:
            continue
        val = patch.get(key)
        if key == "user_goal":
            if (val or "").strip():
                out[key] = str(val).strip()
            continue
        if key == "plan":
            if isinstance(val, list) and val:
                out[key] = list(val)
            continue
        if key == "completed_steps":
            if isinstance(val, list) and val:
                out[key] = list(val)
            continue
        if key == "step_checklist":
            if isinstance(val, list) and val:
                out[key] = list(val)
            continue
        if key == "harness_enabled":
            if "harness_enabled" in patch:
                out[key] = bool(val)
            continue
        if val is not None and val != "":
            out[key] = val
    return out


def task_harness_event_payload(state: dict[str, Any]) -> dict[str, Any]:
    """构造 SSE task_harness 事件，供前端展示结构化任务状态。"""
    phase = state.get("task_phase") or "gather"
    plan = list(state.get("plan") or [])
    plan_index = int(state.get("plan_index") or 0)
    checklist = list(state.get("step_checklist") or build_step_checklist(plan, plan_index))
    return {
        "type": "task_harness",
        "user_goal": (state.get("user_goal") or "").strip(),
        "plan": plan,
        "plan_index": plan_index,
        "task_phase": phase,
        "task_phase_label": PHASE_LABELS.get(phase, str(phase)),
        "harness_enabled": bool(state.get("harness_enabled")),
        "completed_steps": list(state.get("completed_steps") or []),
        "step_checklist": checklist,
        "current_step": plan[plan_index] if plan and 0 <= plan_index < len(plan) else "",
        "plan_total": len(plan),
    }


def sync_run_context_from_values(session_id: str, values: dict[str, Any]) -> None:
    """从 checkpoint 恢复 task 上下文（HITL resume 前调用，供 phase gate 使用）。"""
    _sync_run_context(session_id, values)


def clear_run_context(session_id: str) -> None:
    sid = session_id or "default"
    _run_task_context.pop(sid, None)
    _phase_gate_attempts.pop(sid, None)
    _abandoned_tools.pop(sid, None)


async def prepare_agent_invoke(
    *,
    session_id: str,
    lc_messages: list | None,
    resume_action: str | None,
    fresh_thread: bool,
    file_count: int = 0,
) -> tuple[dict, Any, bool]:
    """准备 Agent config 与 input；返回 (config, input_data, is_continue_request)。"""
    from agent.agent_checkpointer import get_checkpointer, reset_agent_thread
    from config.app_config import AGENT_RECURSION_LIMIT

    is_continue = False
    if lc_messages and resume_action is None:
        _, is_continue = resolve_user_goal(lc_messages)

    if is_continue:
        fresh_thread = True

    checkpointer = get_checkpointer()
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": AGENT_RECURSION_LIMIT,
    } if checkpointer else {"recursion_limit": AGENT_RECURSION_LIMIT}

    if checkpointer and fresh_thread and resume_action is None:
        await reset_agent_thread(session_id)
        import chat.chat_store as chat_store

        if not is_continue:
            chat_store.clear_task_harness_meta(session_id)

    if resume_action is not None:
        return config, Command(resume={"action": resume_action}), False

    if is_continue:
        input_state = await build_initial_agent_state(
            lc_messages or [],
            session_id=session_id,
            file_count=file_count,
        )
        return config, input_state, True

    input_state = await build_initial_agent_state(
        lc_messages or [],
        session_id=session_id,
        file_count=file_count,
    )
    return config, input_state, False


def persist_task_harness_meta(session_id: str, state: dict[str, Any]) -> None:
    if not state.get("harness_enabled"):
        return
    from agent.task_checklist import task_harness_meta_from_state
    import chat.chat_store as chat_store

    chat_store.save_task_harness_meta(session_id, task_harness_meta_from_state(state))
