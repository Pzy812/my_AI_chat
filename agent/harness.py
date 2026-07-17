"""Task Harness：上下文裁剪、重锚定、工具阶段 gate、pre/post hooks。"""
from __future__ import annotations

import logging
from time import perf_counter
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

from agent.planner import (
    build_task_plan,
    ensure_plan_has_delivery,
    format_plan_for_display,
    needs_task_harness,
)
from agent.task_checklist import (
    build_step_checklist,
    extract_user_goal,
    resolve_user_goal,
)
from agent.task_state import (
    PHASE_GATE_EXEMPT_TOOLS,
    PHASE_LABELS,
    TaskPhase,
    allowed_tools_for_phase,
    default_task_fields,
    infer_phase_from_step,
)
from agent.task_runtime import (
    CONTROL_COMPLETE_TOOL,
    build_step_states,
    canonical_tool_name,
    clear_runtime_events,
    evaluate_progress,
    is_delivery_verification_step,
    make_tool_event,
    record_runtime_event,
    runtime_events,
    tool_output_succeeded,
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
    "2. 信息收集阶段优先 search/read；只有状态机推进到交付阶段后，才调用 "
    "send_email、send_wechat_*、export_to_excel（执行前仍需用户确认）。\n"
    "3. 工具型步骤由系统根据成功事件自动更新；纯分析、总结等无工具步骤完成后，"
    "必须调用 mark_step_complete(evidence=...) 显式提交完成证据。\n"
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
    previous = _run_task_context.get(thread_id, {})
    step_states = list(state.get("step_states") or [])
    plan_index = int(state.get("plan_index") or 0)
    current_step_id = ""
    if step_states and 0 <= plan_index < len(step_states):
        current_step_id = str(step_states[plan_index].get("id") or "")
    _run_task_context[thread_id] = {
        "harness_enabled": bool(state.get("harness_enabled")),
        "task_phase": state.get("task_phase") or "gather",
        "user_goal": state.get("user_goal") or "",
        "plan": list(state.get("plan") or []),
        "plan_index": plan_index,
        "current_step_id": current_step_id,
        "step_states": step_states,
        "tool_events": list(state.get("tool_events") or []),
        "deliver_done_tools": set(previous.get("deliver_done_tools") or set()),
        "user_turn_serial": int(previous.get("user_turn_serial") or 0),
    }


def sync_run_context_deliver_state(thread_id: str, messages: list[BaseMessage]) -> None:
    from agent.task_continue import (
        count_real_user_messages,
        get_deliver_done_tools,
        sync_deliver_completion_flags,
    )

    tid = (thread_id or "default").strip() or "default"
    sync_deliver_completion_flags(tid, messages)
    ctx = _run_task_context.setdefault(tid, {})
    ctx["deliver_done_tools"] = get_deliver_done_tools(tid)
    ctx["user_turn_serial"] = count_real_user_messages(messages)


def get_run_context_deliver_done(thread_id: str) -> set[str] | None:
    tid = (thread_id or "default").strip() or "default"
    ctx = _run_task_context.get(tid)
    if ctx is None:
        return None
    done = ctx.get("deliver_done_tools")
    if done is None:
        return None
    return set(done)


def patch_deliver_done_in_run_context(thread_id: str, tool_name: str) -> None:
    tid = (thread_id or "default").strip() or "default"
    ctx = _run_task_context.setdefault(tid, {})
    done = set(ctx.get("deliver_done_tools") or set())
    done.add(tool_name)
    ctx["deliver_done_tools"] = done


def clear_run_context_deliver_state(thread_id: str) -> None:
    tid = (thread_id or "default").strip() or "default"
    ctx = _run_task_context.get(tid)
    if ctx is None:
        return
    ctx.pop("deliver_done_tools", None)
    ctx.pop("user_turn_serial", None)


def _state_get(state: Any, key: str, default=None):
    if isinstance(state, dict):
        return state.get(key, default)
    return getattr(state, key, default)


def compute_plan_index(plan: list[str], messages: list[BaseMessage]) -> int:
    """Legacy compatibility helper; runtime progress no longer calls this."""
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
    step_states = list(_state_get(state, "step_states") or [])
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
        shown = sorted(set(allowed) | set(PHASE_GATE_EXEMPT_TOOLS))
        lines.append("本阶段允许工具：" + ", ".join(shown))
        lines.append("外发类工具仅在交付阶段开放，并仍受 HITL 确认保护。")
        if step_states and plan_index < len(step_states):
            expected = step_states[plan_index].get("expected_tools") or []
            if expected:
                lines.append("当前步骤完成证据：" + ", ".join(expected) + " 成功事件。")
            else:
                lines.append(
                    "当前步骤没有必需工具；完成分析/总结后调用 "
                    "mark_step_complete(evidence=完成依据) 推进状态。"
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
            persisted_plan = list(persisted["plan"])
            fields["plan"] = ensure_plan_has_delivery(
                persisted_plan, fields["user_goal"], max_steps=None
            )
            fields["plan_index"] = int(persisted.get("plan_index") or 0)
            fields["task_phase"] = compute_task_phase(
                fields["plan"], fields["plan_index"], harness_enabled=True
            )
            fields["completed_steps"] = list(persisted.get("completed_steps") or [])
            fields["step_checklist"] = list(persisted.get("step_checklist") or [])
            fields["task_status"] = persisted.get("task_status") or "executing"
            fields["step_states"] = list(persisted.get("step_states") or [])
            if len(fields["plan"]) > len(persisted_plan):
                appended = build_step_states(fields["plan"])[len(persisted_plan) :]
                if fields["plan_index"] >= len(persisted_plan) and appended:
                    appended[0]["status"] = "running"
                fields["step_states"].extend(appended)
                fields["step_checklist"] = build_step_checklist(
                    fields["plan"], fields["plan_index"]
                )
            fields["tool_events"] = list(persisted.get("tool_events") or [])
            fields["harness_enabled"] = bool(persisted.get("harness_enabled", True))
            harness = fields["harness_enabled"]
        elif harness and user_goal:
            fields["task_status"] = "planning"
            plan = await build_task_plan(user_goal)
            fields["plan"] = plan
            fields["plan_index"] = 0
            fields["task_phase"] = compute_task_phase(plan, 0, harness_enabled=True)
            fields["step_checklist"] = build_step_checklist(plan, 0)
            fields["step_states"] = build_step_states(plan)
            fields["task_status"] = "executing"
    elif harness and user_goal:
        fields["task_status"] = "planning"
        plan = await build_task_plan(user_goal)
        fields["plan"] = plan
        fields["plan_index"] = 0
        fields["task_phase"] = compute_task_phase(plan, 0, harness_enabled=True)
        fields["step_checklist"] = build_step_checklist(plan, 0)
        fields["step_states"] = build_step_states(plan)
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


def reconcile_task_runtime(state: dict[str, Any], thread_id: str) -> dict[str, Any]:
    """Fold recorded tool events into persistent task state."""
    plan = list(_state_get(state, "plan") or [])
    harness_enabled = bool(_state_get(state, "harness_enabled"))
    if not harness_enabled or not plan:
        return {
            "plan_index": 0,
            "task_phase": "deliver",
        }
    progress = evaluate_progress(
        plan,
        _state_get(state, "step_states") or [],
        _state_get(state, "tool_events") or [],
        runtime_events(thread_id),
    )
    progress["task_phase"] = compute_task_phase(
        plan,
        int(progress["plan_index"]),
        harness_enabled=True,
    )
    return progress


def finalize_task_after_delivery(
    state: dict[str, Any], messages: list[BaseMessage]
) -> dict[str, Any]:
    """Treat a successful requested side effect as the authoritative task terminal.

    This also repairs old/stale 7-step checkpoints: unfinished preparation steps
    are skipped because the delivered payload proves they can no longer require a
    second side effect, while delivery/verification steps are marked succeeded.
    """
    from agent.task_continue import deliver_goal_satisfied, required_deliver_tools

    goal = str(state.get("user_goal") or "").strip()
    plan = list(state.get("plan") or [])
    if not plan or not deliver_goal_satisfied(goal, messages):
        return {}

    required = required_deliver_tools(goal)
    steps = build_step_states(plan)
    raw_steps = list(state.get("step_states") or [])
    if len(raw_steps) == len(plan):
        steps = [dict(step) for step in raw_steps]

    for step in steps:
        if step.get("status") in ("succeeded", "skipped"):
            continue
        expected = {
            canonical_tool_name(str(name))
            for name in (step.get("expected_tools") or [])
        }
        description = str(step.get("description") or "")
        is_delivery = bool(expected & required) or infer_phase_from_step(description) == "deliver"
        if is_delivery or is_delivery_verification_step(description):
            step["status"] = "succeeded"
        else:
            step["status"] = "skipped"
        step["error"] = None

    checklist = [
        {
            "index": i,
            "step": description,
            "status": steps[i].get("status", "succeeded"),
            "done": True,
            "current": False,
            "attempts": int(steps[i].get("attempts") or 0),
            "error": None,
        }
        for i, description in enumerate(plan)
    ]
    return {
        "plan_index": len(plan),
        "task_phase": "deliver",
        "task_status": "done",
        "step_states": steps,
        "step_checklist": checklist,
        "completed_steps": list(plan),
    }


def format_reanchor_summary(state: dict[str, Any]) -> str:
    text = build_reanchor_text(state)
    return text.replace("【任务续跑上下文 — 每步推理前必读】", "【HITL 确认后继续 — 任务重锚定】", 1)


def make_pre_model_hook():
    async def pre_model_hook(state: dict[str, Any], config: RunnableConfig) -> dict[str, Any]:
        messages = list(_state_get(state, "messages") or [])
        from agent.task_continue import deliver_tools_used

        thread_id = _thread_id_from_config(config)
        sync_abandoned_tools_from_messages(thread_id, messages)
        sync_run_context_deliver_state(thread_id, messages)
        tool_rounds = count_tool_rounds(messages)
        updated = reconcile_task_runtime(state, thread_id)

        base = dict(state) if isinstance(state, dict) else {}
        merged = {**base, **updated}
        terminal = finalize_task_after_delivery(merged, messages)
        if terminal:
            updated.update(terminal)
            merged.update(terminal)
        _sync_run_context(thread_id, merged)

        trimmed = trim_messages_for_llm(messages)
        reanchor_text = build_reanchor_text(merged)
        abandon_nudge = build_abandon_nudge(get_abandoned_tools(thread_id))
        if abandon_nudge:
            reanchor_text += abandon_nudge
        if deliver_tools_used(messages):
            reanchor_text += (
                "\n\n【本轮回发已完成】本轮用户请求中的邮件/微信/导出已成功执行。"
                "请直接向用户总结结果，勿在本轮再次调用 send_email、send_wechat_* 或 export_to_excel。"
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
        thread_id = _thread_id_from_config(config)
        updated = reconcile_task_runtime(state, thread_id)
        base = dict(state) if isinstance(state, dict) else {}
        merged = {**base, **updated}
        terminal = finalize_task_after_delivery(
            merged, list(_state_get(state, "messages") or [])
        )
        if terminal:
            updated.update(terminal)
            merged.update(terminal)
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


def _delivery_wait_message(tool_name: str, phase: TaskPhase) -> str:
    """A deliver request is deferred, not abandoned, while prerequisites remain."""
    label = PHASE_LABELS.get(phase, phase)
    return (
        f"⏳ 工具 {tool_name} 尚未执行。当前阶段：{label}。"
        "仍有缺少成功证据的前置步骤；请完成这些步骤后再次调用。"
        "交付工具不会因为阶段等待而被永久禁用。"
    )


def _delivery_payload_ready(tool_name: str, kwargs: dict[str, Any]) -> bool:
    """Require substantive delivery arguments before treating them as synthesis evidence."""
    if tool_name == "send_email":
        return bool(str(kwargs.get("to_email") or "").strip()) and len(
            str(kwargs.get("content") or "").strip()
        ) >= 40
    if tool_name == "export_to_excel":
        return bool(kwargs.get("headers")) and bool(kwargs.get("rows"))
    if tool_name == "send_wechat_message":
        return bool(str(kwargs.get("to_name") or "").strip()) and len(
            str(kwargs.get("message") or kwargs.get("content") or "").strip()
        ) >= 20
    if tool_name == "send_wechat_files":
        return bool(kwargs.get("file_paths"))
    return False


def _successful_task_events(ctx: dict[str, Any], thread_id: str) -> list[dict[str, Any]]:
    combined = [
        *list(ctx.get("tool_events") or []),
        *runtime_events(thread_id),
    ]
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for event in combined:
        if not isinstance(event, dict) or not event.get("success"):
            continue
        event_id = str(event.get("id") or "")
        if event_id and event_id in seen:
            continue
        if event_id:
            seen.add(event_id)
        out.append(event)
    return out


def _try_fast_forward_to_delivery(
    thread_id: str,
    tool_name: str,
    kwargs: dict[str, Any],
    ctx: dict[str, Any],
) -> dict[str, Any]:
    """Reuse task-level evidence and delivery payload to finish prerequisites.

    A single batch/search can legitimately satisfy several gather plan items. A
    substantive email/export payload also proves that intervening reasoning-only
    synthesis steps have completed. Steps requiring a missing concrete tool (for
    example format_pretty_table) are never bypassed.
    """
    if not ctx.get("harness_enabled") or not _delivery_payload_ready(tool_name, kwargs):
        return ctx
    plan = list(ctx.get("plan") or [])
    steps = list(ctx.get("step_states") or [])
    current = int(ctx.get("plan_index") or 0)
    if not plan or not steps or current >= len(plan):
        return ctx
    deliver_index = next(
        (i for i in range(current, len(plan)) if infer_phase_from_step(plan[i]) == "deliver"),
        None,
    )
    if deliver_index is None or deliver_index <= current:
        return ctx

    successful = _successful_task_events(ctx, thread_id)
    synthetic: list[dict[str, Any]] = []
    for index in range(current, deliver_index):
        step = steps[index]
        if step.get("status") in ("succeeded", "skipped"):
            continue
        expected = list(step.get("expected_tools") or [])
        if expected:
            matches: list[tuple[str, dict[str, Any]]] = []
            for required in expected:
                already_attached = next(
                    (
                        event
                        for event in successful
                        if str(event.get("step_id") or "") == str(step.get("id") or "")
                        and canonical_tool_name(str(event.get("tool_name") or ""))
                        == canonical_tool_name(str(required))
                    ),
                    None,
                )
                if already_attached is not None:
                    continue
                match = next(
                    (
                        event
                        for event in successful
                        if canonical_tool_name(str(event.get("tool_name") or ""))
                        == canonical_tool_name(str(required))
                    ),
                    None,
                )
                if match is None:
                    return ctx
                matches.append((required, match))
            for required, source in matches:
                synthetic.append(
                    make_tool_event(
                        tool_name=required,
                        arguments={"reused_event_id": source.get("id") or ""},
                        success=True,
                        output=(
                            "复用任务级成功证据："
                            + str(source.get("output_summary") or "")[:500]
                        ),
                        latency_ms=0,
                        step_id=str(step.get("id") or f"step-{index + 1}"),
                    )
                )
        else:
            already_completed = any(
                str(event.get("step_id") or "") == str(step.get("id") or "")
                and event.get("tool_name") == CONTROL_COMPLETE_TOOL
                for event in successful
            )
            if not already_completed:
                synthetic.append(
                    make_tool_event(
                        tool_name=CONTROL_COMPLETE_TOOL,
                        arguments={"source": tool_name},
                        success=True,
                        output="交付参数已包含整理后的实质内容，自动确认该推理步骤完成。",
                        latency_ms=0,
                        step_id=str(step.get("id") or f"step-{index + 1}"),
                    )
                )

    for event in synthetic:
        record_runtime_event(thread_id, event)
    progress = evaluate_progress(
        plan,
        steps,
        ctx.get("tool_events") or [],
        runtime_events(thread_id),
    )
    merged = {**ctx, **progress}
    merged["task_phase"] = compute_task_phase(
        plan,
        int(progress["plan_index"]),
        harness_enabled=True,
    )
    _sync_run_context(thread_id, merged)
    return _run_task_context.get(thread_id, merged)


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


def make_progress_control_tool() -> BaseTool:
    """Internal control tool for completing reasoning-only plan steps."""
    from pydantic import BaseModel, Field

    class CompleteStepInput(BaseModel):
        evidence: str = Field(
            description="简要说明本步骤已完成的依据或产出；不能只写‘完成’"
        )

    def _complete(evidence: str) -> str:
        body = (evidence or "").strip()
        if len(body) < 4:
            return "❌ 完成依据过短，请说明本步骤的实际产出"
        return f"✅ 当前步骤已提交完成证据：{body[:500]}"

    return StructuredTool.from_function(
        func=_complete,
        name=CONTROL_COMPLETE_TOOL,
        description=(
            "仅用于完成不需要外部工具的分析、整理、总结步骤。"
            "工具型步骤不要调用它，工具成功事件会自动推进。"
        ),
        args_schema=CompleteStepInput,
    )


def _wrap_tool_phase(tool: BaseTool) -> BaseTool:
    name = tool.name

    async def _gated_coroutine(**kwargs: Any) -> str:
        from langchain_core.runnables import ensure_config

        from agent.task_continue import (
            DELIVER_ACTION_TOOLS,
            deliver_duplicate_block_message,
            is_deliver_tool_done,
            mark_deliver_tool_done,
            release_deliver_tool,
            reserve_deliver_tool,
        )

        config = ensure_config()
        thread_id = _thread_id_from_config(config)
        ctx = _run_task_context.get(thread_id, {})
        started = perf_counter()

        def finish(
            output: Any,
            *,
            success: bool | None = None,
            executed: bool = True,
        ) -> str:
            text = output if isinstance(output, str) else str(output)
            if ctx.get("harness_enabled"):
                record_runtime_event(
                    thread_id,
                    make_tool_event(
                        tool_name=name,
                        arguments=kwargs,
                        success=tool_output_succeeded(text) if success is None else success,
                        output=text,
                        latency_ms=int((perf_counter() - started) * 1000),
                        step_id=str(ctx.get("current_step_id") or ""),
                        executed=executed,
                    ),
                )
            return text

        if name in DELIVER_ACTION_TOOLS and is_deliver_tool_done(thread_id, name):
            return finish(
                deliver_duplicate_block_message(name),
                success=False,
                executed=False,
            )
        if not ctx.get("harness_enabled"):
            if name in DELIVER_ACTION_TOOLS and not reserve_deliver_tool(thread_id, name):
                return deliver_duplicate_block_message(name)
            try:
                result = await tool.ainvoke(kwargs)
                text = result if isinstance(result, str) else str(result)
                if name in DELIVER_ACTION_TOOLS:
                    if tool_output_succeeded(text):
                        mark_deliver_tool_done(thread_id, name)
                    else:
                        release_deliver_tool(thread_id, name)
                return text
            except Exception:
                release_deliver_tool(thread_id, name)
                raise

        from agent.task_state import PHASE_GATE_EXEMPT_TOOLS

        if name in PHASE_GATE_EXEMPT_TOOLS:
            try:
                return finish(await tool.ainvoke(kwargs))
            except Exception as exc:
                finish(f"{type(exc).__name__}: {exc}", success=False)
                raise

        phase: TaskPhase = ctx.get("task_phase") or "gather"
        allowed = allowed_tools_for_phase(phase, harness_enabled=True)
        if name in DELIVER_ACTION_TOOLS and name not in allowed:
            ctx = _try_fast_forward_to_delivery(thread_id, name, kwargs, ctx)
            phase = ctx.get("task_phase") or phase
            allowed = allowed_tools_for_phase(phase, harness_enabled=True)
        if name not in allowed:
            message = (
                _delivery_wait_message(name, phase)
                if name in DELIVER_ACTION_TOOLS
                else _record_phase_gate_block(thread_id, name, phase)
            )
            return finish(message, success=False, executed=False)
        if name in DELIVER_ACTION_TOOLS and not reserve_deliver_tool(thread_id, name):
            return finish(
                deliver_duplicate_block_message(name), success=False, executed=False
            )
        try:
            result = await tool.ainvoke(kwargs)
            text = finish(result)
            if name in DELIVER_ACTION_TOOLS:
                if tool_output_succeeded(text):
                    mark_deliver_tool_done(thread_id, name)
                else:
                    release_deliver_tool(thread_id, name)
            return text
        except Exception as exc:
            release_deliver_tool(thread_id, name)
            finish(f"{type(exc).__name__}: {exc}", success=False)
            raise

    def _gated_sync(**kwargs: Any) -> str:
        from langchain_core.runnables import ensure_config

        from agent.task_continue import (
            DELIVER_ACTION_TOOLS,
            deliver_duplicate_block_message,
            is_deliver_tool_done,
            mark_deliver_tool_done,
            release_deliver_tool,
            reserve_deliver_tool,
        )

        config = ensure_config()
        thread_id = _thread_id_from_config(config)
        ctx = _run_task_context.get(thread_id, {})
        started = perf_counter()

        def finish(
            output: Any,
            *,
            success: bool | None = None,
            executed: bool = True,
        ) -> str:
            text = output if isinstance(output, str) else str(output)
            if ctx.get("harness_enabled"):
                record_runtime_event(
                    thread_id,
                    make_tool_event(
                        tool_name=name,
                        arguments=kwargs,
                        success=tool_output_succeeded(text) if success is None else success,
                        output=text,
                        latency_ms=int((perf_counter() - started) * 1000),
                        step_id=str(ctx.get("current_step_id") or ""),
                        executed=executed,
                    ),
                )
            return text

        if name in DELIVER_ACTION_TOOLS and is_deliver_tool_done(thread_id, name):
            return finish(
                deliver_duplicate_block_message(name),
                success=False,
                executed=False,
            )
        if not ctx.get("harness_enabled"):
            if name in DELIVER_ACTION_TOOLS and not reserve_deliver_tool(thread_id, name):
                return deliver_duplicate_block_message(name)
            try:
                result = tool.invoke(kwargs)
                text = result if isinstance(result, str) else str(result)
                if name in DELIVER_ACTION_TOOLS:
                    if tool_output_succeeded(text):
                        mark_deliver_tool_done(thread_id, name)
                    else:
                        release_deliver_tool(thread_id, name)
                return text
            except Exception:
                release_deliver_tool(thread_id, name)
                raise

        from agent.task_state import PHASE_GATE_EXEMPT_TOOLS

        if name in PHASE_GATE_EXEMPT_TOOLS:
            try:
                return finish(tool.invoke(kwargs))
            except Exception as exc:
                finish(f"{type(exc).__name__}: {exc}", success=False)
                raise

        phase: TaskPhase = ctx.get("task_phase") or "gather"
        allowed = allowed_tools_for_phase(phase, harness_enabled=True)
        if name in DELIVER_ACTION_TOOLS and name not in allowed:
            ctx = _try_fast_forward_to_delivery(thread_id, name, kwargs, ctx)
            phase = ctx.get("task_phase") or phase
            allowed = allowed_tools_for_phase(phase, harness_enabled=True)
        if name not in allowed:
            message = (
                _delivery_wait_message(name, phase)
                if name in DELIVER_ACTION_TOOLS
                else _record_phase_gate_block(thread_id, name, phase)
            )
            return finish(message, success=False, executed=False)
        if name in DELIVER_ACTION_TOOLS and not reserve_deliver_tool(thread_id, name):
            return finish(
                deliver_duplicate_block_message(name), success=False, executed=False
            )
        try:
            result = tool.invoke(kwargs)
            text = finish(result)
            if name in DELIVER_ACTION_TOOLS:
                if tool_output_succeeded(text):
                    mark_deliver_tool_done(thread_id, name)
                else:
                    release_deliver_tool(thread_id, name)
            return text
        except Exception as exc:
            release_deliver_tool(thread_id, name)
            finish(f"{type(exc).__name__}: {exc}", success=False)
            raise

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
    "step_states",
    "tool_events",
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
        if key in ("step_states", "tool_events"):
            if isinstance(val, list):
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
    public_events = [
        {
            "id": event.get("id"),
            "tool_name": event.get("tool_name"),
            "success": bool(event.get("success")),
            "output_summary": str(event.get("output_summary") or "")[:500],
            "latency_ms": int(event.get("latency_ms") or 0),
            "step_id": event.get("step_id") or "",
            "created_at": event.get("created_at") or "",
            "executed": event.get("executed", True),
        }
        for event in list(state.get("tool_events") or [])[-50:]
        if isinstance(event, dict)
    ]
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
        "step_states": list(state.get("step_states") or []),
        "tool_events": public_events,
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
    clear_runtime_events(sid)


async def prepare_agent_invoke(
    *,
    session_id: str,
    lc_messages: list | None,
    resume_action: str | None,
    fresh_thread: bool,
    file_count: int = 0,
    rag_mode: str | None = None,
    stream: bool = False,
) -> tuple[dict, Any, bool]:
    """准备 Agent config 与 input；返回 (config, input_data, is_continue_request)。"""
    from agent.agent_checkpointer import get_checkpointer, reset_agent_thread
    from config.app_config import AGENT_RECURSION_LIMIT
    from observability.langsmith_config import enrich_agent_config
    from observability.langsmith_session import peek_turn_index
    from observability.langsmith_trace import attach_run_capture

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
    attach_run_capture(config, session_id)
    turn_index = peek_turn_index(session_id, is_resume=resume_action is not None)
    config = enrich_agent_config(
        config,
        session_id=session_id,
        rag_mode=rag_mode,
        stream=stream,
        resume_action=resume_action,
        turn_index=turn_index,
    )

    if fresh_thread and resume_action is None:
        if checkpointer:
            await reset_agent_thread(session_id)
            import chat.chat_store as chat_store

            if not is_continue:
                chat_store.clear_task_harness_meta(session_id)
        else:
            from agent.task_continue import clear_deliver_flags

            clear_deliver_flags(session_id)
            clear_run_context(session_id)
        if lc_messages:
            from agent.task_continue import sync_deliver_completion_flags

            sync_deliver_completion_flags(session_id, lc_messages)
            sync_run_context_deliver_state(session_id, lc_messages)

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
