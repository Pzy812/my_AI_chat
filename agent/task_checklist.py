"""复杂任务 checklist、继续指令识别与自动续跑。"""
from __future__ import annotations

import os
import re

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from agent.planner import needs_task_harness
from agent.task_continue import (
    deliver_goal_satisfied,
    deliver_tools_used,
    should_continue_deliver,
    user_goal_requires_deliver,
)
from chat.chat_helpers import messages_have_pending_tool_calls

MAX_TASK_CONTINUATIONS = int(os.getenv("MAX_TASK_CONTINUATIONS", "5"))

_CONTINUE_MESSAGE_RE = re.compile(
    r"^(继续|接着做|接着|继续执行|请继续|往下做|往下|go\s*on|continue)[。.!?\s]*$",
    re.IGNORECASE,
)

_PROMISE_NEXT_STEP_RE = re.compile(
    r"(接下来|下一步|然后|现在|准备|我将|需要先).{0,20}"
    r"(搜索|查询|查找|收集|整理|汇总|发送|导出|生成|制作|撰写|查找论文|发邮件|发微信)",
    re.IGNORECASE,
)


def is_continue_message(text: str) -> bool:
    return bool(_CONTINUE_MESSAGE_RE.match((text or "").strip()))


def human_message_text(msg: HumanMessage) -> str:
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


def extract_user_goal(messages: list[BaseMessage]) -> str:
    for msg in reversed(messages):
        if isinstance(msg, HumanMessage):
            return human_message_text(msg)
    return ""


def extract_primary_user_goal(messages: list[BaseMessage]) -> str:
    """最近一条非「继续」类用户消息；若上一条是继续，则回溯到首条复杂任务指令。"""
    candidates: list[str] = []
    for msg in messages:
        if not isinstance(msg, HumanMessage):
            continue
        text = human_message_text(msg)
        if text and not is_continue_message(text):
            candidates.append(text)

    if not candidates:
        return extract_user_goal(messages)

    last_raw = extract_user_goal(messages)
    if is_continue_message(last_raw) and len(candidates) >= 1:
        for text in reversed(candidates):
            if needs_task_harness(text):
                return text
        return candidates[0]
    return candidates[-1]


def resolve_user_goal(messages: list[BaseMessage]) -> tuple[str, bool]:
    last = extract_user_goal(messages)
    is_continue = is_continue_message(last)
    goal = extract_primary_user_goal(messages) if is_continue else last
    return goal, is_continue


def build_step_checklist(plan: list[str], plan_index: int) -> list[dict]:
    """构造可持久化的子任务 checklist（供前端 ✓/✗ 展示）。"""
    out: list[dict] = []
    idx = max(0, min(int(plan_index or 0), len(plan)))
    for i, step in enumerate(plan):
        out.append(
            {
                "index": i,
                "step": step,
                "done": i < idx,
                "current": i == idx and idx < len(plan),
            }
        )
    return out


def _last_ai_message(messages: list) -> AIMessage | None:
    for m in reversed(messages):
        if isinstance(m, AIMessage):
            return m
    return None


def assistant_promised_next_step(reply: str) -> bool:
    text = (reply or "").strip()
    if not text:
        return False
    return bool(_PROMISE_NEXT_STEP_RE.search(text))


def task_plan_incomplete(state: dict, messages: list) -> bool:
    if not state.get("harness_enabled"):
        return False
    plan = list(state.get("plan") or [])
    if not plan:
        return False
    plan_index = int(state.get("plan_index") or 0)
    goal = (state.get("user_goal") or "").strip()
    if user_goal_requires_deliver(goal):
        delivered = deliver_goal_satisfied(goal, messages)
        if delivered:
            return False
        if not delivered:
            return True
    return plan_index < len(plan)


def format_checklist_text(plan: list[str], plan_index: int) -> str:
    lines: list[str] = []
    for i, step in enumerate(plan):
        if i < plan_index:
            mark = "✓"
        elif i == plan_index:
            mark = "→"
        else:
            mark = "✗"
        lines.append(f"{mark} {i + 1}. {step}")
    return "\n".join(lines)


def build_task_nudge(state: dict, messages: list, reply: str | None) -> str:
    plan = list(state.get("plan") or [])
    plan_index = int(state.get("plan_index") or 0)
    goal = (state.get("user_goal") or "").strip()
    current = plan[plan_index] if plan and 0 <= plan_index < len(plan) else ""
    checklist = format_checklist_text(plan, plan_index) if plan else "（无分步计划）"
    phase = state.get("task_phase") or "gather"
    gather_hint = ""
    if phase == "gather" and len(plan) - plan_index > 1:
        gather_hint = (
            "\n提示：Gather 阶段若有多个独立搜索主题，优先使用 web_search_batch(queries=[...]) 并行检索。"
        )
    return (
        "【系统自动续跑】复杂任务尚未全部完成，请立即继续执行，不要结束本轮。\n"
        f"原始目标：{goal[:1200]}\n"
        f"进度清单：\n{checklist}\n"
        f"当前应完成：第 {plan_index + 1} 步 — {current}\n"
        "请调用必要工具完成当前及后续步骤，勿重复已完成的工作。"
        + gather_hint
        + (f"\n（你刚才的回复末尾：{(reply or '')[-200:]}）" if reply else "")
    )


def should_continue_task(
    state: dict,
    messages: list,
    reply: str | None,
) -> str | None:
    """计划未完成且 Agent 已给出文本回复时，返回续跑 nudge。"""
    if not reply or not reply.strip():
        return None
    if messages_have_pending_tool_calls(messages):
        return None

    last_ai = _last_ai_message(messages)
    if last_ai and last_ai.tool_calls:
        return None

    if not state.get("harness_enabled"):
        return should_continue_deliver(state, messages, reply)

    if not task_plan_incomplete(state, messages):
        return None

    deliver_nudge = should_continue_deliver(state, messages, reply)
    if deliver_nudge:
        return deliver_nudge

    if assistant_promised_next_step(reply) or task_plan_incomplete(state, messages):
        return build_task_nudge(state, messages, reply)
    return None


def task_harness_meta_from_state(state: dict) -> dict:
    plan = list(state.get("plan") or [])
    plan_index = int(state.get("plan_index") or 0)
    return {
        "user_goal": (state.get("user_goal") or "").strip(),
        "plan": plan,
        "plan_index": plan_index,
        "task_phase": state.get("task_phase") or "gather",
        "harness_enabled": bool(state.get("harness_enabled")),
        "completed_steps": list(state.get("completed_steps") or []),
        "step_checklist": list(state.get("step_checklist") or build_step_checklist(plan, plan_index)),
        "task_status": state.get("task_status") or "executing",
        "step_states": list(state.get("step_states") or []),
        "tool_events": list(state.get("tool_events") or []),
    }
