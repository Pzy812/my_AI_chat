"""Task Harness：计划生成与复杂任务检测。"""
from __future__ import annotations

import json
import logging
import re

from langchain_core.messages import HumanMessage, SystemMessage

from agent.task_continue import goal_requires_gather, user_goal_requires_deliver
from agent.task_state import infer_phase_from_step
from agent.task_runtime import is_delivery_verification_step
from config.app_config import (
    AGENT_TASK_HARNESS,
    AGENT_TASK_HARNESS_MIN_SIGNALS,
)
from llm.llm_zhipu import make_summary_llm

logger = logging.getLogger("ai_chat.harness.planner")

PLANNER_SYSTEM = (
    "你是任务规划助手。将用户目标拆解为 3～5 个可执行步骤，按先后顺序排列。\n"
    "要求：\n"
    "1. 信息收集（读文件/搜索/查天气）在前，整理汇总居中，外发/导出（微信/邮件/Excel）在最后；\n"
    "2. 同阶段的多个搜索合并为一个批量收集步骤，多个整理动作合并为一个内容合成步骤；\n"
    "3. 发送邮件/微信或导出必须是最后一步，工具成功即代表任务完成，不要再生成‘确认发送成功’步骤；\n"
    "4. 每步一句话中文，可执行、可验证；\n"
    "5. 只输出 JSON 数组，如 [\"步骤1\", \"步骤2\"]，不要 markdown 或其它文字。"
)

_MULTI_STEP_KEYWORDS = (
    "然后",
    "接着",
    "一并",
    "全部",
    "汇总",
    "再",
    "先",
    "并且",
    "同时",
    "之后",
    "最后",
)
_TOOL_CHAIN_KEYWORDS = (
    "发微信",
    "发邮件",
    "发给",
    "发送到",
    "导出",
    "读取",
    "目录",
    "搜索",
    "联网",
    "表格",
    "邮件",
    "微信",
    "天气",
)
_EMAIL_PATTERN = re.compile(
    r"[@＠]\s*\w+|@\w+\.\w+|发给|发邮件|发送到|send_email|\.com|\.cn|\.qq",
    re.IGNORECASE,
)


def task_harness_enabled() -> bool:
    return AGENT_TASK_HARNESS


def needs_task_harness(user_goal: str, *, file_count: int = 0) -> bool:
    """启发式判断是否需要 Task Harness（避免简单问答也走规划）。"""
    if not task_harness_enabled():
        return False
    text = (user_goal or "").strip()
    if not text:
        return False
    signals = 0
    if file_count > 0:
        signals += 1
    if file_count > 1:
        signals += 1
    if len(text) > 60:
        signals += 1
    if len(text) > 120:
        signals += 1

    kw_hits = sum(1 for k in _MULTI_STEP_KEYWORDS if k in text)
    if kw_hits >= 2:
        signals += 2
    elif kw_hits >= 1:
        signals += 1

    tool_hits = sum(1 for k in _TOOL_CHAIN_KEYWORDS if k in text)
    if tool_hits >= 3:
        signals += 3
    elif tool_hits >= 2:
        signals += 2
    elif tool_hits >= 1:
        signals += 1

    # 邮件/外发类任务一律视为复杂
    if _EMAIL_PATTERN.search(text):
        signals += 2

    # 「联网/搜索」+「发/邮件/微信/表格」组合
    has_search = any(k in text for k in ("搜索", "联网", "查", "天气"))
    has_deliver = any(k in text for k in ("发", "邮件", "微信", "导出", "表格"))
    if has_search and has_deliver:
        signals += 2

    # 「并且」+ 任一工具意图
    if "并且" in text and tool_hits >= 1:
        signals += 2

    # 纯外发（仅发邮件/微信/导出，无需先搜索读文件）不走分阶段 Harness，
    # 否则 plan 仍在 gather 阶段会拦截 send_email 等工具。
    if user_goal_requires_deliver(text) and not goal_requires_gather(text, file_count=file_count):
        return False

    return signals >= AGENT_TASK_HARNESS_MIN_SIGNALS


def _parse_plan_json(raw: str) -> list[str]:
    text = (raw or "").strip()
    if not text:
        return []
    # 去掉 markdown code fence
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        # 尝试提取 [...] 片段
        m = re.search(r"\[.*\]", text, re.DOTALL)
        if not m:
            return _fallback_plan(text)
        try:
            data = json.loads(m.group(0))
        except json.JSONDecodeError:
            return _fallback_plan(text)
    if not isinstance(data, list):
        return _fallback_plan(text)
    steps = [str(x).strip() for x in data if str(x).strip()]
    return steps[:7] if steps else _fallback_plan(text)


def _fallback_plan(user_goal: str) -> list[str]:
    goal = (user_goal or "").strip()
    if not goal:
        return ["理解用户需求", "执行必要操作", "向用户汇报结果"]
    return [
        f"理解并完成：{goal[:120]}",
        "整理中间结果",
        "完成交付并向用户说明",
    ]


def compact_task_plan(steps: list[str]) -> list[str]:
    """Remove redundant verification and merge adjacent same-phase work.

    Delivery is never truncated: it is the user-visible goal, not optional cleanup.
    """
    compacted: list[str] = []
    phases: list[str] = []
    for raw in steps:
        step = str(raw or "").strip()
        if not step:
            continue
        if is_delivery_verification_step(step):
            # The real delivery ToolExecutionEvent already verifies success.
            continue
        phase = infer_phase_from_step(step)
        if compacted and phase == phases[-1] and phase in ("gather", "process"):
            compacted[-1] = f"{compacted[-1]}；{step}"
            continue
        compacted.append(step)
        phases.append(phase)
    if len(compacted) <= 5:
        return compacted
    delivery = [step for step in compacted if infer_phase_from_step(step) == "deliver"]
    if delivery:
        preparation = [
            step for step in compacted if infer_phase_from_step(step) != "deliver"
        ]
        return [*preparation[:4], "；".join(delivery)]
    return compacted[:5]


def ensure_plan_has_delivery(
    steps: list[str], user_goal: str, *, max_steps: int | None = 5
) -> list[str]:
    """Repair planner output/checkpoints that accidentally omit requested delivery."""
    plan = list(steps)
    if not user_goal_requires_deliver(user_goal):
        return plan
    if any(infer_phase_from_step(step) == "deliver" for step in plan):
        return plan

    goal = (user_goal or "").lower()
    actions: list[str] = []
    if any(k in goal for k in ("邮件", "邮箱", "email", "@")):
        actions.append("发送邮件到指定邮箱")
    if "微信" in goal:
        actions.append("发送微信消息或文件给指定联系人")
    if any(k in goal for k in ("excel", "xlsx", "导出")):
        actions.append("导出结果到Excel文件")
    delivery = "；".join(actions) or "完成用户要求的外发交付"

    if max_steps is not None and len(plan) >= max_steps:
        return [*plan[: max_steps - 1], delivery]
    return [*plan, delivery]


async def build_task_plan(user_goal: str) -> list[str]:
    """用小模型生成步骤计划。"""
    goal = (user_goal or "").strip()
    if not goal:
        return _fallback_plan(goal)
    try:
        llm = make_summary_llm()
        resp = await llm.ainvoke(
            [
                SystemMessage(content=PLANNER_SYSTEM),
                HumanMessage(content=f"用户目标：\n{goal}"),
            ]
        )
        content = resp.content
        if isinstance(content, list):
            parts = []
            for block in content:
                if isinstance(block, dict) and block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
            content = "\n".join(parts)
        steps = compact_task_plan(_parse_plan_json(str(content or "")))
        steps = ensure_plan_has_delivery(steps, goal)
        if steps:
            return steps
    except Exception as e:
        logger.warning("任务规划失败，使用默认计划：%s", e)
    return _fallback_plan(goal)


def infer_initial_phase(plan: list[str]) -> str:
    if not plan:
        return "gather"
    return infer_phase_from_step(plan[0])


def format_plan_for_display(plan: list[str], *, plan_index: int = 0) -> str:
    if not plan:
        return ""
    lines = ["【执行计划】"]
    for i, step in enumerate(plan):
        mark = "▶" if i == plan_index else ("✓" if i < plan_index else "○")
        lines.append(f"{mark} {i + 1}. {step}")
    return "\n".join(lines)
