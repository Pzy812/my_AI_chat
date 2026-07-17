"""Harness 纯函数测试（不依赖 MCP / LLM）。"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.harness import count_tool_rounds, finalize_task_after_delivery, trim_messages_for_llm
from agent.task_runtime import build_step_states
from agent.task_state import PHASE_GATE_EXEMPT_TOOLS, allowed_tools_for_phase


def test_count_tool_rounds():
    messages = [
        HumanMessage(content="查天气"),
        AIMessage(
            content="",
            tool_calls=[{"id": "tc1", "name": "web_search", "args": {"q": "天气"}}],
        ),
        ToolMessage(content="晴天", tool_call_id="tc1", name="web_search"),
    ]
    assert count_tool_rounds(messages) == 1


def test_count_tool_rounds_ignores_unanswered():
    messages = [
        AIMessage(
            content="",
            tool_calls=[{"id": "tc2", "name": "web_search", "args": {}}],
        ),
    ]
    assert count_tool_rounds(messages) == 0


def test_trim_messages_keeps_first_user_and_tail():
    first = HumanMessage(content="原始目标")
    filler = [AIMessage(content=f"reply {i}") for i in range(20)]
    messages = [first, *filler]
    trimmed = trim_messages_for_llm(messages, keep_recent=5)
    assert trimmed[0] is first
    assert len(trimmed) == 6


def test_deliver_tools_not_exempt_from_gather_phase_gate():
    gather_allowed = allowed_tools_for_phase("gather", harness_enabled=True)
    assert "send_email" not in gather_allowed
    assert "send_email" not in PHASE_GATE_EXEMPT_TOOLS
    assert "export_to_excel" not in PHASE_GATE_EXEMPT_TOOLS


def test_successful_early_delivery_finalizes_stale_plan():
    plan = ["搜索天气", "搜索研究", "整理邮件正文", "发送邮件", "确认邮件发送成功"]
    state = {
        "user_goal": "搜索天气和研究并发送邮件到 x@y.com",
        "plan": plan,
        "plan_index": 1,
        "step_states": build_step_states(plan),
    }
    messages = [
        HumanMessage(content=state["user_goal"]),
        ToolMessage(content="✅ 邮件已发送到 x@y.com", tool_call_id="1", name="send_email"),
    ]
    patch = finalize_task_after_delivery(state, messages)
    assert patch["plan_index"] == len(plan)
    assert patch["task_status"] == "done"
    assert all(item["done"] for item in patch["step_checklist"])
    assert patch["step_states"][1]["status"] == "skipped"
    assert patch["step_states"][3]["status"] == "succeeded"
    assert patch["step_states"][4]["status"] == "succeeded"
