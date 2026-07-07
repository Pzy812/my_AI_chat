"""Harness 纯函数测试（不依赖 MCP / LLM）。"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.harness import count_tool_rounds, trim_messages_for_llm
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


def test_deliver_tools_exempt_from_gather_phase_gate():
    gather_allowed = allowed_tools_for_phase("gather", harness_enabled=True)
    assert "send_email" not in gather_allowed
    assert "send_email" in PHASE_GATE_EXEMPT_TOOLS
    assert "export_to_excel" in PHASE_GATE_EXEMPT_TOOLS
