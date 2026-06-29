"""Task Harness 规划器启发式与 JSON 解析测试。"""
from __future__ import annotations

from agent.planner import (
    _parse_plan_json,
    format_plan_for_display,
    infer_initial_phase,
    needs_task_harness,
)


def test_simple_chat_does_not_need_harness():
    assert needs_task_harness("你好") is False
    assert needs_task_harness("解释一下 Python 的 list 和 tuple") is False


def test_multi_step_email_needs_harness():
    goal = "搜索最近一周上海天气，整理成表格，发到 test@example.com"
    assert needs_task_harness(goal) is True


def test_file_upload_increases_harness_signal():
    assert needs_task_harness("总结附件", file_count=1) is True


def test_parse_plan_json_array():
    raw = '["搜索天气", "整理表格", "发送邮件"]'
    assert _parse_plan_json(raw) == ["搜索天气", "整理表格", "发送邮件"]


def test_parse_plan_json_strips_markdown_fence():
    raw = '```json\n["步骤一", "步骤二"]\n```'
    assert _parse_plan_json(raw) == ["步骤一", "步骤二"]


def test_parse_plan_json_fallback_on_invalid():
    steps = _parse_plan_json("not json at all")
    assert len(steps) >= 1


def test_infer_initial_phase_gather():
    assert infer_initial_phase(["联网搜索 AI Agent 趋势"]) == "gather"


def test_infer_initial_phase_deliver():
    assert infer_initial_phase(["发送邮件给用户"]) == "deliver"


def test_format_plan_for_display_marks_current_step():
    text = format_plan_for_display(["A", "B", "C"], plan_index=1)
    assert "▶ 2. B" in text
    assert "✓ 1. A" in text
