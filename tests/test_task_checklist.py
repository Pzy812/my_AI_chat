"""Task checklist 与续跑逻辑测试。"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.task_checklist import (
    assistant_promised_next_step,
    build_step_checklist,
    extract_primary_user_goal,
    is_continue_message,
    resolve_user_goal,
    should_continue_task,
)


def test_is_continue_message():
    assert is_continue_message("继续") is True
    assert is_continue_message("请继续。") is True
    assert is_continue_message("continue") is True
    assert is_continue_message("继续搜索天气") is False


def test_extract_primary_user_goal_after_continue():
    goal = "搜索天气并发送邮件到 a@b.com"
    messages = [
        HumanMessage(content=goal),
        AIMessage(content="已完成搜索"),
        HumanMessage(content="继续"),
    ]
    assert extract_primary_user_goal(messages) == goal


def test_resolve_user_goal_continue_flag():
    goal = "搜索并导出 Excel"
    messages = [HumanMessage(content=goal), HumanMessage(content="继续")]
    resolved, is_continue = resolve_user_goal(messages)
    assert is_continue is True
    assert resolved == goal


def test_build_step_checklist_marks_progress():
    checklist = build_step_checklist(["A", "B", "C"], plan_index=1)
    assert checklist[0]["done"] is True
    assert checklist[1]["current"] is True
    assert checklist[2]["done"] is False


def test_assistant_promised_next_step():
    assert assistant_promised_next_step("接下来我将搜索最新论文") is True
    assert assistant_promised_next_step("今天天气不错") is False


def test_should_continue_task_when_plan_incomplete():
    state = {
        "harness_enabled": True,
        "user_goal": "搜索并发邮件到 x@y.com",
        "plan": ["搜索", "整理", "发邮件"],
        "plan_index": 0,
        "task_phase": "gather",
    }
    messages = [HumanMessage(content=state["user_goal"]), AIMessage(content="我先整理一下")]
    nudge = should_continue_task(state, messages, "我先整理一下，接下来将搜索相关内容")
    assert nudge is not None
    assert "系统自动续跑" in nudge


def test_should_not_continue_when_deliver_done():
    state = {
        "harness_enabled": True,
        "user_goal": "发邮件到 x@y.com",
        "plan": ["发邮件"],
        "plan_index": 0,
    }
    messages = [
        HumanMessage(content=state["user_goal"]),
        AIMessage(content="", tool_calls=[{"id": "1", "name": "send_email", "args": {}}]),
        ToolMessage(content="邮件已发送到 x@y.com", tool_call_id="1", name="send_email"),
        AIMessage(content="邮件已发送"),
    ]
    nudge = should_continue_task(state, messages, "邮件已发送")
    assert nudge is None


def test_should_not_continue_stale_plan_after_early_delivery():
    state = {
        "harness_enabled": True,
        "user_goal": "搜索天气、整理资料并发邮件到 x@y.com",
        "plan": ["搜索天气", "搜索资料", "整理内容", "撰写邮件", "发邮件", "确认发送"],
        "plan_index": 2,
        "task_phase": "process",
    }
    messages = [
        HumanMessage(content=state["user_goal"]),
        AIMessage(content="", tool_calls=[{"id": "1", "name": "send_email", "args": {}}]),
        ToolMessage(content="✅ 邮件已发送到 x@y.com", tool_call_id="1", name="send_email"),
        AIMessage(content="邮件已经发送完成。"),
    ]
    assert should_continue_task(state, messages, "邮件已经发送完成。") is None
