"""外发重复检测与轮次边界测试。"""
from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

from agent.task_continue import (
    deliver_duplicate_block_message,
    deliver_tools_used,
    is_deliver_tool_done,
    messages_in_current_user_turn,
    sync_deliver_completion_flags,
)


def _human_message_text(msg: HumanMessage) -> str:
    content = msg.content
    return str(content or "").strip()


def test_messages_in_current_user_turn_skips_earlier_turns():
    messages = [
        HumanMessage(content="第一次：发天气邮件"),
        AIMessage(content="", tool_calls=[{"id": "1", "name": "send_email", "args": {}}]),
        ToolMessage(content="✅ 邮件已发送到：a@b.com", tool_call_id="1", name="send_email"),
        AIMessage(content="第一次已发送"),
        HumanMessage(content="第二次：发生日祝福"),
        AIMessage(content="准备发送"),
    ]
    turn = messages_in_current_user_turn(messages)
    assert turn[0].content == "第二次：发生日祝福"
    assert len(turn) == 2


def test_messages_in_current_user_turn_ignores_system_nudge():
    messages = [
        HumanMessage(content="发邮件给 x@y.com"),
        HumanMessage(content="【系统自动续跑】请继续执行"),
        AIMessage(content="继续中"),
    ]
    turn = messages_in_current_user_turn(messages)
    assert turn[0].content == "发邮件给 x@y.com"
    assert len(turn) == 3


def test_deliver_tools_used_only_in_current_turn():
    messages = [
        HumanMessage(content="第一次发邮件"),
        ToolMessage(content="✅ 邮件已发送到：a@b.com", tool_call_id="1", name="send_email"),
        HumanMessage(content="第二次发邮件"),
        AIMessage(content="正在准备"),
    ]
    assert deliver_tools_used(messages) is False


def test_deliver_tools_used_true_after_send_in_current_turn():
    messages = [
        HumanMessage(content="发邮件"),
        ToolMessage(content="✅ 邮件已发送到：a@b.com", tool_call_id="1", name="send_email"),
    ]
    assert deliver_tools_used(messages) is True


def test_sync_deliver_flags_scoped_to_current_turn():
    thread_id = "test-session"
    messages_turn1 = [
        HumanMessage(content="第一次"),
        ToolMessage(content="✅ 邮件已发送到：a@b.com", tool_call_id="1", name="send_email"),
    ]
    sync_deliver_completion_flags(thread_id, messages_turn1)
    assert is_deliver_tool_done(thread_id, "send_email") is True

    messages_turn2 = [
        HumanMessage(content="第一次"),
        ToolMessage(content="✅ 邮件已发送到：a@b.com", tool_call_id="1", name="send_email"),
        HumanMessage(content="第二次发生日祝福"),
        AIMessage(content="准备发送"),
    ]
    sync_deliver_completion_flags(thread_id, messages_turn2)
    assert is_deliver_tool_done(thread_id, "send_email") is False


def test_turn_serial_reset_allows_second_user_request():
    thread_id = "session-2"
    messages_turn1 = [
        HumanMessage(content="第一次发天气邮件"),
        ToolMessage(content="✅ 邮件已发送到：a@b.com", tool_call_id="1", name="send_email"),
    ]
    sync_deliver_completion_flags(thread_id, messages_turn1)
    assert is_deliver_tool_done(thread_id, "send_email") is True

    messages_turn2 = [
        HumanMessage(content="第一次发天气邮件"),
        ToolMessage(content="✅ 邮件已发送到：a@b.com", tool_call_id="1", name="send_email"),
        HumanMessage(content="第二次发生日祝福"),
    ]
    sync_deliver_completion_flags(thread_id, messages_turn2)
    assert is_deliver_tool_done(thread_id, "send_email") is False


def test_duplicate_block_message_not_success_tone():
    msg = deliver_duplicate_block_message("send_email")
    assert msg.startswith("⛔")
    assert "未实际执行" in msg


def test_summary_human_message_does_not_break_turn_boundary():
    messages = [
        HumanMessage(content="【此前对话摘要（系统自动生成，供理解上下文）】\n第一次已发邮件"),
        AIMessage(content="好的，我已了解此前对话要点。"),
        HumanMessage(content="请发送生日祝福"),
    ]
    turn = messages_in_current_user_turn(messages)
    assert _human_message_text(turn[0]) == "请发送生日祝福"
    assert deliver_tools_used(messages) is False
