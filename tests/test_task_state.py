"""Task Harness 阶段与工具白名单测试。"""
from __future__ import annotations

from agent.task_state import (
    allowed_tools_for_phase,
    infer_phase_from_step,
)


def test_infer_phase_from_step():
    assert infer_phase_from_step("联网搜索最新新闻") == "gather"
    assert infer_phase_from_step("整理成表格并汇总") == "process"
    assert infer_phase_from_step("发送邮件给用户") == "deliver"
    assert infer_phase_from_step("将资料整理成邮件正文") == "process"
    assert infer_phase_from_step("把PDE介绍添加到邮件中") == "process"
    assert infer_phase_from_step("合成邮件内容，包括天气和研究资料") == "process"
    assert infer_phase_from_step("将邮件发送至用户邮箱") == "deliver"


def test_allowed_tools_gather_excludes_send_email():
    allowed = allowed_tools_for_phase("gather", harness_enabled=True)
    assert "web_search" in allowed
    assert "send_email" not in allowed


def test_allowed_tools_deliver_includes_send_email():
    allowed = allowed_tools_for_phase("deliver", harness_enabled=True)
    assert "send_email" in allowed
    assert "export_to_excel" in allowed


def test_harness_disabled_allows_all_deliver_tools():
    allowed = allowed_tools_for_phase("gather", harness_enabled=False)
    assert "send_email" in allowed
