"""Task Harness 启发式检测测试。"""
from __future__ import annotations

from agent.planner import needs_task_harness


def test_pure_email_task_skips_harness():
    goal = "给 971662861@qq.com 发送生日祝福语"
    assert needs_task_harness(goal) is False


def test_search_then_email_uses_harness():
    goal = "搜索上海嘉定天气并整理后发送到 971662861@qq.com"
    assert needs_task_harness(goal) is True


def test_wechat_only_skips_harness():
    assert needs_task_harness("发微信给张三：生日快乐") is False


def test_attachments_force_harness():
    assert needs_task_harness("把附件内容发邮件给 a@b.com", file_count=1) is True
