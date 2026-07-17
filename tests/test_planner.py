"""Task Harness 启发式检测测试。"""
from __future__ import annotations

from agent.planner import compact_task_plan, ensure_plan_has_delivery, needs_task_harness


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


def test_compact_task_plan_merges_phases_and_removes_delivery_confirmation():
    plan = compact_task_plan(
        [
            "查询上海天气",
            "收集Agent Harness研究资料",
            "整理AI解决PDE的介绍内容",
            "将资料整理成邮件正文",
            "将PDE介绍添加到邮件中",
            "将邮件发送至a@example.com",
            "确认邮件已成功发送",
        ]
    )
    assert len(plan) == 3
    assert "查询上海天气" in plan[0]
    assert "Agent Harness" in plan[0]
    assert "邮件正文" in plan[1]
    assert plan[2] == "将邮件发送至a@example.com"


def test_compaction_never_truncates_delivery_step():
    plan = compact_task_plan(
        [
            "查询天气",
            "整理Harness资料",
            "搜索AI解决PDE介绍",
            "整理AI解决PDE介绍",
            "合成邮件内容",
            "发送邮件到指定邮箱",
            "确认邮件发送成功",
        ]
    )
    assert len(plan) <= 5
    assert plan[-1] == "发送邮件到指定邮箱"


def test_repair_old_plan_missing_delivery():
    old_plan = ["查询天气", "整理资料", "搜索PDE", "整理PDE", "合成邮件内容"]
    repaired = ensure_plan_has_delivery(
        old_plan, "整理并发送邮件到 971662861@qq.com", max_steps=None
    )
    assert len(repaired) == 6
    assert repaired[-1] == "发送邮件到指定邮箱"
