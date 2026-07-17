from __future__ import annotations

from agent.task_runtime import (
    build_step_states,
    evaluate_progress,
    make_tool_event,
)


def _event(tool: str, step_id: str, *, success: bool = True, output: str = "ok"):
    return make_tool_event(
        tool_name=tool,
        arguments={"q": "test"},
        success=success,
        output=output,
        latency_ms=12,
        step_id=step_id,
    )


def test_success_event_advances_matching_tool_step():
    plan = ["搜索产品A", "总结结果"]
    progress = evaluate_progress(
        plan,
        build_step_states(plan),
        [],
        [_event("web_search", "step-1")],
    )
    assert progress["plan_index"] == 1
    assert progress["step_states"][0]["status"] == "succeeded"
    assert progress["step_states"][1]["status"] == "running"


def test_failed_tool_does_not_advance_and_retry_can_recover():
    plan = ["搜索产品A"]
    first = _event("web_search", "step-1", success=False, output="请求超时")
    failed = evaluate_progress(plan, build_step_states(plan), [], [first])
    assert failed["plan_index"] == 0
    assert failed["step_states"][0]["status"] == "failed"
    second = _event("web_search", "step-1", success=True)
    recovered = evaluate_progress(
        plan,
        failed["step_states"],
        failed["tool_events"],
        [first, second],
    )
    assert recovered["plan_index"] == 1
    assert recovered["task_status"] == "done"
    assert recovered["step_states"][0]["attempts"] == 2


def test_reasoning_step_requires_explicit_completion_event():
    plan = ["分析已有数据并总结结论"]
    unrelated = _event("web_search", "step-1")
    waiting = evaluate_progress(plan, build_step_states(plan), [], [unrelated])
    assert waiting["plan_index"] == 0
    complete = _event("mark_step_complete", "step-1")
    done = evaluate_progress(
        plan,
        waiting["step_states"],
        waiting["tool_events"],
        [unrelated, complete],
    )
    assert done["plan_index"] == 1
    assert done["task_status"] == "done"


def test_combined_step_requires_all_expected_tool_evidence():
    plan = ["搜索产品A并整理成表格"]
    searched = evaluate_progress(
        plan,
        build_step_states(plan),
        [],
        [_event("web_search_batch", "step-1")],
    )
    assert searched["plan_index"] == 0
    formatted = _event("format_pretty_table", "step-1")
    done = evaluate_progress(
        plan,
        searched["step_states"],
        searched["tool_events"],
        [formatted],
    )
    assert done["plan_index"] == 1


def test_policy_rejection_is_not_a_business_step_failure():
    plan = ["搜索产品A"]
    blocked = make_tool_event(
        tool_name="send_email",
        arguments={},
        success=False,
        output="当前阶段不可用",
        latency_ms=0,
        step_id="step-1",
        executed=False,
    )
    progress = evaluate_progress(plan, build_step_states(plan), [], [blocked])
    assert progress["plan_index"] == 0
    assert progress["step_states"][0]["status"] == "running"
    assert progress["step_states"][0]["attempts"] == 0
    assert progress["step_states"][0]["error"] is None


def test_delivery_success_also_completes_following_confirmation_step():
    plan = ["发送邮件到指定邮箱", "确认邮件已成功发送"]
    sent = make_tool_event(
        tool_name="send_email",
        arguments={"to_email": "a@example.com"},
        success=True,
        output="邮件已发送",
        latency_ms=10,
        step_id="step-1",
    )
    progress = evaluate_progress(plan, build_step_states(plan), [], [sent])
    assert progress["plan_index"] == 2
    assert progress["task_status"] == "done"
    assert [s["status"] for s in progress["step_states"]] == [
        "succeeded",
        "succeeded",
    ]
