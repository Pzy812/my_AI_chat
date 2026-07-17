from __future__ import annotations

from evals.graders import grade_case
from evals.schemas import EvalCase


def _case() -> EvalCase:
    return EvalCase.from_dict(
        {
            "id": "email",
            "category": "email",
            "input": "搜索后发邮件",
            "plan": [],
            "expected": {
                "required_tools": ["web_search", "send_email"],
                "required_order": [["web_search", "send_email"]],
                "exact_tool_counts": {"send_email": 1},
                "tool_argument_checks": {"send_email.to_email": "a@example.com"},
            },
        }
    )


def test_grader_accepts_correct_trajectory():
    trajectory = [
        {"name": "web_search", "arguments": {}, "success": True},
        {
            "name": "send_email",
            "arguments": {"to_email": "a@example.com"},
            "success": True,
        },
    ]
    score, failures = grade_case(_case(), trajectory, "邮件已发送")
    assert score == 100
    assert failures == []


def test_grader_detects_early_and_duplicate_delivery():
    trajectory = [
        {"name": "send_email", "arguments": {"to_email": "a@example.com"}, "success": True},
        {"name": "send_email", "arguments": {"to_email": "a@example.com"}, "success": True},
    ]
    _, failures = grade_case(_case(), trajectory, "邮件已发送")
    assert "missing_required_tool" in failures
    assert "wrong_tool_order" in failures
    assert "duplicate_delivery" in failures
