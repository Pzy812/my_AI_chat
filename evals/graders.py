from __future__ import annotations

from collections import Counter
from typing import Any

from evals.schemas import EvalCase


def _ordered(names: list[str], required_order: list[list[str]]) -> bool:
    for chain in required_order:
        cursor = -1
        for required in chain:
            try:
                cursor = names.index(required, cursor + 1)
            except ValueError:
                return False
    return True


def grade_case(case: EvalCase, trajectory: list[dict[str, Any]], answer: str) -> tuple[float, list[str]]:
    expected = case.expected
    names = [str(call.get("name") or "") for call in trajectory]
    executed_names = [
        str(call.get("name") or "")
        for call in trajectory
        if call.get("executed", True) and call.get("success")
    ]
    counts = Counter(executed_names)
    failures: list[str] = []

    required = list(expected.get("required_tools") or [])
    forbidden = list(expected.get("forbidden_tools") or [])
    if any(counts[name] == 0 for name in required):
        failures.append("missing_required_tool")
    if any(counts[name] > 0 for name in forbidden):
        failures.append("forbidden_tool_called")
    if not _ordered(executed_names, list(expected.get("required_order") or [])):
        failures.append("wrong_tool_order")
    if len(names) > int(expected.get("max_tool_calls") or 999):
        failures.append("excessive_tool_calls")
    for name, wanted in (expected.get("exact_tool_counts") or {}).items():
        if counts[name] != int(wanted):
            failures.append("duplicate_delivery" if name == "send_email" and counts[name] > 1 else "wrong_tool_count")

    arg_checks = expected.get("tool_argument_checks") or {}
    for path, wanted in arg_checks.items():
        tool_name, _, arg_name = str(path).partition(".")
        matching = [
            c
            for c in trajectory
            if c.get("name") == tool_name
            and c.get("executed", True)
            and c.get("success")
        ]
        if not matching or all((c.get("arguments") or {}).get(arg_name) != wanted for c in matching):
            failures.append("wrong_tool_arguments")

    lower_answer = (answer or "").lower()
    if any(str(text).lower() not in lower_answer for text in expected.get("must_contain") or []):
        failures.append("missing_answer_fact")
    if any(str(text).lower() in lower_answer for text in expected.get("must_not_contain") or []):
        failures.append("forbidden_answer_content")
    if any(not bool(c.get("success")) for c in trajectory) and any(
        marker in answer for marker in ("已发送", "已保存", "已成功")
    ):
        last_effect = [c for c in trajectory if c.get("name") in ("send_email", "export_to_excel")]
        if last_effect and not last_effect[-1].get("success"):
            failures.append("false_success_claim")

    failures = list(dict.fromkeys(failures))
    penalties = {
        "missing_required_tool": 30,
        "forbidden_tool_called": 35,
        "wrong_tool_order": 20,
        "wrong_tool_arguments": 20,
        "duplicate_delivery": 40,
        "wrong_tool_count": 15,
        "excessive_tool_calls": 10,
        "missing_answer_fact": 15,
        "forbidden_answer_content": 30,
        "false_success_claim": 40,
    }
    score = max(0.0, 100.0 - sum(penalties.get(f, 10) for f in failures))
    critical = {"forbidden_tool_called", "duplicate_delivery", "false_success_claim"}
    success = score >= 80 and not critical.intersection(failures)
    return (score if success or failures else 100.0), failures
