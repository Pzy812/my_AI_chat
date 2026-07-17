"""Event-driven task runtime for the Task Harness.

The runtime deliberately keeps JSON-serialisable state.  LangGraph checkpoints and
the Redis continuation metadata can therefore persist the same StepState and tool
events without a second object model.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
from threading import Lock
from typing import Any, Literal, NotRequired, TypedDict
from uuid import uuid4


StepStatus = Literal["pending", "running", "succeeded", "failed", "skipped"]


class ToolResultEvidence(TypedDict):
    event_id: str
    tool_name: str
    success: bool
    output_summary: str
    latency_ms: int


class StepState(TypedDict):
    id: str
    index: int
    description: str
    status: StepStatus
    attempts: int
    evidence: list[ToolResultEvidence]
    error: str | None
    expected_tools: NotRequired[list[str]]


class ToolExecutionEvent(TypedDict):
    id: str
    tool_name: str
    arguments: dict[str, Any]
    success: bool
    output_summary: str
    latency_ms: int
    step_id: str
    created_at: str
    executed: NotRequired[bool]


CONTROL_COMPLETE_TOOL = "mark_step_complete"
_OUTPUT_PREVIEW = 1200
_runtime_events: dict[str, list[ToolExecutionEvent]] = {}
_events_lock = Lock()


def is_delivery_verification_step(description: str) -> bool:
    text = (description or "").strip().lower()
    verification = any(k in text for k in ("确认", "验证", "检查", "核实"))
    delivery = any(k in text for k in ("发送", "邮件", "导出", "交付"))
    completion = any(k in text for k in ("成功", "完成", "结果", "状态"))
    return verification and delivery and completion


def _clip(value: Any, cap: int = _OUTPUT_PREVIEW) -> str:
    text = str(value or "").strip()
    return text if len(text) <= cap else text[:cap] + "…"


def _argument_preview(value: Any, depth: int = 0) -> Any:
    if depth >= 4:
        return _clip(value, 300)
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _clip(value, 1000)
    if isinstance(value, dict):
        return {
            str(k): _argument_preview(v, depth + 1)
            for k, v in list(value.items())[:40]
        }
    if isinstance(value, (list, tuple)):
        return [_argument_preview(v, depth + 1) for v in list(value)[:50]]
    return _clip(value, 500)


def tool_output_succeeded(output: Any) -> bool:
    """Conservative success classification shared by the runtime and evals."""
    text = str(output or "").strip().lower()
    if not text:
        return False
    failure_markers = (
        "❌",
        "⛔",
        "失败",
        "请求超时",
        "未配置",
        "不存在",
        "permission denied",
        "error:",
        "traceback",
    )
    return not any(marker in text for marker in failure_markers)


def expected_tools_for_step(description: str) -> list[str]:
    """Infer completion evidence required by a planned step.

    These are evidence requirements, not a planner.  Multiple returned names mean
    that the step explicitly combines multiple operations and all must succeed.
    """
    text = (description or "").lower()
    expected: list[str] = []

    def add(name: str) -> None:
        if name not in expected:
            expected.append(name)

    search_action = any(k in text for k in ("搜索", "联网", "查询", "查找"))
    search_action = search_action or ("检索" in text and "检索结果" not in text and "检索证据" not in text)
    if search_action:
        add("web_search")
    if any(k in text for k in ("列出目录", "目录列表", "遍历目录", "查找文件")):
        add("list_local_directory")
    if any(k in text for k in ("glob", "匹配文件", "全部文件")):
        add("glob_local_files")
    if ("读取" in text or "读" in text) and any(k in text for k in ("文件", "附件")):
        add("read_local_file")
    if any(k in text for k in ("表格", "格式化")) and not any(
        k in text for k in ("excel", "xlsx", "导出")
    ):
        add("format_pretty_table")
    if any(k in text for k in ("excel", "xlsx", "导出")):
        add("export_to_excel")
    if any(k in text for k in ("邮件", "邮箱", "email")) and any(
        k in text for k in ("发", "发送", "投递", "交付")
    ):
        add("send_email")
    if any(k in text for k in ("当前时间", "日期", "星期几", "几点")):
        add("get_current_time")
    return expected


def build_step_states(plan: list[str]) -> list[StepState]:
    states: list[StepState] = []
    for index, description in enumerate(plan):
        states.append(
            {
                "id": f"step-{index + 1}",
                "index": index,
                "description": str(description),
                "status": "running" if index == 0 else "pending",
                "attempts": 0,
                "evidence": [],
                "error": None,
                "expected_tools": expected_tools_for_step(str(description)),
            }
        )
    return states


def normalize_step_states(plan: list[str], raw: Any) -> list[StepState]:
    """Hydrate old checkpoints and validate newer JSON state."""
    if not isinstance(raw, list) or len(raw) != len(plan):
        return build_step_states(plan)
    out: list[StepState] = []
    valid_statuses = {"pending", "running", "succeeded", "failed", "skipped"}
    for index, description in enumerate(plan):
        item = raw[index] if isinstance(raw[index], dict) else {}
        status = item.get("status")
        if status not in valid_statuses:
            status = "running" if index == 0 else "pending"
        evidence = item.get("evidence") if isinstance(item.get("evidence"), list) else []
        out.append(
            {
                "id": str(item.get("id") or f"step-{index + 1}"),
                "index": index,
                "description": str(description),
                "status": status,
                "attempts": max(0, int(item.get("attempts") or 0)),
                "evidence": [dict(e) for e in evidence if isinstance(e, dict)],
                "error": str(item.get("error")) if item.get("error") else None,
                "expected_tools": list(
                    item.get("expected_tools") or expected_tools_for_step(str(description))
                ),
            }
        )
    _activate_first_incomplete(out)
    return out


def current_step_index(step_states: list[StepState]) -> int:
    for index, step in enumerate(step_states):
        if step["status"] not in ("succeeded", "skipped"):
            return index
    return len(step_states)


def _activate_first_incomplete(step_states: list[StepState]) -> None:
    current = current_step_index(step_states)
    for index, step in enumerate(step_states):
        if index == current and step["status"] == "pending":
            step["status"] = "running"


def canonical_tool_name(name: str) -> str:
    if name == "web_search_batch":
        return "web_search"
    return name


def _step_has_required_evidence(step: StepState) -> bool:
    expected = {canonical_tool_name(n) for n in step.get("expected_tools") or []}
    if not expected:
        return False
    succeeded = {
        canonical_tool_name(str(e.get("tool_name") or ""))
        for e in step.get("evidence") or []
        if e.get("success")
    }
    return expected.issubset(succeeded)


def make_tool_event(
    *,
    tool_name: str,
    arguments: dict[str, Any],
    success: bool,
    output: Any,
    latency_ms: int,
    step_id: str,
    executed: bool = True,
) -> ToolExecutionEvent:
    return {
        "id": uuid4().hex,
        "tool_name": tool_name,
        "arguments": _argument_preview(arguments),
        "success": bool(success),
        "output_summary": _clip(output),
        "latency_ms": max(0, int(latency_ms)),
        "step_id": step_id,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "executed": bool(executed),
    }


def record_runtime_event(thread_id: str, event: ToolExecutionEvent) -> None:
    tid = (thread_id or "default").strip() or "default"
    with _events_lock:
        bucket = _runtime_events.setdefault(tid, [])
        bucket.append(deepcopy(event))
        if len(bucket) > 200:
            del bucket[:-200]


def runtime_events(thread_id: str) -> list[ToolExecutionEvent]:
    tid = (thread_id or "default").strip() or "default"
    with _events_lock:
        return deepcopy(_runtime_events.get(tid, []))


def clear_runtime_events(thread_id: str) -> None:
    tid = (thread_id or "default").strip() or "default"
    with _events_lock:
        _runtime_events.pop(tid, None)


def evaluate_progress(
    plan: list[str],
    raw_step_states: Any,
    persisted_events: Any,
    new_events: list[ToolExecutionEvent],
) -> dict[str, Any]:
    """Fold tool events into StepState and return a complete state patch."""
    steps = normalize_step_states(plan, raw_step_states)
    events = [dict(e) for e in (persisted_events or []) if isinstance(e, dict)]
    seen = {str(e.get("id") or "") for e in events}
    by_id = {step["id"]: step for step in steps}

    for event in new_events:
        event_id = str(event.get("id") or "")
        if not event_id or event_id in seen:
            continue
        event_copy = deepcopy(event)
        events.append(event_copy)
        seen.add(event_id)
        # Policy/phase rejection means the requested tool never ran. Keep it in
        # the trajectory, but do not fail the current business step or count an
        # execution attempt.
        if event.get("executed") is False:
            continue
        step = by_id.get(str(event.get("step_id") or ""))
        if step is None:
            current = current_step_index(steps)
            step = steps[current] if current < len(steps) else None
        if step is None:
            continue

        step["attempts"] += 1
        step["evidence"].append(
            {
                "event_id": event_id,
                "tool_name": str(event.get("tool_name") or ""),
                "success": bool(event.get("success")),
                "output_summary": _clip(event.get("output_summary")),
                "latency_ms": max(0, int(event.get("latency_ms") or 0)),
            }
        )
        if not event.get("success"):
            step["status"] = "failed"
            step["error"] = _clip(event.get("output_summary"), 500) or "工具执行失败"
            continue

        control_completes = (
            event.get("tool_name") == CONTROL_COMPLETE_TOOL
            and not (step.get("expected_tools") or [])
        )
        if control_completes or _step_has_required_evidence(step):
            step["status"] = "succeeded"
            step["error"] = None
        elif step["status"] == "pending":
            step["status"] = "running"

    # A successful side effect is itself the evidence for a following
    # "确认邮件已成功发送" step. Never ask the model to execute the side effect
    # again merely to confirm it.
    successful_events = [
        event
        for event in events
        if event.get("success") and event.get("executed", True)
    ]
    for step in steps:
        if step["status"] in ("succeeded", "skipped"):
            continue
        if not is_delivery_verification_step(step["description"]):
            continue
        expected = list(step.get("expected_tools") or [])
        if not expected and "邮件" in step["description"]:
            expected = ["send_email"]
        matches: list[dict[str, Any]] = []
        for required in expected:
            match = next(
                (
                    event
                    for event in successful_events
                    if canonical_tool_name(str(event.get("tool_name") or ""))
                    == canonical_tool_name(required)
                ),
                None,
            )
            if match is None:
                matches = []
                break
            matches.append(match)
        if not matches:
            continue
        known = {str(e.get("event_id") or "") for e in step["evidence"]}
        for match in matches:
            event_id = str(match.get("id") or "")
            if event_id in known:
                continue
            step["evidence"].append(
                {
                    "event_id": event_id,
                    "tool_name": str(match.get("tool_name") or ""),
                    "success": True,
                    "output_summary": "复用真实交付成功事件完成状态确认。",
                    "latency_ms": 0,
                }
            )
        step["status"] = "succeeded"
        step["error"] = None

    _activate_first_incomplete(steps)
    index = current_step_index(steps)
    completed = [s["description"] for s in steps if s["status"] in ("succeeded", "skipped")]
    checklist = [
        {
            "index": s["index"],
            "step": s["description"],
            "status": s["status"],
            "done": s["status"] in ("succeeded", "skipped"),
            "current": s["index"] == index and index < len(steps),
            "attempts": s["attempts"],
            "error": s["error"],
        }
        for s in steps
    ]
    # Bound checkpoint/Redis growth while retaining enough trajectory for debugging.
    events = events[-200:]
    return {
        "step_states": steps,
        "tool_events": events,
        "plan_index": index,
        "completed_steps": completed,
        "step_checklist": checklist,
        "task_status": "done" if plan and index >= len(plan) else "executing",
    }
