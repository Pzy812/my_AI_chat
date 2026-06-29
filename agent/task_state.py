"""Task Harness 结构化状态：goal / plan / phase / progress。"""
from __future__ import annotations

from collections.abc import Sequence
from typing import Annotated, Literal, NotRequired

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.managed import RemainingSteps
from typing_extensions import TypedDict

TaskPhase = Literal["gather", "process", "deliver"]
TaskStatus = Literal["planning", "executing", "done"]

# gather：读文件 / 搜索 / 查时间
GATHER_TOOLS: frozenset[str] = frozenset(
    {
        "get_current_time",
        "web_search",
        "web_search_batch",
        "get_wechat_messages",
        "list_local_directory",
        "glob_local_files",
        "read_local_file",
        "hello",
        "add",
    }
)

# process：整理、表格化（不含外发）
PROCESS_TOOLS: frozenset[str] = frozenset(GATHER_TOOLS | {"format_pretty_table"})

# deliver：外发 / 导出（HITL 工具在此阶段才允许）
DELIVER_TOOLS: frozenset[str] = frozenset(
    PROCESS_TOOLS
    | {
        "send_wechat_message",
        "send_wechat_files",
        "send_email",
        "export_to_excel",
    }
)

PHASE_ALLOWED_TOOLS: dict[TaskPhase, frozenset[str]] = {
    "gather": GATHER_TOOLS,
    "process": PROCESS_TOOLS,
    "deliver": DELIVER_TOOLS,
}

PHASE_LABELS: dict[TaskPhase, str] = {
    "gather": "信息收集（读文件/搜索/查时间）",
    "process": "整理加工（汇总/表格）",
    "deliver": "外发交付（微信/邮件/导出）",
}

DELIVER_KEYWORDS = ("发微信", "发送", "邮件", "导出", "export", "发给", "发到")
PROCESS_KEYWORDS = ("表格", "整理", "汇总", "格式化", "format", "排列")


class TaskHarnessState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    remaining_steps: NotRequired[RemainingSteps]
    user_goal: NotRequired[str]
    plan: NotRequired[list[str]]
    plan_index: NotRequired[int]
    task_phase: NotRequired[TaskPhase]
    harness_enabled: NotRequired[bool]
    completed_steps: NotRequired[list[str]]
    step_checklist: NotRequired[list[dict]]
    task_status: NotRequired[TaskStatus]


def allowed_tools_for_phase(phase: TaskPhase, *, harness_enabled: bool) -> frozenset[str]:
    if not harness_enabled:
        return DELIVER_TOOLS
    return PHASE_ALLOWED_TOOLS.get(phase, GATHER_TOOLS)


def infer_phase_from_step(step_text: str) -> TaskPhase:
    text = (step_text or "").strip()
    if any(k in text for k in DELIVER_KEYWORDS):
        return "deliver"
    if any(k in text for k in PROCESS_KEYWORDS):
        return "process"
    return "gather"


def default_task_fields() -> dict:
    return {
        "user_goal": "",
        "plan": [],
        "plan_index": 0,
        "task_phase": "gather",
        "harness_enabled": False,
        "completed_steps": [],
        "step_checklist": [],
        "task_status": "executing",
    }
