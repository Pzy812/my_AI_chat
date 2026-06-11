"""Human-in-the-Loop：需用户确认后才执行的工具。"""
from __future__ import annotations

HITL_TOOL_NAMES: frozenset[str] = frozenset(
    {
        "send_wechat_message",
        "send_wechat_files",
        "send_email",
        "format_pretty_table",
        "export_to_excel",
    }
)

HITL_TOOL_LABELS: dict[str, str] = {
    "send_wechat_message": "发送微信消息",
    "send_wechat_files": "发送微信文件",
    "send_email": "发送邮件",
    "format_pretty_table": "生成表格",
    "export_to_excel": "导出 Excel",
}


def hitl_tool_label(name: str) -> str:
    return HITL_TOOL_LABELS.get(name, name)


def hitl_payload_for_tool(name: str, args: dict) -> dict:
    """构造前端 HITL 卡片 payload。"""
    from agent.hitl_tools import _args_summary

    return {
        "type": "tool_approval",
        "tool": name,
        "label": hitl_tool_label(name),
        "args": args or {},
        "summary": _args_summary(name, args or {}),
    }
