"""为敏感 MCP 工具包装 LangGraph interrupt（Human-in-the-Loop）。"""
from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import BaseTool, StructuredTool
from langgraph.types import interrupt

from agent.hitl_config import HITL_TOOL_LABELS, HITL_TOOL_NAMES, hitl_tool_label
from agent.task_continue import (
    DELIVER_ACTION_TOOLS,
    deliver_duplicate_block_message,
    is_deliver_success_text,
    is_deliver_tool_done,
    mark_deliver_tool_done,
)


def _parse_decision(decision: Any) -> str:
    if isinstance(decision, dict):
        return str(decision.get("action") or "reject").strip().lower()
    if decision is True or decision == "approve":
        return "approve"
    if isinstance(decision, str):
        return decision.strip().lower()
    return "reject"


def normalize_interrupts(raw: Any) -> list[dict[str, Any]]:
    """将 LangGraph __interrupt__ 转为前端可展示的结构。"""
    if raw is None:
        return []
    items = raw if isinstance(raw, (list, tuple)) else [raw]
    out: list[dict[str, Any]] = []
    for item in items:
        val = getattr(item, "value", item)
        if isinstance(val, dict) and val.get("type") == "tool_approval":
            out.append(
                {
                    "tool": val.get("tool") or "",
                    "label": val.get("label") or hitl_tool_label(val.get("tool") or ""),
                    "args": val.get("args") or {},
                    "summary": _args_summary(val.get("tool") or "", val.get("args") or {}),
                }
            )
        elif isinstance(val, dict):
            out.append(
                {
                    "tool": val.get("tool") or "",
                    "label": val.get("label") or hitl_tool_label(val.get("tool") or ""),
                    "args": val.get("args") or val,
                    "summary": _args_summary(val.get("tool") or "", val.get("args") or val),
                }
            )
    return out


def _args_summary(tool: str, args: dict[str, Any]) -> str:
    if tool == "send_wechat_message":
        return f"收件人：{args.get('to_name', '')} · 内容：{_clip(str(args.get('content', '')))}"
    if tool == "send_wechat_files":
        paths = args.get("file_paths") or []
        names = ", ".join(str(p) for p in paths[:5])
        if len(paths) > 5:
            names += f" 等 {len(paths)} 个"
        return f"收件人：{args.get('to_name', '')} · 文件：{names or '(无)'}"
    if tool == "send_email":
        return f"收件：{args.get('to_email', '')} · 内容：{_clip(str(args.get('content', '')))}"
    if tool in ("format_pretty_table", "export_to_excel"):
        headers = args.get("headers") or []
        rows = args.get("rows") or []
        fn = args.get("filename") or ""
        parts = [f"列：{headers}", f"行数：{len(rows)}"]
        if fn:
            parts.append(f"文件名：{fn}")
        return " · ".join(parts)
    try:
        return _clip(json.dumps(args, ensure_ascii=False))
    except Exception:
        return _clip(str(args))


def _clip(s: str, n: int = 200) -> str:
    s = (s or "").strip()
    return s if len(s) <= n else s[:n] + "…"


def wrap_tools_with_hitl(tools: list[BaseTool], *, enabled: bool) -> list[BaseTool]:
    if not enabled:
        return tools
    wrapped: list[BaseTool] = []
    for tool in tools:
        if tool.name in HITL_TOOL_NAMES:
            wrapped.append(_wrap_one(tool))
        else:
            wrapped.append(tool)
    return wrapped


def _wrap_one(tool: BaseTool) -> BaseTool:
    label = HITL_TOOL_LABELS.get(tool.name, tool.name)

    async def _hitl_coroutine(**kwargs: Any) -> str:
        from langchain_core.runnables import ensure_config

        from agent.harness import _thread_id_from_config

        config = ensure_config()
        thread_id = _thread_id_from_config(config)
        if tool.name in DELIVER_ACTION_TOOLS and is_deliver_tool_done(thread_id, tool.name):
            return deliver_duplicate_block_message(tool.name)

        payload = {
            "type": "tool_approval",
            "tool": tool.name,
            "label": label,
            "args": kwargs,
        }
        decision = interrupt(payload)
        if _parse_decision(decision) != "approve":
            return f"⏸️ 用户已取消：{label}（未执行）"
        result = await tool.ainvoke(kwargs)
        text = result if isinstance(result, str) else str(result)
        if tool.name in DELIVER_ACTION_TOOLS and is_deliver_success_text(text):
            mark_deliver_tool_done(thread_id, tool.name)
        return text

    return StructuredTool(
        name=tool.name,
        description=tool.description or label,
        args_schema=tool.args_schema,
        coroutine=_hitl_coroutine,
    )
