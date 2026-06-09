"""对话消息构建、MCP 附件与工具调试辅助。"""
import json
import re
from pathlib import Path

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage

import chat_store
from app_config import EXPORTS_DIR, MCP_TABLE_ATTACH_MAX


def dict_history_to_lc_messages(rows: list) -> list:
    out = []
    for m in rows:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    return out


def build_user_message_text(
    text: str,
    file_ids: list[str],
    session_id: str,
    *,
    omit_attachment_body: bool = False,
) -> str:
    parts: list[str] = []
    if text.strip():
        parts.append(text.strip())
    for meta in chat_store.get_upload_metas(session_id, file_ids):
        name = meta.get("name") or meta.get("file_id") or "附件"
        kind = meta.get("kind") or "file"
        if omit_attachment_body:
            parts.append(
                f"\n\n--- 附件 [{kind}] {name} ---\n"
                "（正文已写入知识库，相关内容见系统检索结果，请勿重复粘贴全文。）"
            )
            continue
        parsed = (meta.get("parsed_text") or "").strip()
        if not parsed:
            parsed = "（附件解析结果为空）"
        cap = 100_000
        if len(parsed) > cap:
            parsed = parsed[:cap] + "\n…(附件内容已截断)"
        parts.append(f"\n\n--- 附件 [{kind}] {name} ---\n{parsed}")
    return "\n".join(parts).strip()


def upload_meta_for_message(file_ids: list[str], session_id: str) -> list[dict]:
    items = []
    for meta in chat_store.get_upload_metas(session_id, file_ids):
        preview = (meta.get("parsed_text") or "")[:500]
        items.append(
            {
                "file_id": meta.get("file_id"),
                "name": meta.get("name"),
                "kind": meta.get("kind"),
                "preview": preview,
            }
        )
    return items


def last_assistant_text(messages: list) -> str:
    for m in reversed(messages):
        if isinstance(m, AIMessage) and getattr(m, "content", None):
            c = m.content
            if isinstance(c, str) and c.strip():
                return c.strip()
            if isinstance(c, list):
                parts = []
                for block in c:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                if parts:
                    return "\n".join(parts).strip()
    last = messages[-1] if messages else None
    if last is not None and getattr(last, "content", None):
        return str(last.content).strip()
    return ""


def table_display_kind(content: str) -> str:
    s = (content or "").strip()
    if not s:
        return "text"
    sl = s.lower()
    if sl.startswith("<table") or sl.startswith("<!doctype html"):
        return "html"
    return "text"


def excel_basename_from_export_tool(content: str) -> str | None:
    if not isinstance(content, str):
        return None
    m = re.search(r"Excel\s*已保存\s*[：:]\s*(.+)", content, re.I | re.DOTALL)
    if not m:
        return None
    raw = m.group(1).strip().strip('`"\'').splitlines()[0].strip()
    try:
        p = Path(raw).expanduser().resolve()
        p.relative_to(EXPORTS_DIR)
    except (ValueError, OSError):
        return None
    name = p.name
    if not name.lower().endswith(".xlsx"):
        return None
    return name


def extract_mcp_attachments_from_messages(messages: list) -> list[dict]:
    """从本轮 ToolMessage 提取 format_pretty_table / export_to_excel，供前端展示与下载。"""
    out: list[dict] = []
    tbl_idx = 0
    for m in messages:
        if not isinstance(m, ToolMessage):
            continue
        name = m.name or ""
        c = m.content
        if not isinstance(c, str):
            try:
                c = json.dumps(c, ensure_ascii=False)
            except Exception:
                c = str(c)
        if name == "format_pretty_table":
            tbl_idx += 1
            body = c[:MCP_TABLE_ATTACH_MAX]
            if len(c) > MCP_TABLE_ATTACH_MAX:
                body += "\n…(内容已截断，完整版请重新生成或缩小表格)"
            out.append(
                {
                    "type": "table",
                    "label": f"MCP 表格 {tbl_idx}",
                    "format": table_display_kind(body),
                    "content": body,
                }
            )
        elif name == "export_to_excel":
            fn = excel_basename_from_export_tool(c)
            if fn:
                out.append({"type": "excel", "label": fn, "filename": fn})
    return out


def build_tool_debug_from_messages(messages: list) -> dict:
    """汇总本轮 MCP 工具返回，便于区分模型选错工具与底层执行失败。"""
    items: list[dict] = []
    for m in messages:
        if isinstance(m, ToolMessage):
            c = m.content
            if not isinstance(c, str):
                try:
                    c = json.dumps(c, ensure_ascii=False)
                except Exception:
                    c = str(c)
            cap = 16000
            if len(c) > cap:
                c = c[:cap] + "\n…(已截断)"
            items.append({"name": (m.name or ""), "content": c})
    names = {x["name"] for x in items}
    return {
        "tools": items,
        "used_send_wechat_message": "send_wechat_message" in names,
        "used_send_wechat_files": "send_wechat_files" in names,
        "used_get_wechat_messages": "get_wechat_messages" in names,
        "used_send_email": "send_email" in names,
        "used_web_search": "web_search" in names,
        "used_export_to_excel": "export_to_excel" in names,
        "used_format_pretty_table": "format_pretty_table" in names,
        "used_list_local_directory": "list_local_directory" in names,
        "used_glob_local_files": "glob_local_files" in names,
        "used_read_local_file": "read_local_file" in names,
    }
