import json
import logging
import re
import socket
import subprocess
import asyncio
import sys
import time
from pathlib import Path

from flask import Flask, render_template, request, jsonify, send_from_directory, abort
from flask_cors import CORS
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client
import os

import env_config
from env_config import ensure_zhipuai_api_key_in_environ

import chat_redis
import rag_service
from llm_zhipu import make_chat_llm
from file_upload import (
    ALLOWED_EXT,
    DOC_EXT,
    MAX_UPLOAD_BYTES,
    detect_kind,
    new_file_id,
    parse_uploaded_file,
    safe_filename,
)

BASE_DIR = Path(__file__).resolve().parent
EXPORTS_DIR = (BASE_DIR / "exports").resolve()
UPLOADS_DIR = (BASE_DIR / "uploads").resolve()
EXPORTS_DIR.mkdir(parents=True, exist_ok=True)
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
MCP_TABLE_ATTACH_MAX = 120_000
app = Flask(__name__, template_folder=str(BASE_DIR / "template"))
CORS(app)  # 开启跨域，彻底解决请求不到问题

# 打印当次请求的 System Prompt（含 GraphRAG 片段）。设为 0 可关闭。
LOG_LLM_PROMPT = os.getenv("LOG_LLM_PROMPT", "1").strip().lower() not in ("0", "false", "no")
LOG_LLM_PROMPT_MAX = int(os.getenv("LOG_LLM_PROMPT_MAX", "12000"))
logger = logging.getLogger("ai_chat")
if not logger.handlers:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s %(message)s",
    )

# 配置：密钥从项目根 .env 读取（见 env_config.py）
ensure_zhipuai_api_key_in_environ()
MCP_HOST = os.getenv("MCP_HOST", "localhost")
MCP_PORT = int(os.getenv("MCP_PORT", "8090"))
MCP_URL = f"http://{MCP_HOST}:{MCP_PORT}/mcp"

# 全局变量
mcp_process = None


def _format_error(e: BaseException) -> str:
    """展开 asyncio TaskGroup / ExceptionGroup，便于前端展示真实原因。"""
    if isinstance(e, BaseExceptionGroup):
        parts = [_format_error(x) for x in e.exceptions]
        joined = "; ".join(p for p in parts if p)
        return joined or str(e)
    return str(e).strip() or repr(e)


def _tracked_mcp_running() -> bool:
    return mcp_process is not None and mcp_process.poll() is None


def _mcp_port_pids() -> set[int]:
    """Return process ids listening on the MCP port, including old external runs."""
    try:
        out = subprocess.check_output(
            ["netstat", "-ano", "-p", "tcp"],
            text=True,
            errors="ignore",
        )
    except Exception:
        return set()

    pids: set[int] = set()
    needle = f":{MCP_PORT}"
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].upper() == "TCP" and parts[3].upper() == "LISTENING":
            if parts[1].endswith(needle):
                try:
                    pids.add(int(parts[-1]))
                except ValueError:
                    pass
    return pids


def _kill_mcp_port_processes() -> None:
    current_pid = os.getpid()
    for pid in _mcp_port_pids():
        if pid == current_pid:
            continue
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def _mcp_port_open() -> bool:
    try:
        with socket.create_connection((MCP_HOST, MCP_PORT), timeout=1.5):
            return True
    except OSError:
        return False


def ensure_mcp_server_started(wait_sec: float = 25.0) -> bool:
    """本地未跑 mcp_server 时由 app 拉起子进程并等待 8081 就绪。"""
    global mcp_process
    if _mcp_port_open():
        return True
    if not _tracked_mcp_running() and not _mcp_port_pids():
        mcp_process = subprocess.Popen(
            [sys.executable, str(BASE_DIR / "mcp_server.py")],
            cwd=str(BASE_DIR),
        )
    deadline = time.time() + wait_sec
    while time.time() < deadline:
        if _mcp_port_open():
            time.sleep(0.6)
            return True
        if mcp_process is not None and mcp_process.poll() is not None:
            return False
        time.sleep(0.35)
    return False


# ------------------------------
# 页面
# ------------------------------
@app.route('/')
def index():
    return render_template('1.html')


@app.route("/exports/<filename>")
def export_file_download(filename: str):
    """下载 MCP export_to_excel 写入 exports/ 目录下的文件（仅允许单层文件名）。"""
    if not filename or "/" in filename or "\\" in filename or filename.strip() in (".", ".."):
        abort(404)
    safe = Path(filename).name
    if safe != filename:
        abort(400)
    if not safe.lower().endswith(".xlsx"):
        abort(404)
    fp = (EXPORTS_DIR / safe).resolve()
    try:
        fp.relative_to(EXPORTS_DIR)
    except ValueError:
        abort(404)
    if not fp.is_file():
        abort(404)
    return send_from_directory(str(EXPORTS_DIR), safe, as_attachment=True, download_name=safe)


def _table_display_kind(content: str) -> str:
    s = (content or "").strip()
    if not s:
        return "text"
    sl = s.lower()
    if sl.startswith("<table") or sl.startswith("<!doctype html"):
        return "html"
    return "text"


def _excel_basename_from_export_tool(content: str) -> str | None:
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


def _extract_mcp_attachments_from_messages(messages: list) -> list[dict]:
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
                    "format": _table_display_kind(body),
                    "content": body,
                }
            )
        elif name == "export_to_excel":
            fn = _excel_basename_from_export_tool(c)
            if fn:
                out.append({"type": "excel", "label": fn, "filename": fn})
    return out


# ------------------------------
# MCP 服务控制
# ------------------------------
@app.route('/service/status', methods=['POST'])
def service_status():
    running = _tracked_mcp_running() or bool(_mcp_port_pids())
    return jsonify({"code": 0, "running": running})

@app.route('/service/start', methods=['POST'])
def service_start():
    global mcp_process
    try:
        if not _tracked_mcp_running() and _mcp_port_pids():
            _kill_mcp_port_processes()
            time.sleep(0.5)
        if mcp_process and mcp_process.poll() is None:
            return jsonify({"code": 0, "msg": "服务已在运行中"})
        mcp_process = subprocess.Popen(
            [sys.executable, str(BASE_DIR / "mcp_server.py")],
            cwd=str(BASE_DIR),
        )
        return jsonify({"code": 0, "msg": "MCP 服务启动成功"})
    except Exception as e:
        return jsonify({"code": -1, "msg": f"启动失败：{str(e)}"})

@app.route('/service/stop', methods=['POST'])
def service_stop():
    global mcp_process
    try:
        if mcp_process:
            mcp_process.terminate()
            mcp_process = None
        _kill_mcp_port_processes()
        return jsonify({"code": 0, "msg": "MCP 服务已停止"})
    except Exception as e:
        return jsonify({"code": -1, "msg": f"停止失败：{str(e)}"})

# ------------------------------
# 直接发送接口
# ------------------------------
@app.route('/send/wechat', methods=['POST'])
def send_wechat():
    data = request.get_json()
    name = data.get('name', '')
    content = data.get('content', '')
    if not name or not content:
        return jsonify({"code": -1, "msg": "参数不完整"})
    try:
        asyncio.run(send_wechat_agent(name, content))
        return jsonify({"code": 0, "msg": "微信消息发送成功"})
    except Exception as e:
        return jsonify({"code": -1, "msg": f"发送失败：{str(e)}"})

@app.route('/send/email', methods=['POST'])
def send_email():
    data = request.get_json()
    to = data.get('to', '')
    content = data.get('content', '')
    if not to or not content:
        return jsonify({"code": -1, "msg": "参数不完整"})
    try:
        asyncio.run(send_email_agent(to, content))
        return jsonify({"code": 0, "msg": "邮件发送成功"})
    except Exception as e:
        return jsonify({"code": -1, "msg": f"发送失败：{str(e)}"})

# ------------------------------
# AI 指令执行
# ------------------------------
def _dict_history_to_lc_messages(rows: list) -> list:
    out = []
    for m in rows:
        role = m.get("role")
        content = m.get("content") or ""
        if role == "user":
            out.append(HumanMessage(content=content))
        elif role == "assistant":
            out.append(AIMessage(content=content))
    return out


def _build_user_message_text(
    text: str,
    file_ids: list[str],
    session_id: str,
    *,
    omit_attachment_body: bool = False,
) -> str:
    parts: list[str] = []
    if text.strip():
        parts.append(text.strip())
    for meta in chat_redis.get_upload_metas(session_id, file_ids):
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


def _upload_meta_for_message(file_ids: list[str], session_id: str) -> list[dict]:
    items = []
    for meta in chat_redis.get_upload_metas(session_id, file_ids):
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


def _last_assistant_text(messages: list) -> str:
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


@app.route("/chat/upload", methods=["POST"])
def chat_upload():
    session_id = (request.form.get("session_id") or "default").strip() or "default"
    f = request.files.get("file")
    if not f or not f.filename:
        return jsonify({"code": -1, "msg": "请选择文件"})
    raw_name = safe_filename(f.filename)
    ext = Path(raw_name).suffix.lower()
    if ext not in ALLOWED_EXT:
        return jsonify(
            {
                "code": -1,
                "msg": f"不支持的类型 {ext}，允许：文档 {', '.join(sorted(DOC_EXT))} 或常见图片格式",
            }
        )
    data = f.read()
    if len(data) > MAX_UPLOAD_BYTES:
        return jsonify({"code": -1, "msg": f"文件超过 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB 限制"})
    kind = detect_kind(ext)
    if kind == "unknown":
        return jsonify({"code": -1, "msg": "无法识别文件类型"})

    file_id = new_file_id()
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{file_id}_{raw_name}"
    dest = session_dir / stored_name
    dest.write_bytes(data)

    try:
        parsed_text = asyncio.run(parse_uploaded_file(dest, UPLOADS_DIR, kind))
    except Exception as e:
        dest.unlink(missing_ok=True)
        return jsonify({"code": -1, "msg": f"解析失败：{_format_error(e)}"})

    rag_mode = rag_service.normalize_rag_mode(request.form.get("rag_mode"))

    meta = {
        "file_id": file_id,
        "name": raw_name,
        "kind": kind,
        "stored_name": stored_name,
        "relative_path": f"{session_id}/{stored_name}",
        "parsed_text": parsed_text,
        "parse_method": "glm-4v" if kind == "image" else "pdfplumber-markitdown",
        "rag_mode": rag_mode,
    }
    rag_result = rag_service.index_document(
        rag_mode, session_id, file_id, raw_name, parsed_text
    )
    if rag_result.get("database"):
        meta["milvus_database"] = rag_result["database"]
    if rag_result.get("collection"):
        meta["milvus_collection"] = rag_result["collection"]
    if rag_result.get("hierarchy"):
        meta["milvus_hierarchy"] = rag_result["hierarchy"]
    if rag_result.get("entities") is not None:
        meta["graph_entities"] = rag_result["entities"]
    if rag_result.get("relations") is not None:
        meta["graph_relations"] = rag_result["relations"]
    if rag_result.get("neo4j_database"):
        meta["neo4j_database"] = rag_result["neo4j_database"]
    chat_redis.save_upload_meta(session_id, file_id, meta)
    preview = parsed_text[:800] + ("…" if len(parsed_text) > 800 else "")
    return jsonify(
        {
            "code": 0,
            "msg": "上传并解析成功",
            "file": {
                "file_id": file_id,
                "name": raw_name,
                "kind": kind,
                "preview": preview,
                "parse_method": meta["parse_method"],
            },
            "graphrag": rag_result if rag_mode == "graphrag" else None,
            "rag": rag_result,
            "rag_mode": rag_mode,
        }
    )


@app.route('/chat/history', methods=['POST'])
def chat_history():
    data = request.get_json() or {}
    session_id = data.get("session_id") or "default"
    try:
        rows = chat_redis.get_all_messages(session_id)
        return jsonify({"code": 0, "session_id": session_id, "messages": rows})
    except Exception as e:
        return jsonify({"code": -1, "msg": f"读取 Redis 失败：{str(e)}"})


@app.route('/chat/clear', methods=['POST'])
def chat_clear():
    data = request.get_json() or {}
    session_id = data.get("session_id") or "default"
    try:
        chat_redis.clear_session(session_id)
        rag_service.delete_session_indexes(session_id)
        return jsonify({"code": 0, "msg": "会话记忆已清空"})
    except Exception as e:
        return jsonify({"code": -1, "msg": str(e)})


@app.route("/chat/sessions", methods=["GET"])
def chat_sessions_list():
    try:
        limit = int(request.args.get("limit", 80))
    except (TypeError, ValueError):
        limit = 80
    limit = max(1, min(limit, 200))
    try:
        return jsonify({"code": 0, "sessions": chat_redis.list_sessions(limit)})
    except Exception as e:
        return jsonify({"code": -1, "msg": str(e)})


@app.route("/chat/session/rename", methods=["POST"])
def chat_session_rename():
    data = request.get_json() or {}
    session_id = data.get("session_id") or "default"
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"code": -1, "msg": "标题不能为空"})
    try:
        chat_redis.set_session_title(session_id, title)
        return jsonify({"code": 0, "msg": "已重命名"})
    except Exception as e:
        return jsonify({"code": -1, "msg": str(e)})


@app.route("/chat/session/delete", methods=["POST"])
def chat_session_delete():
    data = request.get_json() or {}
    session_id = data.get("session_id") or "default"
    try:
        chat_redis.clear_session(session_id)
        rag_service.delete_session_indexes(session_id)
        return jsonify({"code": 0, "msg": "已删除会话"})
    except Exception as e:
        return jsonify({"code": -1, "msg": str(e)})


def _build_tool_debug_from_messages(messages: list) -> dict:
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
        "used_send_message": "send_message" in names,
        "used_send_email": "send_email" in names,
        "used_web_search": "web_search" in names,
        "used_export_to_excel": "export_to_excel" in names,
        "used_format_pretty_table": "format_pretty_table" in names,
    }


@app.route("/chat/rag/index", methods=["GET"])
def chat_rag_index():
    """列出当前会话知识库索引；mode=rag|graphrag|all。"""
    session_id = (request.args.get("session_id") or "default").strip() or "default"
    mode = (request.args.get("mode") or "all").strip().lower()
    try:
        if mode == "all":
            indexes = rag_service.list_all_session_indexes(session_id)
            return jsonify(
                {
                    "code": 0,
                    "session_id": session_id,
                    "indexes": indexes,
                }
            )
        items = rag_service.list_session_index(session_id, mode)
        return jsonify(
            {
                "code": 0,
                "session_id": session_id,
                "mode": rag_service.normalize_rag_mode(mode),
                "documents": items,
            }
        )
    except Exception as e:
        return jsonify({"code": -1, "msg": str(e)})


@app.route('/chat/message', methods=['POST'])
def chat_message():
    data = request.get_json() or {}
    session_id = data.get("session_id") or "default"
    text = (data.get("message") or "").strip()
    file_ids = data.get("file_ids") or []
    if isinstance(file_ids, str):
        file_ids = [file_ids] if file_ids else []
    file_ids = [str(x).strip() for x in file_ids if str(x).strip()]
    include_tool_debug = bool(data.get("include_tool_debug"))
    rag_mode = rag_service.normalize_rag_mode(data.get("rag_mode"))
    rag_query = text.strip() or "请根据已上传文档回答用户问题"
    rag_context = rag_service.build_rag_context(
        session_id, rag_query, file_ids=file_ids or None, mode=rag_mode
    )
    use_rag_attachments = rag_service.should_omit_attachment_body(
        session_id, file_ids, mode=rag_mode
    )
    full_text = _build_user_message_text(
        text,
        file_ids,
        session_id,
        omit_attachment_body=use_rag_attachments,
    )
    if not full_text:
        return jsonify({"code": -1, "msg": "请输入消息或先上传附件"})
    try:
        user_uploads = _upload_meta_for_message(file_ids, session_id) if file_ids else None
        chat_redis.save_message(
            session_id,
            "user",
            full_text,
            user_uploads=user_uploads,
        )
        history_rows = chat_redis.get_all_messages(session_id)
        lc_messages = _dict_history_to_lc_messages(history_rows)
        agent_system_prompt = _chat_agent_prompt_with_rag(rag_context)
        reply, msgs = asyncio.run(
            run_agent_with_history(
                lc_messages,
                rag_context=rag_context,
                session_id=session_id,
                log_prompt=LOG_LLM_PROMPT or include_tool_debug,
            )
        )
        attachments = _extract_mcp_attachments_from_messages(msgs)
        chat_redis.save_message(
            session_id,
            "assistant",
            reply,
            mcp_attachments=attachments or None,
        )
        out: dict = {"code": 0, "msg": reply, "session_id": session_id}
        if attachments:
            out["mcp_attachments"] = attachments
        if rag_context:
            out["rag_used"] = True
            out["rag_mode"] = rag_mode
            if rag_mode == "graphrag":
                out["graphrag_used"] = True
        if include_tool_debug:
            out["tool_debug"] = _build_tool_debug_from_messages(msgs)
            out["prompt_debug"] = _prompt_debug_payload(agent_system_prompt, rag_context)
        return jsonify(out)
    except Exception as e:
        err_name = type(e).__name__
        hint = ""
        if "Timeout" in err_name or "timeout" in str(e).lower():
            hint = "（多为文档过长或智谱 API 响应超时，已启用 GraphRAG 精简上下文；可设置 LLM_REQUEST_TIMEOUT=300 后重启）"
        elif "429" in str(e) or "too many requests" in str(e).lower() or "频率" in str(e):
            hint = "（智谱 API 触发限速 HTTP 429，已自动重试；若仍失败请增大 API_REQUEST_INTERVAL_SEC / GRAPHRAG_EXTRACT_BATCH_DELAY_SEC 后重启）"
        return jsonify({"code": -1, "msg": f"对话失败：{_format_error(e)}{hint}"})


@app.route('/ai/run', methods=['POST'])
def ai_run():
    prompt = (request.get_json() or {}).get('prompt', '')
    if not prompt:
        return jsonify({"code": -1, "msg": "请输入指令"})
    try:
        result = asyncio.run(run_agent(prompt))
        return jsonify({"code": 0, "msg": result})
    except Exception as e:
        return jsonify({"code": -1, "msg": f"执行失败：{_format_error(e)}"})

# ------------------------------
# MCP：拉全量工具（修复 MCPToolkit 只读 tools/list 第一页导致 web_search 等丢失）
# 参见 rectalogic/langchain-mcp MCPToolkit.initialize 仅一次 list_tools、不跟 nextCursor。
# ------------------------------
async def langchain_tools_from_mcp_session(session: ClientSession):
    from langchain_mcp.toolkit import MCPTool

    await session.initialize()
    defs: list = []
    page = await session.list_tools()
    defs.extend(page.tools)
    cursor = getattr(page, "nextCursor", None)
    while cursor:
        page = await session.list_tools(params=types.PaginatedRequestParams(cursor=cursor))
        defs.extend(page.tools)
        cursor = getattr(page, "nextCursor", None)
    return [
        MCPTool(
            session=session,
            name=t.name,
            description=t.description or "",
            args_schema=t.inputSchema,
        )
        for t in defs
    ]


# ------------------------------
# Agent 执行
# ------------------------------
async def run_agent(prompt: str):
    if not ensure_mcp_server_started():
        raise RuntimeError(
            f"MCP 未在 {MCP_HOST}:{MCP_PORT} 就绪，请另开终端运行: python mcp_server.py"
        )
    llm = make_chat_llm()
    async with streamable_http_client(MCP_URL) as (r, w, _):
        async with ClientSession(r, w) as session:
            tools = await langchain_tools_from_mcp_session(session)
            agent = create_react_agent(llm, tools)
            state = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
            return _last_assistant_text(state["messages"])


CHAT_AGENT_PROMPT = (
    "你是智能助手，能记住当前对话里用户说过的话。\n"
    "用户消息中「--- 附件 [...] ---」区块是系统已解析的上传文件（PDF/Office 已提取正文，图片经 GLM-4V），请直接基于附件内容回答。\n"
    "若系统额外提供了「知识库检索结果」或「GraphRAG 混合检索结果」，说明已从 Milvus / Neo4j 做了文档检索；请优先依据检索结果作答，并可在必要时结合附件全文。\n"
    "用户仅询问已上传文档/文章时，直接根据知识库检索结果与对话内容回答，不要调用 web_search 等联网工具。\n"
    "需要发微信或发邮件时，分别使用 send_message、send_email 工具。\n"
    "涉及时效、新闻、股价、黄金/汇率/商品价格、天气、政策等需要联网核实时，必须先调用 web_search 工具（需服务端已配置 TAVILY_API_KEY），再基于搜索结果回答。\n"
    "如果 web_search 工具可用，不要回答“没有实时查询能力”或让用户自行去网站查询。\n"
    "用户要表格展示时用 format_pretty_table；明确要求导出 / 保存为 Excel 时用 export_to_excel，并传入表头 headers 与二维 rows。\n"
    "纯聊天可直接回答。"
)


def _chat_agent_prompt_with_rag(rag_context: str | None) -> str:
    prompt = CHAT_AGENT_PROMPT
    if rag_context and rag_context.strip():
        prompt = f"{prompt}\n\n{rag_context.strip()}"
    return prompt


def _clip_for_log(text: str, max_len: int | None = None) -> str:
    cap = max_len if max_len is not None else LOG_LLM_PROMPT_MAX
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n…(日志已截断，全文 {len(text)} 字符，可调大 LOG_LLM_PROMPT_MAX)"


def _log_llm_system_prompt(
    label: str,
    system_prompt: str,
    *,
    session_id: str = "",
    rag_context: str | None = None,
) -> None:
    """在运行 python app.py 的终端打印 System Prompt（不会写入 Redis）。"""
    sep = "=" * 72
    logger.info(
        "%s\n[LLM System Prompt] mode=%s session_id=%s len=%s\n%s\n%s",
        sep,
        label,
        session_id or "-",
        len(system_prompt),
        _clip_for_log(system_prompt),
        sep,
    )
    if rag_context is not None:
        logger.info(
            "[GraphRAG context only] session_id=%s len=%s\n%s",
            session_id or "-",
            len(rag_context),
            _clip_for_log(rag_context or "(empty)"),
        )


def _prompt_debug_payload(
    system_prompt: str,
    rag_context: str | None,
) -> dict:
    return {
        "system_prompt_length": len(system_prompt),
        "system_prompt": _clip_for_log(system_prompt, 8000),
        "rag_context_length": len(rag_context or ""),
        "rag_context": _clip_for_log(rag_context or "", 8000),
        "note": "以上内容仅当次请求注入模型，不会存入 Redis 对话 List",
    }

CHAT_OFFLINE_PROMPT_SUFFIX = (
    "\n（说明：当前未连接 MCP 工具服务，你只能根据对话与附件内容作答，"
    "不能发微信、发邮件、联网搜索或导出 Excel；若用户要求这些能力，请说明需先启动 MCP。）"
)


async def run_chat_llm_only(
    lc_messages: list,
    rag_context: str | None = None,
    *,
    session_id: str = "",
    log_prompt: bool = False,
) -> tuple[str, list]:
    """MCP 不可用时：直接用 GLM 对话（附件正文已在消息里）。"""
    llm = make_chat_llm()
    prompt = _chat_agent_prompt_with_rag(rag_context) + CHAT_OFFLINE_PROMPT_SUFFIX
    if log_prompt:
        _log_llm_system_prompt(
            "llm_only_offline",
            prompt,
            session_id=session_id,
            rag_context=rag_context,
        )
    resp = await llm.ainvoke([SystemMessage(content=prompt)] + list(lc_messages))
    msgs = list(lc_messages) + [resp]
    return _last_assistant_text(msgs), msgs


async def run_agent_with_history(
    lc_messages: list,
    rag_context: str | None = None,
    *,
    session_id: str = "",
    log_prompt: bool = False,
) -> tuple[str, list]:
    """带完整上下文的 Agent；返回 (助手可见文本, 完整消息列表供工具调试)。"""
    if not ensure_mcp_server_started():
        return await run_chat_llm_only(
            lc_messages,
            rag_context=rag_context,
            session_id=session_id,
            log_prompt=log_prompt,
        )
    try:
        if log_prompt:
            _log_llm_system_prompt(
                "react_agent",
                _chat_agent_prompt_with_rag(rag_context),
                session_id=session_id,
                rag_context=rag_context,
            )
        llm = make_chat_llm()
        async with streamable_http_client(MCP_URL) as (r, w, _):
            async with ClientSession(r, w) as session:
                tools = await langchain_tools_from_mcp_session(session)
                agent = create_react_agent(
                    llm,
                    tools,
                    prompt=_chat_agent_prompt_with_rag(rag_context),
                )
                state = await agent.ainvoke({"messages": lc_messages})
                msgs = state.get("messages") or []
                return _last_assistant_text(msgs), msgs
    except BaseException as e:
        logger.warning(
            "Agent+MCP 调用失败，已降级为纯 LLM（本轮不会出现 ToolMessage）：%s",
            _format_error(e),
        )
        return await run_chat_llm_only(
            lc_messages,
            rag_context=rag_context,
            session_id=session_id,
            log_prompt=log_prompt,
        )

async def send_wechat_agent(name, content):
    llm = make_chat_llm()
    async with streamable_http_client(MCP_URL) as (r, w, _):
        async with ClientSession(r, w) as session:
            tools = await langchain_tools_from_mcp_session(session)
            agent = create_react_agent(llm, tools)
            await agent.ainvoke({"messages": [HumanMessage(content=f"给微信名称{name}发消息：{content}")]})

async def send_email_agent(to, content):
    llm = make_chat_llm()
    async with streamable_http_client(MCP_URL) as (r, w, _):
        async with ClientSession(r, w) as session:
            tools = await langchain_tools_from_mcp_session(session)
            agent = create_react_agent(llm, tools)
            await agent.ainvoke({"messages": [HumanMessage(content=f"给邮箱{to}发送内容：{content}")]})
if __name__ == '__main__':
    if ensure_mcp_server_started():
        print(f"MCP 已就绪: {MCP_URL}")
    else:
        print(
            f"警告: MCP 未在 {MCP_PORT} 端口就绪，对话将降级为纯模型（附件问答仍可用）。"
            " 可手动运行: python mcp_server.py"
        )
    app.run(host='0.0.0.0', port=5001, debug=False, use_reloader=False)
