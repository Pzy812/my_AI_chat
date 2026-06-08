"""对话、上传、会话与 RAG 相关 API。"""
import asyncio
from pathlib import Path

from flask import Blueprint, jsonify, request

import chat_store
import rag_service
from agent_service import (
    chat_agent_prompt_with_rag,
    prompt_debug_payload,
    run_agent,
    run_agent_with_history,
)
from app_config import LOG_LLM_PROMPT, UPLOADS_DIR
from app_utils import format_error
from chat_helpers import (
    build_tool_debug_from_messages,
    build_user_message_text,
    dict_history_to_lc_messages,
    extract_mcp_attachments_from_messages,
    upload_meta_for_message,
)
from file_upload import (
    ALLOWED_EXT,
    DOC_EXT,
    MAX_UPLOAD_BYTES,
    detect_kind,
    new_file_id,
    parse_uploaded_file,
    safe_filename,
)

bp = Blueprint("chat", __name__)


@bp.route("/chat/upload", methods=["POST"])
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
        return jsonify({"code": -1, "msg": f"解析失败：{format_error(e)}"})

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
    chat_store.save_upload_meta(session_id, file_id, meta)
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


@bp.route("/chat/history", methods=["POST"])
def chat_history():
    data = request.get_json() or {}
    session_id = data.get("session_id") or "default"
    try:
        rows = chat_store.get_all_messages(session_id)
        return jsonify({"code": 0, "session_id": session_id, "messages": rows})
    except Exception as e:
        return jsonify({"code": -1, "msg": f"读取会话历史失败：{str(e)}"})


@bp.route("/chat/clear", methods=["POST"])
def chat_clear():
    data = request.get_json() or {}
    session_id = data.get("session_id") or "default"
    try:
        chat_store.clear_session(session_id)
        rag_service.delete_session_indexes(session_id)
        return jsonify({"code": 0, "msg": "会话记忆已清空"})
    except Exception as e:
        return jsonify({"code": -1, "msg": str(e)})


@bp.route("/chat/sessions", methods=["GET"])
def chat_sessions_list():
    try:
        limit = int(request.args.get("limit", 80))
    except (TypeError, ValueError):
        limit = 80
    limit = max(1, min(limit, 200))
    try:
        return jsonify({"code": 0, "sessions": chat_store.list_sessions(limit)})
    except Exception as e:
        return jsonify({"code": -1, "msg": str(e)})


@bp.route("/chat/session/rename", methods=["POST"])
def chat_session_rename():
    data = request.get_json() or {}
    session_id = data.get("session_id") or "default"
    title = (data.get("title") or "").strip()
    if not title:
        return jsonify({"code": -1, "msg": "标题不能为空"})
    try:
        chat_store.set_session_title(session_id, title)
        return jsonify({"code": 0, "msg": "已重命名"})
    except Exception as e:
        return jsonify({"code": -1, "msg": str(e)})


@bp.route("/chat/session/delete", methods=["POST"])
def chat_session_delete():
    data = request.get_json() or {}
    session_id = data.get("session_id") or "default"
    try:
        chat_store.clear_session(session_id)
        rag_service.delete_session_indexes(session_id)
        return jsonify({"code": 0, "msg": "已删除会话"})
    except Exception as e:
        return jsonify({"code": -1, "msg": str(e)})


@bp.route("/chat/rag/index", methods=["GET"])
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


@bp.route("/chat/message", methods=["POST"])
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
    full_text = build_user_message_text(
        text,
        file_ids,
        session_id,
        omit_attachment_body=use_rag_attachments,
    )
    if not full_text:
        return jsonify({"code": -1, "msg": "请输入消息或先上传附件"})
    try:
        user_uploads = upload_meta_for_message(file_ids, session_id) if file_ids else None
        chat_store.save_message(
            session_id,
            "user",
            full_text,
            user_uploads=user_uploads,
        )
        history_rows = chat_store.get_recent_messages(session_id)
        lc_messages = dict_history_to_lc_messages(history_rows)
        agent_system_prompt = chat_agent_prompt_with_rag(rag_context)
        reply, msgs = asyncio.run(
            run_agent_with_history(
                lc_messages,
                rag_context=rag_context,
                session_id=session_id,
                log_prompt=LOG_LLM_PROMPT or include_tool_debug,
            )
        )
        attachments = extract_mcp_attachments_from_messages(msgs)
        chat_store.save_message(
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
            out["tool_debug"] = build_tool_debug_from_messages(msgs)
            out["prompt_debug"] = prompt_debug_payload(agent_system_prompt, rag_context)
        return jsonify(out)
    except Exception as e:
        err_name = type(e).__name__
        hint = ""
        if "Timeout" in err_name or "timeout" in str(e).lower():
            hint = "（多为文档过长或智谱 API 响应超时，已启用 GraphRAG 精简上下文；可设置 LLM_REQUEST_TIMEOUT=300 后重启）"
        elif "429" in str(e) or "too many requests" in str(e).lower() or "频率" in str(e):
            hint = "（智谱 API 触发限速 HTTP 429，已自动重试；若仍失败请增大 API_REQUEST_INTERVAL_SEC / GRAPHRAG_EXTRACT_BATCH_DELAY_SEC 后重启）"
        return jsonify({"code": -1, "msg": f"对话失败：{format_error(e)}{hint}"})


@bp.route("/ai/run", methods=["POST"])
def ai_run():
    prompt = (request.get_json() or {}).get("prompt", "")
    if not prompt:
        return jsonify({"code": -1, "msg": "请输入指令"})
    try:
        result = asyncio.run(run_agent(prompt))
        return jsonify({"code": 0, "msg": result})
    except Exception as e:
        return jsonify({"code": -1, "msg": f"执行失败：{format_error(e)}"})
