"""对话、上传、会话与 RAG 相关 API。"""
from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable
from pathlib import Path
from typing import Any

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi.responses import StreamingResponse

from core.async_runner import cancel_active_run, register_run, run_cancellable, schedule_async, unregister_run

import chat.chat_store as chat_store
import rag.rag_service as rag_service
from agent.agent_service import (
    chat_agent_prompt_with_rag,
    hitl_available,
    prompt_debug_payload,
    run_agent,
    run_agent_hitl_resume,
    run_agent_with_history,
)
from agent.agent_stream import (
    build_stream_done_payload,
    build_stream_hitl_payload,
    stream_agent_hitl_resume,
    stream_agent_with_history,
)
from config.app_config import LOG_LLM_PROMPT, UPLOADS_DIR
from core.app_utils import format_error
from observability.langsmith_trace import finalize_trace_for_session
from observability.langsmith_session import clear_session_trace
from chat.chat_helpers import (
    build_tool_debug_from_messages,
    build_user_message_text,
    extract_mcp_attachments_from_messages,
    upload_meta_for_message,
)
from chat.chat_summary import prepare_agent_lc_messages
from upload.file_upload import (
    ALLOWED_EXT,
    DOC_EXT,
    MAX_UPLOAD_BYTES,
    detect_kind,
    new_file_id,
    parse_uploaded_file,
    safe_filename,
)

from llm.model_config import normalize_llm_config

router = APIRouter(tags=["chat"])

_SSE_HEADERS = {
    "Cache-Control": "no-cache",
    "Connection": "keep-alive",
    "X-Accel-Buffering": "no",
}


def _extract_llm_config(data: dict) -> dict:
    return normalize_llm_config(data.get("llm_config") or data.get("llm") or None)


def _extract_hitl_enabled(data: dict) -> bool:
    val = data.get("hitl_enabled")
    if val is None:
        return True
    if isinstance(val, str):
        return val.strip().lower() not in ("0", "false", "no", "off")
    return bool(val)


def _maybe_schedule_auto_title(session_id: str) -> None:
    from chat.session_title import schedule_session_auto_title

    schedule_session_auto_title(session_id)


def _sse_payload(data: dict) -> str:
    try:
        body = json.dumps(data, ensure_ascii=False, default=str)
    except Exception as e:
        body = json.dumps(
            {"type": "error", "code": -1, "msg": f"SSE 序列化失败：{e}"},
            ensure_ascii=False,
        )
    return f"data: {body}\n\n"


def _chat_error_hint(exc: BaseException) -> str:
    err_name = type(exc).__name__
    err_text = str(exc).lower()
    hint = ""
    if "Timeout" in err_name or "timeout" in err_text:
        hint = "（多为文档过长或智谱 API 响应超时，已启用 GraphRAG 精简上下文；可设置 LLM_REQUEST_TIMEOUT=300 后重启）"
    elif "429" in str(exc) or "too many requests" in err_text or "频率" in str(exc):
        hint = "（智谱 API 触发限速 HTTP 429，已自动重试；若仍失败请增大 API_REQUEST_INTERVAL_SEC / GRAPHRAG_EXTRACT_BATCH_DELAY_SEC 后重启）"
    elif err_name == "SSEError" or (
        "text/event-stream" in err_text and "got ''" in err_text
    ):
        hint = (
            "（智谱流式接口返回异常，多为 API 429/限速或网络问题，与 MCP 工具服务无关；"
            "请稍后重试或关闭「推理流式展示」）"
        )
    elif "connection attempts failed" in err_text or "connecterror" in err_text:
        hint = "（多为 Windows 系统代理导致无法连接本机 MCP；已修复 trust_env，请重启 app.py 与 mcp_server.py；或检查 MCP 8090 端口）"
    elif "text/event-stream" in err_text or err_name in ("TransportError",):
        hint = "（MCP 工具服务连接异常，系统已尝试自动重启 MCP；若仍失败请手动重启 app.py 与 mcp_server.py）"
    return hint


async def _stream_agent_events(
    async_gen: AsyncIterator[dict],
    *,
    on_result: Callable[[dict], list[dict]],
    run_key: str | None = None,
) -> AsyncIterator[str]:
    """SSE 异步生成器：转发 step 事件，在 _agent_result 时调用 on_result。"""
    yield ": stream-open\n\n"
    task = asyncio.current_task()
    if run_key and task is not None:
        register_run(run_key, task)
    try:
        async for event in async_gen:
            if event.get("type") == "_agent_result":
                try:
                    outs = on_result(event) or []
                except Exception as e:
                    outs = [
                        {
                            "type": "error",
                            "code": -1,
                            "msg": f"保存结果失败：{format_error(e)}",
                        }
                    ]
                for out in outs:
                    yield _sse_payload(out)
                continue
            yield _sse_payload(event)
    except asyncio.CancelledError:
        yield _sse_payload(
            {
                "type": "cancelled",
                "code": 0,
                "status": "cancelled",
                "msg": "执行已由用户中断",
            }
        )
    except BaseException as e:
        yield _sse_payload(
            {
                "type": "error",
                "code": -1,
                "msg": f"对话失败：{format_error(e)}{_chat_error_hint(e)}",
            }
        )
    finally:
        if run_key and task is not None:
            unregister_run(run_key, task)
    yield _sse_payload({"type": "stream_end", "code": 0})


def _prepare_chat_message_context(data: dict) -> dict[str, Any]:
    """解析 /chat/message 与 /chat/message/stream 共用参数。"""
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
    return {
        "session_id": session_id,
        "text": text,
        "file_ids": file_ids,
        "include_tool_debug": include_tool_debug,
        "rag_mode": rag_mode,
        "rag_context": rag_context,
        "full_text": full_text,
        "llm_config": _extract_llm_config(data),
        "hitl_enabled": _extract_hitl_enabled(data),
    }


def _build_chat_success_payload(
    session_id: str,
    reply: str,
    msgs: list,
    *,
    rag_context: str,
    rag_mode: str,
    include_tool_debug: bool,
    agent_system_prompt: str,
    langsmith_trace: dict | None = None,
) -> dict:
    attachments = extract_mcp_attachments_from_messages(msgs)
    out: dict = {"code": 0, "status": "completed", "msg": reply, "session_id": session_id}
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
    if langsmith_trace:
        out["langsmith"] = langsmith_trace
    return out


@router.post("/chat/upload")
async def chat_upload(
    session_id: str = Form("default"),
    rag_mode: str | None = Form(None),
    file: UploadFile = File(...),
):
    session_id = (session_id or "default").strip() or "default"
    if not file.filename:
        return {"code": -1, "msg": "请选择文件"}
    raw_name = safe_filename(file.filename)
    ext = Path(raw_name).suffix.lower()
    if ext not in ALLOWED_EXT:
        return {
            "code": -1,
            "msg": f"不支持的类型 {ext}，允许：文档 {', '.join(sorted(DOC_EXT))} 或常见图片格式",
        }
    data = await file.read()
    if len(data) > MAX_UPLOAD_BYTES:
        return {"code": -1, "msg": f"文件超过 {MAX_UPLOAD_BYTES // (1024 * 1024)}MB 限制"}
    kind = detect_kind(ext)
    if kind == "unknown":
        return {"code": -1, "msg": "无法识别文件类型"}

    file_id = new_file_id()
    session_dir = UPLOADS_DIR / session_id
    session_dir.mkdir(parents=True, exist_ok=True)
    stored_name = f"{file_id}_{raw_name}"
    dest = session_dir / stored_name
    dest.write_bytes(data)

    try:
        parsed_text = await parse_uploaded_file(dest, UPLOADS_DIR, kind)
    except Exception as e:
        dest.unlink(missing_ok=True)
        return {"code": -1, "msg": f"解析失败：{format_error(e)}"}

    mode = rag_service.normalize_rag_mode(rag_mode)

    meta = {
        "file_id": file_id,
        "name": raw_name,
        "kind": kind,
        "stored_name": stored_name,
        "relative_path": f"{session_id}/{stored_name}",
        "parsed_text": parsed_text,
        "parse_method": "glm-4v" if kind == "image" else "pdfplumber-markitdown",
        "rag_mode": mode,
    }
    rag_result = await asyncio.to_thread(
        rag_service.index_document, mode, session_id, file_id, raw_name, parsed_text
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
    return {
        "code": 0,
        "msg": "上传并解析成功",
        "file": {
            "file_id": file_id,
            "name": raw_name,
            "kind": kind,
            "preview": preview,
            "parse_method": meta["parse_method"],
        },
        "graphrag": rag_result if mode == "graphrag" else None,
        "rag": rag_result,
        "rag_mode": mode,
    }


@router.post("/chat/history")
async def chat_history(request: Request):
    data = await request.json() or {}
    session_id = data.get("session_id") or "default"
    try:
        rows = await asyncio.to_thread(chat_store.get_all_messages, session_id)
        return {"code": 0, "session_id": session_id, "messages": rows}
    except Exception as e:
        return {"code": -1, "msg": f"读取会话历史失败：{str(e)}"}


@router.post("/chat/clear")
async def chat_clear(request: Request):
    data = await request.json() or {}
    session_id = data.get("session_id") or "default"
    try:
        await asyncio.to_thread(chat_store.clear_session, session_id)
        await asyncio.to_thread(rag_service.delete_session_indexes, session_id)
        clear_session_trace(session_id)
        return {"code": 0, "msg": "会话记忆已清空"}
    except Exception as e:
        return {"code": -1, "msg": str(e)}


@router.get("/chat/sessions")
async def chat_sessions_list(limit: int = 80):
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 80
    limit = max(1, min(limit, 200))
    try:
        sessions = await asyncio.to_thread(chat_store.list_sessions, limit)
        return {"code": 0, "sessions": sessions}
    except Exception as e:
        return {"code": -1, "msg": str(e)}


@router.post("/chat/session/rename")
async def chat_session_rename(request: Request):
    data = await request.json() or {}
    session_id = data.get("session_id") or "default"
    title = (data.get("title") or "").strip()
    if not title:
        return {"code": -1, "msg": "标题不能为空"}
    try:
        await asyncio.to_thread(chat_store.set_session_title, session_id, title)
        return {"code": 0, "msg": "已重命名"}
    except Exception as e:
        return {"code": -1, "msg": str(e)}


@router.post("/chat/session/delete")
async def chat_session_delete(request: Request):
    data = await request.json() or {}
    session_id = data.get("session_id") or "default"
    try:
        await asyncio.to_thread(chat_store.clear_session, session_id)
        await asyncio.to_thread(rag_service.delete_session_indexes, session_id)
        clear_session_trace(session_id)
        return {"code": 0, "msg": "已删除会话"}
    except Exception as e:
        return {"code": -1, "msg": str(e)}


@router.get("/chat/rag/index")
async def chat_rag_index(session_id: str = "default", mode: str = "all"):
    """列出当前会话知识库索引；mode=rag|graphrag|all。"""
    session_id = (session_id or "default").strip() or "default"
    mode = (mode or "all").strip().lower()
    try:
        if mode == "all":
            indexes = await asyncio.to_thread(rag_service.list_all_session_indexes, session_id)
            return {
                "code": 0,
                "session_id": session_id,
                "indexes": indexes,
            }
        items = await asyncio.to_thread(rag_service.list_session_index, session_id, mode)
        return {
            "code": 0,
            "session_id": session_id,
            "mode": rag_service.normalize_rag_mode(mode),
            "documents": items,
        }
    except Exception as e:
        return {"code": -1, "msg": str(e)}


@router.post("/chat/message")
async def chat_message(request: Request):
    data = await request.json() or {}
    ctx = _prepare_chat_message_context(data)
    if not ctx["full_text"]:
        return {"code": -1, "msg": "请输入消息或先上传附件"}
    session_id = ctx["session_id"]
    file_ids = ctx["file_ids"]
    include_tool_debug = ctx["include_tool_debug"]
    rag_mode = ctx["rag_mode"]
    rag_context = ctx["rag_context"]
    try:
        user_uploads = upload_meta_for_message(file_ids, session_id) if file_ids else None
        await asyncio.to_thread(
            chat_store.save_message,
            session_id,
            "user",
            ctx["full_text"],
            user_uploads=user_uploads,
        )
        agent_system_prompt = chat_agent_prompt_with_rag(rag_context)

        async def _invoke_chat():
            lc_messages = await prepare_agent_lc_messages(session_id)
            return await run_agent_with_history(
                lc_messages,
                rag_context=rag_context,
                session_id=session_id,
                log_prompt=LOG_LLM_PROMPT or include_tool_debug,
                file_count=len(file_ids),
                llm_config=ctx["llm_config"],
                hitl_enabled=ctx["hitl_enabled"],
            )

        reply, msgs, hitl_pending = await run_cancellable(
            _invoke_chat(), run_key=session_id
        )
        langsmith_trace = finalize_trace_for_session(session_id)
        if hitl_pending:
            pending = hitl_pending[0] if hitl_pending else {}
            hitl_out = {
                "code": 0,
                "status": "hitl_pending",
                "session_id": session_id,
                "rag_mode": rag_mode,
                "hitl": {
                    "pending": hitl_pending,
                    "tool": pending.get("tool"),
                    "label": pending.get("label"),
                    "summary": pending.get("summary"),
                    "args": pending.get("args"),
                    "hitl_enabled": hitl_available(),
                },
                "msg": "等待您确认是否执行敏感操作",
            }
            if langsmith_trace:
                hitl_out["langsmith"] = langsmith_trace
            return hitl_out
        if not reply:
            return {"code": -1, "msg": "模型未返回有效回复"}
        await asyncio.to_thread(
            chat_store.save_message,
            session_id,
            "assistant",
            reply,
            mcp_attachments=extract_mcp_attachments_from_messages(msgs) or None,
        )
        _maybe_schedule_auto_title(session_id)
        return _build_chat_success_payload(
            session_id,
            reply,
            msgs,
            rag_context=rag_context,
            rag_mode=rag_mode,
            include_tool_debug=include_tool_debug,
            agent_system_prompt=agent_system_prompt,
            langsmith_trace=langsmith_trace,
        )
    except asyncio.CancelledError:
        return {
            "code": 0,
            "status": "cancelled",
            "msg": "执行已由用户中断",
            "session_id": session_id,
        }
    except Exception as e:
        return {"code": -1, "msg": f"对话失败：{format_error(e)}{_chat_error_hint(e)}"}


@router.post("/chat/message/stream")
async def chat_message_stream(request: Request):
    """SSE：流式推送 Agent Thought → Action → Observation。"""
    data = await request.json() or {}
    ctx = _prepare_chat_message_context(data)
    if not ctx["full_text"]:
        return {"code": -1, "msg": "请输入消息或先上传附件"}
    session_id = ctx["session_id"]
    file_ids = ctx["file_ids"]
    include_tool_debug = ctx["include_tool_debug"]
    rag_mode = ctx["rag_mode"]
    rag_context = ctx["rag_context"]
    agent_system_prompt = chat_agent_prompt_with_rag(rag_context)

    user_uploads = upload_meta_for_message(file_ids, session_id) if file_ids else None
    await asyncio.to_thread(
        chat_store.save_message,
        session_id,
        "user",
        ctx["full_text"],
        user_uploads=user_uploads,
    )

    async def _stream_with_summary():
        from agent.harness import (
            needs_task_harness,
            task_harness_event_payload,
        )
        from agent.task_checklist import extract_primary_user_goal, is_continue_message
        from agent.task_state import default_task_fields

        yield {
            "type": "step",
            "phase": "status",
            "content": "正在准备对话上下文…",
        }
        is_continue = is_continue_message(ctx["text"])
        lc_messages = await prepare_agent_lc_messages(session_id, fast=is_continue)
        user_goal = extract_primary_user_goal(lc_messages)
        if user_goal:
            preview = default_task_fields()
            preview["user_goal"] = user_goal
            preview["harness_enabled"] = needs_task_harness(
                user_goal, file_count=len(file_ids)
            )
            if is_continue:
                persisted = chat_store.get_task_harness_meta(session_id)
                if persisted.get("plan"):
                    preview["plan"] = list(persisted["plan"])
                    preview["plan_index"] = int(persisted.get("plan_index") or 0)
                    preview["task_phase"] = persisted.get("task_phase") or "gather"
                    preview["step_checklist"] = list(persisted.get("step_checklist") or [])
                    preview["harness_enabled"] = bool(
                        persisted.get("harness_enabled", preview["harness_enabled"])
                    )
            yield task_harness_event_payload(preview)
        yield {
            "type": "step",
            "phase": "status",
            "content": (
                "正在恢复任务进度并启动 Agent…"
                if is_continue
                else "正在生成任务计划并启动 Agent…"
            ),
        }
        async for event in stream_agent_with_history(
            lc_messages,
            rag_context=rag_context,
            session_id=session_id,
            log_prompt=LOG_LLM_PROMPT or include_tool_debug,
            file_count=len(file_ids),
            llm_config=ctx["llm_config"],
            hitl_enabled=ctx["hitl_enabled"],
            rag_mode=rag_mode,
        ):
            yield event

    def on_result(event: dict):
        if event.get("_error"):
            return [{"type": "error", "code": -1, "msg": event["_error"]}]
        reply = event.get("reply")
        msgs = event.get("messages") or []
        hitl_pending = event.get("hitl")
        langsmith_trace = event.get("langsmith")
        if hitl_pending:
            return [
                build_stream_hitl_payload(
                    session_id=session_id,
                    hitl_pending=hitl_pending,
                    rag_mode=rag_mode,
                    langsmith_trace=langsmith_trace,
                )
            ]
        if not reply:
            return [
                {
                    "type": "error",
                    "code": -1,
                    "msg": "Agent 在工具调用阶段中断且未完成，请刷新后重试；若涉及发邮件/微信请点击「确认执行」",
                }
            ]
        chat_store.save_message(
            session_id,
            "assistant",
            reply,
            mcp_attachments=extract_mcp_attachments_from_messages(msgs) or None,
        )
        _maybe_schedule_auto_title(session_id)
        return [
            build_stream_done_payload(
                session_id=session_id,
                reply=reply,
                msgs=msgs,
                rag_context=rag_context,
                rag_mode=rag_mode,
                include_tool_debug=include_tool_debug,
                agent_system_prompt=agent_system_prompt,
                langsmith_trace=langsmith_trace,
            )
        ]

    return StreamingResponse(
        _stream_agent_events(
            _stream_with_summary(), on_result=on_result, run_key=session_id
        ),
        media_type="text/event-stream; charset=utf-8",
        headers=_SSE_HEADERS,
    )


@router.post("/chat/cancel")
async def chat_cancel(request: Request):
    """用户主动打断：取消正在执行的 Agent，并清理 checkpoint。"""
    data = await request.json() or {}
    session_id = (data.get("session_id") or "default").strip() or "default"
    cancelled = cancel_active_run(session_id)
    try:
        from agent.agent_checkpointer import reset_agent_thread

        schedule_async(reset_agent_thread(session_id))
    except Exception as e:
        if not cancelled:
            return {
                "code": -1,
                "msg": f"停止失败：{format_error(e)}",
                "cancelled": False,
                "session_id": session_id,
            }
    return {
        "code": 0,
        "msg": "已停止执行",
        "cancelled": cancelled,
        "session_id": session_id,
    }


@router.post("/chat/hitl/resume")
async def chat_hitl_resume(request: Request):
    """用户在前端确认/取消后，恢复 LangGraph checkpoint 继续 Agent。"""
    data = await request.json() or {}
    session_id = data.get("session_id") or "default"
    action = (data.get("action") or "").strip().lower()
    if action not in ("approve", "reject"):
        return {"code": -1, "msg": "action 须为 approve 或 reject"}
    include_tool_debug = bool(data.get("include_tool_debug"))
    rag_mode = rag_service.normalize_rag_mode(data.get("rag_mode"))
    rag_query = (data.get("rag_query") or "").strip() or "请继续完成上一轮任务"
    file_ids = data.get("file_ids") or []
    if isinstance(file_ids, str):
        file_ids = [file_ids] if file_ids else []
    file_ids = [str(x).strip() for x in file_ids if str(x).strip()]
    rag_context = rag_service.build_rag_context(
        session_id, rag_query, file_ids=file_ids or None, mode=rag_mode
    )
    agent_system_prompt = chat_agent_prompt_with_rag(rag_context)
    llm_config = _extract_llm_config(data)
    hitl_enabled = _extract_hitl_enabled(data)
    try:
        reply, msgs, hitl_pending = await run_cancellable(
            run_agent_hitl_resume(
                session_id,
                action,
                rag_context=rag_context,
                log_prompt=LOG_LLM_PROMPT or include_tool_debug,
                llm_config=llm_config,
                hitl_enabled=hitl_enabled,
            ),
            run_key=session_id,
        )
        langsmith_trace = finalize_trace_for_session(session_id)
        if hitl_pending:
            pending = hitl_pending[0] if hitl_pending else {}
            hitl_out = {
                "code": 0,
                "status": "hitl_pending",
                "session_id": session_id,
                "rag_mode": rag_mode,
                "hitl": {
                    "pending": hitl_pending,
                    "tool": pending.get("tool"),
                    "label": pending.get("label"),
                    "summary": pending.get("summary"),
                    "args": pending.get("args"),
                },
                "msg": "等待您确认是否执行敏感操作",
            }
            if langsmith_trace:
                hitl_out["langsmith"] = langsmith_trace
            return hitl_out
        if not reply:
            return {"code": -1, "msg": "模型未返回有效回复"}
        await asyncio.to_thread(
            chat_store.save_message,
            session_id,
            "assistant",
            reply,
            mcp_attachments=extract_mcp_attachments_from_messages(msgs) or None,
        )
        _maybe_schedule_auto_title(session_id)
        return _build_chat_success_payload(
            session_id,
            reply,
            msgs,
            rag_context=rag_context,
            rag_mode=rag_mode,
            include_tool_debug=include_tool_debug,
            agent_system_prompt=agent_system_prompt,
            langsmith_trace=langsmith_trace,
        )
    except asyncio.CancelledError:
        return {
            "code": 0,
            "status": "cancelled",
            "msg": "执行已由用户中断",
            "session_id": session_id,
        }
    except Exception as e:
        return {"code": -1, "msg": f"HITL 恢复失败：{format_error(e)}"}


@router.post("/chat/hitl/resume/stream")
async def chat_hitl_resume_stream(request: Request):
    """SSE：HITL 确认/取消后继续流式 Agent。"""
    data = await request.json() or {}
    session_id = data.get("session_id") or "default"
    action = (data.get("action") or "").strip().lower()
    if action not in ("approve", "reject"):
        return {"code": -1, "msg": "action 须为 approve 或 reject"}
    include_tool_debug = bool(data.get("include_tool_debug"))
    rag_mode = rag_service.normalize_rag_mode(data.get("rag_mode"))
    rag_query = (data.get("rag_query") or "").strip() or "请继续完成上一轮任务"
    file_ids = data.get("file_ids") or []
    if isinstance(file_ids, str):
        file_ids = [file_ids] if file_ids else []
    file_ids = [str(x).strip() for x in file_ids if str(x).strip()]
    rag_context = rag_service.build_rag_context(
        session_id, rag_query, file_ids=file_ids or None, mode=rag_mode
    )
    agent_system_prompt = chat_agent_prompt_with_rag(rag_context)
    llm_config = _extract_llm_config(data)
    hitl_enabled = _extract_hitl_enabled(data)

    async_gen = stream_agent_hitl_resume(
        session_id,
        action,
        rag_context=rag_context,
        log_prompt=LOG_LLM_PROMPT or include_tool_debug,
        llm_config=llm_config,
        hitl_enabled=hitl_enabled,
        rag_mode=rag_mode,
    )

    def on_result(event: dict):
        if event.get("_error"):
            return [{"type": "error", "code": -1, "msg": event["_error"]}]
        reply = event.get("reply")
        msgs = event.get("messages") or []
        hitl_pending = event.get("hitl")
        langsmith_trace = event.get("langsmith")
        if hitl_pending:
            return [
                build_stream_hitl_payload(
                    session_id=session_id,
                    hitl_pending=hitl_pending,
                    rag_mode=rag_mode,
                    langsmith_trace=langsmith_trace,
                )
            ]
        if not reply:
            return [
                {
                    "type": "error",
                    "code": -1,
                    "msg": "Agent 在工具调用阶段中断且未完成，请刷新后重试；若涉及发邮件/微信请点击「确认执行」",
                }
            ]
        chat_store.save_message(
            session_id,
            "assistant",
            reply,
            mcp_attachments=extract_mcp_attachments_from_messages(msgs) or None,
        )
        _maybe_schedule_auto_title(session_id)
        return [
            build_stream_done_payload(
                session_id=session_id,
                reply=reply,
                msgs=msgs,
                rag_context=rag_context,
                rag_mode=rag_mode,
                include_tool_debug=include_tool_debug,
                agent_system_prompt=agent_system_prompt,
                langsmith_trace=langsmith_trace,
            )
        ]

    try:
        return StreamingResponse(
            _stream_agent_events(async_gen, on_result=on_result, run_key=session_id),
            media_type="text/event-stream; charset=utf-8",
            headers=_SSE_HEADERS,
        )
    except Exception as e:
        return {"code": -1, "msg": f"HITL 流式恢复失败：{format_error(e)}"}


@router.post("/ai/run")
async def ai_run(request: Request):
    prompt = (await request.json() or {}).get("prompt", "")
    if not prompt:
        return {"code": -1, "msg": "请输入指令"}
    try:
        result = await run_agent(prompt)
        return {"code": 0, "msg": result}
    except Exception as e:
        return {"code": -1, "msg": f"执行失败：{format_error(e)}"}
