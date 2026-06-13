"""LangGraph ReAct Agent 与 MCP 工具集成。"""
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from langgraph.types import Command
from mcp import ClientSession, types

from app_mcp.mcp_http_client import open_mcp_transport
from agent.agent_checkpointer import get_checkpointer, reset_agent_thread
from config.app_config import HITL_ENABLED, LOG_LLM_PROMPT_MAX, logger
from chat.chat_helpers import last_assistant_text, messages_have_pending_tool_calls
from agent.hitl_tools import normalize_interrupts, wrap_tools_with_hitl
from agent.harness import (
    TASK_DISCIPLINE_PROMPT,
    build_initial_agent_state,
    make_post_model_hook,
    make_pre_model_hook,
    wrap_tools_with_phase_gate,
)
from agent.task_state import TaskHarnessState
from llm.llm_zhipu import make_chat_llm
import app_mcp.mcp_lifecycle as mcp_lifecycle

CHAT_AGENT_PROMPT = (
    "你是智能助手，能记住当前对话里用户说过的话。\n"
    "用户消息中「--- 附件 [...] ---」区块是系统已解析的上传文件（PDF/Office 已提取正文，图片经 GLM-4V），请直接基于附件内容回答。\n"
    "若系统额外提供了「知识库检索结果」或「GraphRAG 混合检索结果」，说明已从 Milvus / Neo4j 做了文档检索；请优先依据检索结果作答，并可在必要时结合附件全文。\n"
    "用户仅询问已上传文档/文章时，直接根据知识库检索结果与对话内容回答，不要调用 web_search 等联网工具。\n"
    "需要发微信或发邮件时，分别使用 send_wechat_message、send_email 工具（执行前会由用户在前端确认）。\n"
    "需要查看某好友最近聊天记录时用 get_wechat_messages(to_name, count)；发文件到微信用 send_wechat_files(to_name, file_paths)（file_paths 须为 MCP 服务端本机存在的绝对路径，执行前需用户确认）。\n"
    "读取本机文件夹/文件：先用 list_local_directory 或 glob_local_files 列出真实绝对路径，再用 read_local_file 读内容；不要把猜测的路径直接传给 send_wechat_files。\n"
    "用户要把某目录下全部文件发微信时：glob_local_files(directory, pattern='*', recursive=True) → 取 files[].path → send_wechat_files。\n"
    "询问当前日期、时间、星期几、今天几号等，必须调用 get_current_time（本机时间，无需联网），禁止为此调用 web_search。\n"
    "涉及时效、新闻、股价、黄金/汇率/商品价格、天气、政策等需要联网核实时，必须先调用 web_search（系统会以本机时间为检索基准；也可先 get_current_time 再搜索），再基于搜索结果回答。\n"
    "如果 web_search 工具可用，不要回答“没有实时查询能力”或让用户自行去网站查询。\n"
    "用户要表格展示时用 format_pretty_table；明确要求导出 / 保存为 Excel 时用 export_to_excel，并传入表头 headers 与二维 rows（表格/Excel 执行前需用户确认）。\n"
    "纯聊天可直接回答。"
) + TASK_DISCIPLINE_PROMPT

CHAT_OFFLINE_PROMPT_SUFFIX = (
    "\n（说明：当前未连接 MCP 工具服务，你只能根据对话与附件内容作答，"
    "不能发微信、发邮件、联网搜索或导出 Excel；若用户要求这些能力，请说明需先启动 MCP。）"
)


def hitl_available() -> bool:
    return bool(HITL_ENABLED and get_checkpointer() is not None)


def chat_agent_prompt_with_rag(rag_context: str | None) -> str:
    prompt = CHAT_AGENT_PROMPT
    if rag_context and rag_context.strip():
        prompt = f"{prompt}\n\n{rag_context.strip()}"
    return prompt


def clip_for_log(text: str, max_len: int | None = None) -> str:
    cap = max_len if max_len is not None else LOG_LLM_PROMPT_MAX
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n…(日志已截断，全文 {len(text)} 字符，可调大 LOG_LLM_PROMPT_MAX)"


def log_llm_system_prompt(
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
        clip_for_log(system_prompt),
        sep,
    )
    if rag_context is not None:
        logger.info(
            "[GraphRAG context only] session_id=%s len=%s\n%s",
            session_id or "-",
            len(rag_context),
            clip_for_log(rag_context or "(empty)"),
        )


def prompt_debug_payload(system_prompt: str, rag_context: str | None) -> dict:
    return {
        "system_prompt_length": len(system_prompt),
        "system_prompt": clip_for_log(system_prompt, 8000),
        "rag_context_length": len(rag_context or ""),
        "rag_context": clip_for_log(rag_context or "", 8000),
        "note": "以上内容仅当次请求注入模型，不会存入 Postgres/Redis 对话记录",
    }


async def langchain_tools_from_mcp_session(session: ClientSession):
    """拉全量 MCP 工具（分页 list_tools，避免只读到第一页）。"""
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
    tools = [
        MCPTool(
            session=session,
            name=t.name,
            description=t.description or "",
            args_schema=t.inputSchema,
        )
        for t in defs
    ]
    tools = wrap_tools_with_phase_gate(wrap_tools_with_hitl(tools, enabled=hitl_available()))
    return tools


async def _create_agent(llm, tools: list, rag_context: str | None):
    prompt = chat_agent_prompt_with_rag(rag_context)
    checkpointer = get_checkpointer()
    kwargs: dict = {
        "model": llm,
        "tools": tools,
        "prompt": prompt,
        "state_schema": TaskHarnessState,
        "pre_model_hook": make_pre_model_hook(),
        "post_model_hook": make_post_model_hook(),
        "version": "v2",
    }
    if checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
        return create_react_agent(**kwargs)
    return create_react_agent(**kwargs)


async def _invoke_agent(
    agent,
    *,
    session_id: str,
    lc_messages: list | None = None,
    resume_action: str | None = None,
    fresh_thread: bool = True,
    file_count: int = 0,
) -> tuple[str | None, list, list[dict] | None]:
    """返回 (reply_text|None, messages, hitl_pending|None)。"""
    from config.app_config import AGENT_RECURSION_LIMIT

    checkpointer = get_checkpointer()
    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": AGENT_RECURSION_LIMIT,
    } if checkpointer else {"recursion_limit": AGENT_RECURSION_LIMIT}
    if checkpointer and fresh_thread and resume_action is None:
        await reset_agent_thread(session_id)
    if resume_action is not None:
        if checkpointer:
            try:
                from agent.harness import sync_run_context_from_values

                snap = await agent.aget_state(config)
                if snap and snap.values:
                    sync_run_context_from_values(session_id, dict(snap.values))
            except Exception:
                pass
        state = await agent.ainvoke(
            Command(resume={"action": resume_action}), config=config
        )
    else:
        input_state = await build_initial_agent_state(
            lc_messages or [],
            session_id=session_id,
            file_count=file_count,
        )
        state = await agent.ainvoke(input_state, config=config)
    hitl = normalize_interrupts(state.get("__interrupt__"))
    msgs = state.get("messages") or []
    pending = messages_have_pending_tool_calls(msgs)
    if hitl and not pending:
        hitl = []
    elif not hitl and pending:
        from agent.agent_state import synthetic_hitl_from_messages

        hitl = synthetic_hitl_from_messages(msgs) or None
    if hitl:
        return None, msgs, hitl
    if pending:
        return None, msgs, None
    return last_assistant_text(msgs), msgs, None


async def _invoke_react_agent(
    llm,
    tools: list,
    lc_messages: list,
    *,
    rag_context: str | None,
    session_id: str,
    resume_action: str | None = None,
    fresh_thread: bool = True,
    file_count: int = 0,
) -> tuple[str | None, list, list[dict] | None]:
    agent = await _create_agent(llm, tools, rag_context)
    return await _invoke_agent(
        agent,
        session_id=session_id,
        lc_messages=lc_messages,
        resume_action=resume_action,
        fresh_thread=fresh_thread,
        file_count=file_count,
    )


async def run_agent(prompt: str) -> str:
    if not await mcp_lifecycle.ensure_mcp_server_started_async():
        from config.app_config import MCP_HOST, MCP_PORT

        raise RuntimeError(
            f"MCP 未在 {MCP_HOST}:{MCP_PORT} 就绪，请另开终端运行: python mcp_server.py"
        )
    llm = make_chat_llm()
    async with open_mcp_transport() as (r, w, _):
        async with ClientSession(r, w) as session:
            tools = await langchain_tools_from_mcp_session(session)
            checkpointer = get_checkpointer()
            if checkpointer is not None:
                agent = await _create_agent(llm, tools, None)
                config = {"configurable": {"thread_id": "ai_run"}}
                state = await agent.ainvoke(
                    {"messages": [HumanMessage(content=prompt)]}, config=config
                )
            else:
                agent = create_react_agent(llm, tools)
                state = await agent.ainvoke({"messages": [HumanMessage(content=prompt)]})
            return last_assistant_text(state["messages"])


async def run_chat_llm_only(
    lc_messages: list,
    rag_context: str | None = None,
    *,
    session_id: str = "",
    log_prompt: bool = False,
) -> tuple[str | None, list, list[dict] | None]:
    """MCP 不可用时：直接用 GLM 对话（附件正文已在消息里）。"""
    llm = make_chat_llm()
    prompt = chat_agent_prompt_with_rag(rag_context) + CHAT_OFFLINE_PROMPT_SUFFIX
    if log_prompt:
        log_llm_system_prompt(
            "llm_only_offline",
            prompt,
            session_id=session_id,
            rag_context=rag_context,
        )
    resp = await llm.ainvoke([SystemMessage(content=prompt)] + list(lc_messages))
    msgs = list(lc_messages) + [resp]
    return last_assistant_text(msgs), msgs, None


async def run_agent_with_history(
    lc_messages: list,
    rag_context: str | None = None,
    *,
    session_id: str = "",
    log_prompt: bool = False,
    file_count: int = 0,
) -> tuple[str | None, list, list[dict] | None]:
    """带完整上下文的 Agent；返回 (助手文本|None, 消息列表, HITL pending|None)。"""
    from core.app_utils import format_error

    if not await mcp_lifecycle.ensure_mcp_server_started_async():
        return await run_chat_llm_only(
            lc_messages,
            rag_context=rag_context,
            session_id=session_id,
            log_prompt=log_prompt,
        )
    try:
        if log_prompt:
            log_llm_system_prompt(
                "react_agent",
                chat_agent_prompt_with_rag(rag_context),
                session_id=session_id,
                rag_context=rag_context,
            )
        llm = make_chat_llm()
        async with open_mcp_transport() as (r, w, _):
            async with ClientSession(r, w) as session:
                tools = await langchain_tools_from_mcp_session(session)
                return await _invoke_react_agent(
                    llm,
                    tools,
                    lc_messages,
                    rag_context=rag_context,
                    session_id=session_id,
                    fresh_thread=True,
                    file_count=file_count,
                )
    except BaseException as e:
        logger.warning(
            "Agent+MCP 调用失败，已降级为纯 LLM（本轮不会出现 ToolMessage）：%s",
            format_error(e),
        )
        return await run_chat_llm_only(
            lc_messages,
            rag_context=rag_context,
            session_id=session_id,
            log_prompt=log_prompt,
        )


async def run_agent_hitl_resume(
    session_id: str,
    action: str,
    *,
    rag_context: str | None = None,
    log_prompt: bool = False,
) -> tuple[str | None, list, list[dict] | None]:
    """用户确认/取消后，从 checkpoint 继续 Agent（不再 reset thread）。"""
    if not await mcp_lifecycle.ensure_mcp_server_started_async():
        raise RuntimeError("MCP 未就绪，无法恢复 HITL 会话")
    if not hitl_available():
        raise RuntimeError("HITL 未启用或未配置 Postgres Checkpointer")
    if log_prompt:
        log_llm_system_prompt(
            "react_agent_hitl_resume",
            chat_agent_prompt_with_rag(rag_context),
            session_id=session_id,
            rag_context=rag_context,
        )
    llm = make_chat_llm()
    async with open_mcp_transport() as (r, w, _):
        async with ClientSession(r, w) as session:
            tools = await langchain_tools_from_mcp_session(session)
            return await _invoke_react_agent(
                llm,
                tools,
                [],
                rag_context=rag_context,
                session_id=session_id,
                resume_action=action,
                fresh_thread=False,
            )


async def send_wechat_agent(name: str, content: str) -> None:
    llm = make_chat_llm()
    async with open_mcp_transport() as (r, w, _):
        async with ClientSession(r, w) as session:
            tools = await langchain_tools_from_mcp_session(session)
            agent = create_react_agent(llm, tools)
            await agent.ainvoke(
                {
                    "messages": [
                        HumanMessage(
                            content=(
                                f"请使用 send_wechat_message 工具，"
                                f"给微信好友【{name}】发送消息：{content}"
                            )
                        )
                    ]
                }
            )


async def send_email_agent(to: str, content: str) -> None:
    llm = make_chat_llm()
    async with open_mcp_transport() as (r, w, _):
        async with ClientSession(r, w) as session:
            tools = await langchain_tools_from_mcp_session(session)
            agent = create_react_agent(llm, tools)
            await agent.ainvoke(
                {"messages": [HumanMessage(content=f"给邮箱{to}发送内容：{content}")]}
            )
