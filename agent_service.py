"""LangGraph ReAct Agent 与 MCP 工具集成。"""
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.prebuilt import create_react_agent
from mcp import ClientSession, types
from mcp.client.streamable_http import streamable_http_client

from agent_checkpointer import get_checkpointer, reset_agent_thread
from app_config import LOG_LLM_PROMPT_MAX, MCP_URL, logger
from chat_helpers import last_assistant_text
from llm_zhipu import make_chat_llm
import mcp_lifecycle

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

CHAT_OFFLINE_PROMPT_SUFFIX = (
    "\n（说明：当前未连接 MCP 工具服务，你只能根据对话与附件内容作答，"
    "不能发微信、发邮件、联网搜索或导出 Excel；若用户要求这些能力，请说明需先启动 MCP。）"
)


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
    return [
        MCPTool(
            session=session,
            name=t.name,
            description=t.description or "",
            args_schema=t.inputSchema,
        )
        for t in defs
    ]


async def _invoke_react_agent(
    llm,
    tools: list,
    lc_messages: list,
    *,
    rag_context: str | None,
    session_id: str,
):
    """创建 ReAct Agent 并 invoke；若已配置 PostgresSaver 则挂 checkpointer。"""
    checkpointer = get_checkpointer()
    prompt = chat_agent_prompt_with_rag(rag_context)
    if checkpointer is not None:
        await reset_agent_thread(session_id)
        agent = create_react_agent(llm, tools, prompt=prompt, checkpointer=checkpointer)
        config = {"configurable": {"thread_id": session_id}}
        return await agent.ainvoke({"messages": lc_messages}, config=config)
    agent = create_react_agent(llm, tools, prompt=prompt)
    return await agent.ainvoke({"messages": lc_messages})


async def run_agent(prompt: str) -> str:
    if not mcp_lifecycle.ensure_mcp_server_started():
        from app_config import MCP_HOST, MCP_PORT

        raise RuntimeError(
            f"MCP 未在 {MCP_HOST}:{MCP_PORT} 就绪，请另开终端运行: python mcp_server.py"
        )
    llm = make_chat_llm()
    async with streamable_http_client(MCP_URL) as (r, w, _):
        async with ClientSession(r, w) as session:
            tools = await langchain_tools_from_mcp_session(session)
            checkpointer = get_checkpointer()
            if checkpointer is not None:
                agent = create_react_agent(llm, tools, checkpointer=checkpointer)
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
) -> tuple[str, list]:
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
    return last_assistant_text(msgs), msgs


async def run_agent_with_history(
    lc_messages: list,
    rag_context: str | None = None,
    *,
    session_id: str = "",
    log_prompt: bool = False,
) -> tuple[str, list]:
    """带完整上下文的 Agent；返回 (助手可见文本, 完整消息列表供工具调试)。"""
    from app_utils import format_error

    if not mcp_lifecycle.ensure_mcp_server_started():
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
        async with streamable_http_client(MCP_URL) as (r, w, _):
            async with ClientSession(r, w) as session:
                tools = await langchain_tools_from_mcp_session(session)
                state = await _invoke_react_agent(
                    llm,
                    tools,
                    lc_messages,
                    rag_context=rag_context,
                    session_id=session_id,
                )
                msgs = state.get("messages") or []
                return last_assistant_text(msgs), msgs
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


async def send_wechat_agent(name: str, content: str) -> None:
    llm = make_chat_llm()
    async with streamable_http_client(MCP_URL) as (r, w, _):
        async with ClientSession(r, w) as session:
            tools = await langchain_tools_from_mcp_session(session)
            agent = create_react_agent(llm, tools)
            await agent.ainvoke(
                {"messages": [HumanMessage(content=f"给微信名称{name}发消息：{content}")]}
            )


async def send_email_agent(to: str, content: str) -> None:
    llm = make_chat_llm()
    async with streamable_http_client(MCP_URL) as (r, w, _):
        async with ClientSession(r, w) as session:
            tools = await langchain_tools_from_mcp_session(session)
            agent = create_react_agent(llm, tools)
            await agent.ainvoke(
                {"messages": [HumanMessage(content=f"给邮箱{to}发送内容：{content}")]}
            )
