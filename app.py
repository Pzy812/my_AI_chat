"""FastAPI 应用入口：创建 app 并启动 Web 服务。"""
import os
import sys
from contextlib import asynccontextmanager

# 今天上海嘉定天气怎么样，收集最近一周结果，并且把最近agent记忆模块相关的哪一些地方做改进的发给他，并且给他发送一些AI前沿新闻的内容，整理一份邮件发送给971662861@qq.com
_NO_PROXY_HOSTS = "localhost,127.0.0.1,0.0.0.0"
for _proxy_key in ("NO_PROXY", "no_proxy"):
    _cur = os.environ.get(_proxy_key, "")
    if "127.0.0.1" not in _cur:
        os.environ[_proxy_key] = f"{_cur},{_NO_PROXY_HOSTS}".strip(",")

# Windows 下 psycopg 需要 SelectorEventLoop（须在创建任何 loop 之前）
if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# 尽早恢复 asyncio.create_task（PyCharm Console 会破坏 LangGraph 所需的 context= 参数）
import core.async_runner  # noqa: F401, E402

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from config.app_config import APP_HOST, APP_PORT, MCP_PORT, MCP_URL
from core.async_runner import setup_async_services, shutdown_async_services
import app_mcp.mcp_lifecycle as mcp_lifecycle
from routes import register_routes


@asynccontextmanager
async def lifespan(_application: FastAPI):
    from chat.chat_store import init_chat_store

    init_chat_store()
    await setup_async_services()
    yield
    await shutdown_async_services()


def create_app() -> FastAPI:
    application = FastAPI(title="AI Chat", lifespan=lifespan)
    application.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    register_routes(application)
    return application


app = create_app()


if __name__ == "__main__":
    import uvicorn

    from config.app_config import AGENT_CHECKPOINT_ENABLED, HITL_ENABLED, LANGSMITH_STATUS

    if mcp_lifecycle.ensure_mcp_server_started():
        print(f"MCP 已就绪: {MCP_URL}")
    else:
        print(
            f"警告: MCP 未在 {MCP_PORT} 端口就绪，对话将降级为纯模型（附件问答仍可用）。"
            " 可手动运行: python mcp_server.py"
        )

    if HITL_ENABLED and AGENT_CHECKPOINT_ENABLED:
        print(
            "Human-in-the-Loop 已配置启用（checkpointer 将在服务启动时初始化；"
            "发邮件/微信等需前端确认）"
        )
    else:
        print(
            "警告: Human-in-the-Loop 未启用，敏感 MCP 工具将直接执行"
            "（需 HITL_ENABLED=1 且 checkpointer 就绪）"
        )
    if LANGSMITH_STATUS.get("enabled"):
        print(
            f"LangSmith 追踪已启用（project={LANGSMITH_STATUS.get('project')}，"
            f"可在前端查看 Agent trace 或打开 {LANGSMITH_STATUS.get('web_url')}）"
        )
    else:
        print("LangSmith 未启用（可选：在 .env 配置 LANGSMITH_API_KEY 与 LANGSMITH_PROJECT）")

    uvicorn.run(
        "app:app",
        host=APP_HOST,
        port=APP_PORT,
        reload=False,
        log_level="info",
    )
