"""Flask 应用入口：创建 app 并启动 Web 服务。"""
import asyncio
import sys

# Windows 强制使用 SelectorEventLoop（解决 psycopg 不兼容）
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from flask import Flask
from flask_cors import CORS

from app_config import BASE_DIR, FLASK_HOST, FLASK_PORT, MCP_HOST, MCP_PORT, MCP_URL
import mcp_lifecycle
from routes import register_routes


def create_app() -> Flask:
    application = Flask(__name__, template_folder=str(BASE_DIR / "template"))
    CORS(application)
    from chat_store import init_chat_store

    init_chat_store()
    _ensure_agent_checkpointer()
    register_routes(application)
    return application


def _ensure_agent_checkpointer() -> None:
    from agent_checkpointer import init_checkpointer

    try:
        asyncio.get_running_loop()
        return
    except RuntimeError:
        pass
    try:
        asyncio.run(init_checkpointer())
    except RuntimeError:
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(init_checkpointer())
        finally:
            loop.close()


app = create_app()


def _startup_async_services() -> None:
    """保留供外部 WSGI 启动脚本调用（create_app 已自动初始化）。"""
    _ensure_agent_checkpointer()


if __name__ == "__main__":
    if mcp_lifecycle.ensure_mcp_server_started():
        print(f"MCP 已就绪: {MCP_URL}")
    else:
        print(
            f"警告: MCP 未在 {MCP_PORT} 端口就绪，对话将降级为纯模型（附件问答仍可用）。"
            " 可手动运行: python mcp_server.py"
        )
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, use_reloader=False)
