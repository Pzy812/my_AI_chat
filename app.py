"""Flask 应用入口：创建 app 并启动 Web 服务。"""
import sys

# Windows 下 psycopg 需要 SelectorEventLoop（须在创建任何 loop 之前）
if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from flask import Flask
from flask_cors import CORS

from app_config import BASE_DIR, FLASK_HOST, FLASK_PORT, MCP_HOST, MCP_PORT, MCP_URL
from async_runner import setup_async_services
import mcp_lifecycle
from routes import register_routes


def create_app() -> Flask:
    application = Flask(__name__, template_folder=str(BASE_DIR / "template"))
    CORS(application)
    from chat_store import init_chat_store

    init_chat_store()
    setup_async_services()
    register_routes(application)
    return application


app = create_app()


if __name__ == "__main__":
    if mcp_lifecycle.ensure_mcp_server_started():
        print(f"MCP 已就绪: {MCP_URL}")
    else:
        print(
            f"警告: MCP 未在 {MCP_PORT} 端口就绪，对话将降级为纯模型（附件问答仍可用）。"
            " 可手动运行: python mcp_server.py"
        )
    app.run(host=FLASK_HOST, port=FLASK_PORT, debug=False, use_reloader=False)
