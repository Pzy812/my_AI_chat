"""Flask 应用入口：创建 app 并启动 Web 服务。"""
import os
import sys
# 今天上海嘉定天气怎么样，把最近一周结果发送给971662861@qq.com，并且把最近agent相关的哪一些地方做改进的发给他，并且给他发送一些生日祝福语，并且把最近一周上海嘉定天气作成表格存储到本地
# Windows 系统代理常导致 httpx 无法直连 localhost（MCP/Redis 等）
_NO_PROXY_HOSTS = "localhost,127.0.0.1,0.0.0.0"
for _proxy_key in ("NO_PROXY", "no_proxy"):
    _cur = os.environ.get(_proxy_key, "")
    if "127.0.0.1" not in _cur:
        os.environ[_proxy_key] = f"{_cur},{_NO_PROXY_HOSTS}".strip(",")

# Windows 下 psycopg 需要 SelectorEventLoop（须在创建任何 loop 之前）
if sys.platform == "win32":
    import asyncio

    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

from flask import Flask
from flask_cors import CORS

from config.app_config import BASE_DIR, FLASK_HOST, FLASK_PORT, MCP_HOST, MCP_PORT, MCP_URL
from core.async_runner import setup_async_services
import app_mcp.mcp_lifecycle as mcp_lifecycle
from routes import register_routes


def create_app() -> Flask:
    application = Flask(__name__, template_folder=str(BASE_DIR / "template"))
    CORS(application)
    from chat.chat_store import init_chat_store

    init_chat_store()
    setup_async_services()
    register_routes(application)
    return application


app = create_app()


if __name__ == "__main__":
    from agent.agent_checkpointer import checkpointer_kind
    from agent.agent_service import hitl_available

    if mcp_lifecycle.ensure_mcp_server_started():
        print(f"MCP 已就绪: {MCP_URL}")
    else:
        print(
            f"警告: MCP 未在 {MCP_PORT} 端口就绪，对话将降级为纯模型（附件问答仍可用）。"
            " 可手动运行: python mcp_server.py"
        )
    cp = checkpointer_kind()
    if hitl_available():
        print(f"Human-in-the-Loop 已启用（checkpointer={cp}，发邮件/微信等需前端确认）")
    else:
        print("警告: Human-in-the-Loop 未启用，敏感 MCP 工具将直接执行（需 HITL_ENABLED=1 且 checkpointer 就绪）")
    app.run(
        host=FLASK_HOST,
        port=FLASK_PORT,
        debug=False,
        use_reloader=False,
        threaded=True,
    )
