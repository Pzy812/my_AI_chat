"""MCP 服务启停与状态。"""
import asyncio
import time

from fastapi import APIRouter

import app_mcp.mcp_lifecycle as mcp_lifecycle
from agent.agent_checkpointer import checkpointer_kind, enabled as checkpointer_ready
from agent.agent_service import hitl_available
from config.app_config import AGENT_CHECKPOINT_ENABLED, HITL_ENABLED, LANGSMITH_STATUS, POSTGRES_URI

router = APIRouter(tags=["service"])


def _status_payload() -> dict:
    running = mcp_lifecycle.tracked_mcp_running() or bool(mcp_lifecycle.mcp_port_pids())
    cp_kind = checkpointer_kind()
    return {
        "code": 0,
        "running": running,
        "hitl_enabled": HITL_ENABLED,
        "hitl_available": hitl_available(),
        "checkpointer": cp_kind,
        "checkpointer_ready": checkpointer_ready(),
        "postgres_configured": bool(POSTGRES_URI),
        "agent_checkpoint_enabled": AGENT_CHECKPOINT_ENABLED,
        "langsmith": LANGSMITH_STATUS,
    }


@router.api_route("/service/status", methods=["GET", "POST"])
async def service_status():
    return _status_payload()


@router.post("/service/start")
async def service_start():
    try:
        await asyncio.to_thread(mcp_lifecycle.restart_mcp_server)
        mcp_lifecycle.mcp_process = await asyncio.to_thread(mcp_lifecycle.start_mcp_subprocess)
        deadline = time.time() + 25
        while time.time() < deadline:
            if mcp_lifecycle.mcp_port_open() and (
                await mcp_lifecycle.mcp_wechat_tools_status_async()
            ) == "ok":
                return {"code": 0, "msg": "MCP 服务启动成功"}
            if (
                mcp_lifecycle.mcp_process is not None
                and mcp_lifecycle.mcp_process.poll() is not None
            ):
                return {"code": -1, "msg": "MCP 子进程启动失败，请查看终端日志"}
            await asyncio.sleep(0.35)
        return {"code": -1, "msg": "MCP 启动超时或未加载微信工具，请检查 mcp_server.py 日志"}
    except Exception as e:
        return {"code": -1, "msg": f"启动失败：{str(e)}"}


@router.post("/service/stop")
async def service_stop():
    try:
        await asyncio.to_thread(mcp_lifecycle.stop_mcp_subprocess)
        await asyncio.to_thread(mcp_lifecycle.kill_mcp_port_processes)
        return {"code": 0, "msg": "MCP 服务已停止"}
    except Exception as e:
        return {"code": -1, "msg": f"停止失败：{str(e)}"}
