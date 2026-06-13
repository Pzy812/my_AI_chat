"""MCP 服务启停与状态。"""
import time

from flask import Blueprint, jsonify

import app_mcp.mcp_lifecycle as mcp_lifecycle
from agent.agent_checkpointer import checkpointer_kind, enabled as checkpointer_ready
from agent.agent_service import hitl_available
from config.app_config import AGENT_CHECKPOINT_ENABLED, HITL_ENABLED, POSTGRES_URI

bp = Blueprint("service", __name__)


@bp.route("/service/status", methods=["POST", "GET"])
def service_status():
    running = mcp_lifecycle.tracked_mcp_running() or bool(mcp_lifecycle.mcp_port_pids())
    cp_kind = checkpointer_kind()
    return jsonify(
        {
            "code": 0,
            "running": running,
            "hitl_enabled": HITL_ENABLED,
            "hitl_available": hitl_available(),
            "checkpointer": cp_kind,
            "checkpointer_ready": checkpointer_ready(),
            "postgres_configured": bool(POSTGRES_URI),
            "agent_checkpoint_enabled": AGENT_CHECKPOINT_ENABLED,
        }
    )


@bp.route("/service/start", methods=["POST"])
def service_start():
    try:
        mcp_lifecycle.restart_mcp_server()
        mcp_lifecycle.mcp_process = mcp_lifecycle.start_mcp_subprocess()
        deadline = time.time() + 25
        while time.time() < deadline:
            if (
                mcp_lifecycle.mcp_port_open()
                and mcp_lifecycle.mcp_wechat_tools_status() == "ok"
            ):
                return jsonify({"code": 0, "msg": "MCP 服务启动成功"})
            if (
                mcp_lifecycle.mcp_process is not None
                and mcp_lifecycle.mcp_process.poll() is not None
            ):
                return jsonify({"code": -1, "msg": "MCP 子进程启动失败，请查看终端日志"})
            time.sleep(0.35)
        return jsonify({"code": -1, "msg": "MCP 启动超时或未加载微信工具，请检查 mcp_server.py 日志"})
    except Exception as e:
        return jsonify({"code": -1, "msg": f"启动失败：{str(e)}"})


@bp.route("/service/stop", methods=["POST"])
def service_stop():
    try:
        mcp_lifecycle.stop_mcp_subprocess()
        mcp_lifecycle.kill_mcp_port_processes()
        return jsonify({"code": 0, "msg": "MCP 服务已停止"})
    except Exception as e:
        return jsonify({"code": -1, "msg": f"停止失败：{str(e)}"})
