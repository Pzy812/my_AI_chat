"""MCP 服务启停与状态。"""
import time

from flask import Blueprint, jsonify

import mcp_lifecycle

bp = Blueprint("service", __name__)


@bp.route("/service/status", methods=["POST"])
def service_status():
    running = mcp_lifecycle.tracked_mcp_running() or bool(mcp_lifecycle.mcp_port_pids())
    return jsonify({"code": 0, "running": running})


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
