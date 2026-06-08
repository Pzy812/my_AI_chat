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
        if not mcp_lifecycle.tracked_mcp_running() and mcp_lifecycle.mcp_port_pids():
            mcp_lifecycle.kill_mcp_port_processes()
            time.sleep(0.5)
        if mcp_lifecycle.mcp_process and mcp_lifecycle.mcp_process.poll() is None:
            return jsonify({"code": 0, "msg": "服务已在运行中"})
        mcp_lifecycle.mcp_process = mcp_lifecycle.start_mcp_subprocess()
        return jsonify({"code": 0, "msg": "MCP 服务启动成功"})
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
