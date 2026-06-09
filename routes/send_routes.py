"""微信 / 邮件直发接口。"""
from flask import Blueprint, jsonify, request

from agent_service import send_email_agent, send_wechat_agent
from async_runner import run_async

bp = Blueprint("send", __name__)


@bp.route("/send/wechat", methods=["POST"])
def send_wechat():
    data = request.get_json()
    name = data.get("name", "")
    content = data.get("content", "")
    if not name or not content:
        return jsonify({"code": -1, "msg": "参数不完整"})
    try:
        run_async(send_wechat_agent(name, content))
        return jsonify({"code": 0, "msg": "微信消息发送成功"})
    except Exception as e:
        return jsonify({"code": -1, "msg": f"发送失败：{str(e)}"})


@bp.route("/send/email", methods=["POST"])
def send_email():
    data = request.get_json()
    to = data.get("to", "")
    content = data.get("content", "")
    if not to or not content:
        return jsonify({"code": -1, "msg": "参数不完整"})
    try:
        run_async(send_email_agent(to, content))
        return jsonify({"code": 0, "msg": "邮件发送成功"})
    except Exception as e:
        return jsonify({"code": -1, "msg": f"发送失败：{str(e)}"})
