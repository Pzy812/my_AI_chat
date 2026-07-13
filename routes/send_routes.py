"""微信 / 邮件直发接口。"""
from fastapi import APIRouter, Request

from agent.agent_service import send_email_agent, send_wechat_agent

router = APIRouter(tags=["send"])


@router.post("/send/wechat")
async def send_wechat(request: Request):
    data = await request.json()
    name = data.get("name", "")
    content = data.get("content", "")
    if not name or not content:
        return {"code": -1, "msg": "参数不完整"}
    try:
        await send_wechat_agent(name, content)
        return {"code": 0, "msg": "微信消息发送成功"}
    except Exception as e:
        return {"code": -1, "msg": f"发送失败：{str(e)}"}


@router.post("/send/email")
async def send_email(request: Request):
    data = await request.json()
    to = data.get("to", "")
    content = data.get("content", "")
    if not to or not content:
        return {"code": -1, "msg": "参数不完整"}
    try:
        await send_email_agent(to, content)
        return {"code": 0, "msg": "邮件发送成功"}
    except Exception as e:
        return {"code": -1, "msg": f"发送失败：{str(e)}"}
