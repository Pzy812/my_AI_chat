"""微信 MCP 工具（基于 test_wx：COM 专用线程 + wx_patch + 搜索框打开会话）。"""
from __future__ import annotations

import os
import shutil
import sys
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, TypeVar

T = TypeVar("T")

_wx_executor: ThreadPoolExecutor | None = None
_wx_client = None
_wx_unavailable_reason: str | None = None


def _platform_supported() -> bool:
    return sys.platform == "win32"


def _ensure_com() -> None:
    if not _platform_supported():
        return
    import pythoncom

    try:
        pythoncom.CoInitialize()
    except pythoncom.com_error:
        pass


def _ensure_executor() -> ThreadPoolExecutor:
    global _wx_executor
    if _wx_executor is None:
        _wx_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="wechat")
    return _wx_executor


def _get_wechat():
    """Windows + 本机微信可用时返回 WeChat 实例；否则抛出 RuntimeError。"""
    global _wx_client, _wx_unavailable_reason
    if not _platform_supported():
        raise RuntimeError("微信 UI 自动化仅支持 Windows")
    if _wx_unavailable_reason is not None:
        raise RuntimeError(_wx_unavailable_reason)
    if _wx_client is not None:
        return _wx_client
    try:
        import app_mcp.wx_patch  # noqa: F401  必须在 wxauto4 之前加载
        from wxauto4 import WeChat

        _ensure_com()
        _wx_client = WeChat(ads=False)
        time.sleep(0.5)
        return _wx_client
    except Exception as e:
        _wx_unavailable_reason = str(e)
        raise RuntimeError(str(e)) from e


def run_wechat(action: Callable[[object], T]) -> T:
    """在专用 COM 线程中执行微信操作。"""

    def task() -> T:
        wx = _get_wechat()
        return action(wx)

    return _ensure_executor().submit(task).result(timeout=120)


def wechat_unavailable_message() -> str:
    reason = _wx_unavailable_reason or "未知"
    if not _platform_supported():
        reason = "微信 UI 自动化仅支持 Windows（Docker/Linux 无桌面）"
    return (
        "❌ 微信不可用（常见于 Docker/Linux 无桌面，或未安装 wxauto4 / pywin32）。"
        f" 原因: {reason}"
    )


def open_chat(wx, to_name: str) -> None:
    """打开聊天窗口，优先用搜索框避免会话列表解析问题。"""
    wx.ChatWith(to_name, force=True, force_wait=1.0)


def validate_file_paths(file_paths: list[str]) -> list[str]:
    missing = [p for p in file_paths if not Path(p).is_file()]
    if missing:
        raise FileNotFoundError(f"文件不存在: {', '.join(missing)}")
    return [str(Path(p).resolve()) for p in file_paths]


def _check_wx_response(result, action: str) -> None:
    """校验 wxauto4 WxResponse；None 或非成功状态均视为失败。"""
    if result is None:
        raise RuntimeError(f"{action} 未返回结果（可能聊天窗口未就绪或文件未发出）")
    if hasattr(result, "is_success"):
        if not result.is_success:
            msg = getattr(result, "get", lambda k, d=None: d)("message", None)
            raise RuntimeError(msg or str(result))
        return
    if isinstance(result, dict):
        if result.get("status") != "成功":
            raise RuntimeError(result.get("message") or str(result))
        return
    if result is False:
        raise RuntimeError(f"{action} 返回失败")
    # 非 WxResponse 且无明确失败信号时继续（兼容旧版返回值）


def _prepare_send_paths(paths: list[str]) -> tuple[list[str], list[Path]]:
    """wxauto4 SendFiles 对含中文等非 ASCII 路径可能假成功，复制到 ASCII 临时目录再发。"""
    send_paths: list[str] = []
    temp_files: list[Path] = []
    tmp_dir = Path(tempfile.gettempdir()) / "wechat_mcp_send"
    for p in paths:
        src = Path(p).resolve()
        if str(src).isascii():
            send_paths.append(str(src))
            continue
        tmp_dir.mkdir(parents=True, exist_ok=True)
        dest = tmp_dir / src.name
        shutil.copy2(src, dest)
        temp_files.append(dest)
        send_paths.append(str(dest.resolve()))
    return send_paths, temp_files


def _send_files(wx, to_name: str, paths: list[str]):
    """打开会话并发送文件；与发消息同样只走 open_chat，避免重复 ChatWith 切错窗口。"""
    send_paths, temp_files = _prepare_send_paths(paths)
    try:
        open_chat(wx, to_name)
        time.sleep(0.8)
        result = wx.SendFiles(send_paths, who=to_name)
        _check_wx_response(result, "SendFiles")
        return result
    finally:
        for fp in temp_files:
            try:
                fp.unlink(missing_ok=True)
            except OSError:
                pass


def register_wechat_tools(mcp) -> None:
    """向 FastMCP 实例注册微信相关工具。"""

    @mcp.tool(
        name="send_wechat_message",
        description="给微信好友发送文字消息（需用户在前端确认后才会真正发送）",
    )
    def send_wechat_message(to_name: str, content: str):
        try:

            def action(wx) -> str:
                open_chat(wx, to_name)
                wx.SendMsg(content)
                return f"✅ 成功发送消息到【{to_name}】：{content}"

            return run_wechat(action)
        except Exception as e:
            if _wx_unavailable_reason or not _platform_supported():
                return wechat_unavailable_message()
            return f"❌ 发送失败：{str(e)}"

    @mcp.tool(
        name="get_wechat_messages",
        description="获取指定微信好友聊天窗口的最近消息（只读，无需用户确认）",
    )
    def get_wechat_messages(to_name: str, count: int = 20):
        try:

            def action(wx):
                wx.ChatWith(to_name)
                time.sleep(0.5)
                messages = wx.GetAllMessage()
                messages = messages[-count:]
                return [
                    {
                        "sender": getattr(msg, "sender", "未知"),
                        "content": msg.content,
                    }
                    for msg in messages
                ]

            return run_wechat(action)
        except Exception as e:
            if _wx_unavailable_reason or not _platform_supported():
                return wechat_unavailable_message()
            return f"❌ 获取消息失败：{str(e)}"

    @mcp.tool(
        name="send_wechat_files",
        description="给微信好友发送文件（需用户在前端确认后才会真正发送）",
    )
    def send_wechat_files(to_name: str, file_paths: list[str]):
        try:
            paths = validate_file_paths(file_paths)

            def action(wx) -> str:
                _send_files(wx, to_name, paths)
                names = ", ".join(os.path.basename(p) for p in paths)
                return f"✅ 成功发送 {len(paths)} 个文件到【{to_name}】：{names}"

            return run_wechat(action)
        except FileNotFoundError as e:
            return f"❌ 发送文件失败：{str(e)}"
        except Exception as e:
            if _wx_unavailable_reason or not _platform_supported():
                return wechat_unavailable_message()
            return f"❌ 发送文件失败：{str(e)}"
