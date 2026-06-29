"""MCP 子进程与端口生命周期管理。"""
import asyncio
import logging
import os
import socket
import subprocess
import sys
import time

from config.app_config import BASE_DIR, MCP_HOST, MCP_PORT, MCP_URL

logger = logging.getLogger("ai_chat")

mcp_process = None

# 当前代码版本应注册的微信 MCP 工具（用于检测旧进程未重启）
REQUIRED_WECHAT_MCP_TOOLS = frozenset(
    {
        "send_wechat_message",
        "get_wechat_messages",
        "send_wechat_files",
    }
)

def tracked_mcp_running() -> bool:
    return mcp_process is not None and mcp_process.poll() is None


def mcp_port_pids() -> set[int]:
    """Return process ids listening on the MCP port, including old external runs."""
    try:
        out = subprocess.check_output(
            ["netstat", "-ano", "-p", "tcp"],
            text=True,
            errors="ignore",
        )
    except Exception:
        return set()

    pids: set[int] = set()
    needle = f":{MCP_PORT}"
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 5 and parts[0].upper() == "TCP" and parts[3].upper() == "LISTENING":
            if parts[1].endswith(needle):
                try:
                    pids.add(int(parts[-1]))
                except ValueError:
                    pass
    return pids


def kill_mcp_port_processes() -> None:
    current_pid = os.getpid()
    for pid in mcp_port_pids():
        if pid == current_pid:
            continue
        subprocess.run(
            ["taskkill", "/PID", str(pid), "/T", "/F"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        )


def mcp_port_open() -> bool:
    try:
        with socket.create_connection((MCP_HOST, MCP_PORT), timeout=1.5):
            return True
    except OSError:
        return False


async def _fetch_mcp_tool_names_async() -> set[str]:
    from app_mcp.mcp_http_client import ensure_mcp_session_healthy

    session = await ensure_mcp_session_healthy()
    page = await session.list_tools()
    return {t.name for t in page.tools}


def mcp_tool_names() -> set[str] | None:
    """拉取当前 MCP 已注册工具名；连接失败时返回 None。"""
    if not mcp_port_open():
        return None
    try:
        from core.async_runner import run_async

        return run_async(_fetch_mcp_tool_names_async(), timeout=15)
    except Exception as e:
        logger.warning("无法读取 MCP 工具列表: %s", e)
        return None


_mcp_status_cache: tuple[float, str] | None = None
_MCP_STATUS_CACHE_SEC = 45.0


def mcp_wechat_tools_status(*, force: bool = False) -> str:
    """返回微信工具探测结果：ok / missing / unknown。"""
    global _mcp_status_cache
    import time

    now = time.time()
    if (
        not force
        and _mcp_status_cache is not None
        and now - _mcp_status_cache[0] < _MCP_STATUS_CACHE_SEC
    ):
        return _mcp_status_cache[1]

    names = mcp_tool_names()
    if names is None:
        status = "unknown"
    elif REQUIRED_WECHAT_MCP_TOOLS.issubset(names):
        status = "ok"
    else:
        status = "missing"
    _mcp_status_cache = (now, status)
    return status


def mcp_has_current_wechat_tools() -> bool:
    status = mcp_wechat_tools_status()
    if status == "ok":
        return True
    if status == "unknown":
        # 读取失败时不误判为旧进程，避免反复重启 MCP
        return mcp_port_open()
    return False

def restart_mcp_server() -> None:
    """终止占用 MCP 端口的旧进程（含外部手动启动的 mcp_server）。"""
    global mcp_process
    from app_mcp.mcp_http_client import get_mcp_manager

    get_mcp_manager().drop_session_sync()
    stop_mcp_subprocess()
    kill_mcp_port_processes()
    mcp_process = None
    time.sleep(0.5)


def ensure_mcp_server_started(wait_sec: float = 25.0) -> bool:
    """本地未跑 mcp_server 时由 app 拉起子进程并等待端口就绪。"""
    global mcp_process
    if mcp_port_open():
        from app_mcp.mcp_http_client import get_mcp_manager

        if get_mcp_manager().has_cached_session():
            return True
        status = mcp_wechat_tools_status()
        if status in ("ok", "unknown"):
            return True
        logger.warning(
            "MCP 端口 %s 已占用，但未发现微信工具 %s，正在重启 MCP…",
            MCP_PORT,
            sorted(REQUIRED_WECHAT_MCP_TOOLS),
        )
        restart_mcp_server()
    if not tracked_mcp_running() and not mcp_port_pids():
        mcp_process = subprocess.Popen(
            [sys.executable, str(BASE_DIR / "mcp_server.py")],
            cwd=str(BASE_DIR),
        )
    deadline = time.time() + wait_sec
    while time.time() < deadline:
        if not mcp_port_open():
            if mcp_process is not None and mcp_process.poll() is not None:
                return False
            time.sleep(0.35)
            continue
        status = mcp_wechat_tools_status()
        if status in ("ok", "unknown"):
            time.sleep(0.6)
            return True
        if mcp_process is not None and mcp_process.poll() is not None:
            return False
        time.sleep(0.35)
    return mcp_port_open()


async def ensure_mcp_server_started_async(wait_sec: float = 25.0) -> bool:
    """异步上下文安全版：避免 sync sleep / 线程阻塞卡住 event loop。"""
    return await asyncio.get_running_loop().run_in_executor(
        None, lambda: ensure_mcp_server_started(wait_sec)
    )


async def recover_mcp_server_async() -> bool:
    """重启 MCP 并等待就绪（供 Agent 连接异常后重试）。"""
    global _mcp_status_cache
    _mcp_status_cache = None
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, restart_mcp_server)
    ok = await ensure_mcp_server_started_async()
    if ok:
        await asyncio.sleep(0.6)
    return ok


def start_mcp_subprocess():
    """启动 MCP 子进程（不等待就绪）。"""
    global mcp_process
    return subprocess.Popen(
        [sys.executable, str(BASE_DIR / "mcp_server.py")],
        cwd=str(BASE_DIR),
    )


def stop_mcp_subprocess() -> None:
    """终止已跟踪的 MCP 子进程。"""
    global mcp_process
    if mcp_process:
        mcp_process.terminate()
        mcp_process = None
