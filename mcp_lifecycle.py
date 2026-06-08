"""MCP 子进程与端口生命周期管理。"""
import os
import socket
import subprocess
import sys
import time

from app_config import BASE_DIR, MCP_HOST, MCP_PORT

mcp_process = None


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


def ensure_mcp_server_started(wait_sec: float = 25.0) -> bool:
    """本地未跑 mcp_server 时由 app 拉起子进程并等待端口就绪。"""
    global mcp_process
    if mcp_port_open():
        return True
    if not tracked_mcp_running() and not mcp_port_pids():
        mcp_process = subprocess.Popen(
            [sys.executable, str(BASE_DIR / "mcp_server.py")],
            cwd=str(BASE_DIR),
        )
    deadline = time.time() + wait_sec
    while time.time() < deadline:
        if mcp_port_open():
            time.sleep(0.6)
            return True
        if mcp_process is not None and mcp_process.poll() is not None:
            return False
        time.sleep(0.35)
    return False


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
