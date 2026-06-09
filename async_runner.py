"""在 Flask 同步请求中统一使用单一后台 event loop 执行协程。

AsyncPostgresSaver / AsyncConnectionPool 内的 asyncio.Lock 会绑定创建时的 loop。
若启动时用 asyncio.run() 初始化、请求里再次 asyncio.run()，会触发：
「Lock is bound to a different event loop」
"""
from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")

_loop: asyncio.AbstractEventLoop | None = None
_thread: threading.Thread | None = None
_ready = threading.Event()
_lock = threading.Lock()


def _ensure_windows_selector_policy() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def _loop_thread_main(loop: asyncio.AbstractEventLoop) -> None:
    asyncio.set_event_loop(loop)
    _ready.set()
    loop.run_forever()


def ensure_loop() -> asyncio.AbstractEventLoop:
    """获取（或启动）全局后台 event loop。"""
    global _loop, _thread
    with _lock:
        if _loop is not None and _loop.is_running():
            return _loop
        _ensure_windows_selector_policy()
        _ready.clear()
        loop = asyncio.new_event_loop()
        _thread = threading.Thread(
            target=_loop_thread_main,
            args=(loop,),
            name="ai-chat-async-loop",
            daemon=True,
        )
        _thread.start()
        if not _ready.wait(timeout=30):
            raise RuntimeError("后台 asyncio 事件循环启动超时")
        _loop = loop
        return _loop


def run_async(coro: Coroutine[Any, Any, T], *, timeout: float | None = None) -> T:
    """在后台 loop 上运行协程并阻塞等待结果（供 Flask 路由调用）。"""
    loop = ensure_loop()
    future = asyncio.run_coroutine_threadsafe(coro, loop)
    if timeout is None:
        return future.result()
    return future.result(timeout=timeout)


def setup_async_services() -> None:
    """应用启动：在同一后台 loop 上初始化 Checkpointer。"""
    from agent_checkpointer import init_checkpointer

    run_async(init_checkpointer())


def shutdown_async_services() -> None:
    """可选：关闭后台 loop（进程退出时 daemon 线程会自动结束）。"""
    global _loop, _thread
    loop = _loop
    if loop is None or not loop.is_running():
        return
    try:
        from agent_checkpointer import shutdown_checkpointer

        run_async(shutdown_checkpointer(), timeout=15)
    except Exception:
        pass
    loop.call_soon_threadsafe(loop.stop)
    if _thread is not None:
        _thread.join(timeout=5)
    _loop = None
    _thread = None
