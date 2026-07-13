"""FastAPI / uvicorn 主事件循环上的协程调度与可取消运行跟踪。

AsyncPostgresSaver / AsyncConnectionPool 内的 asyncio.Lock 会绑定创建时的 loop，
因此 Checkpointer 与 Agent 必须在同一 loop（uvicorn 主循环）上初始化与执行。
"""
from __future__ import annotations

import asyncio
import sys
import threading
from collections.abc import Coroutine
from typing import Any, TypeVar

T = TypeVar("T")


def _restore_asyncio_create_task() -> None:
    """恢复标准库 asyncio.create_task（支持 context=）。

    PyCharm PyDev Console 会替换 asyncio.create_task 为仅 (coro, name=) 的 shim，
    导致 LangGraph 1.1+ 调用 create_task(..., context=...) 失败并降级为无 MCP 模式。
    """
    import asyncio.tasks as asyncio_tasks

    real = asyncio_tasks.create_task
    if asyncio.create_task is real:
        return
    asyncio.create_task = real


_restore_asyncio_create_task()

_main_loop: asyncio.AbstractEventLoop | None = None
_runs_lock = threading.Lock()
_active_runs: dict[str, asyncio.Future[Any]] = {}


def _ensure_windows_selector_policy() -> None:
    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def set_main_loop(loop: asyncio.AbstractEventLoop) -> None:
    """在 FastAPI lifespan 中绑定 uvicorn 主事件循环。"""
    global _main_loop
    _main_loop = loop


def get_main_loop() -> asyncio.AbstractEventLoop | None:
    return _main_loop


def _register_run(run_key: str, future: asyncio.Future[Any]) -> None:
    with _runs_lock:
        prev = _active_runs.get(run_key)
        if prev is not None and not prev.done():
            prev.cancel()
        _active_runs[run_key] = future


def _unregister_run(run_key: str, future: asyncio.Future[Any]) -> None:
    with _runs_lock:
        if _active_runs.get(run_key) is future:
            _active_runs.pop(run_key, None)


def register_run(run_key: str, future: asyncio.Future[Any]) -> None:
    _register_run(run_key, future)


def unregister_run(run_key: str, future: asyncio.Future[Any]) -> None:
    _unregister_run(run_key, future)


def cancel_active_run(run_key: str) -> bool:
    """取消指定 session 正在执行的 Agent 协程（供用户主动打断）。"""
    with _runs_lock:
        future = _active_runs.get(run_key)
    if future is None or future.done():
        return False
    return bool(future.cancel())


def schedule_async(coro: Coroutine[Any, Any, Any]) -> None:
    """在主 loop 上调度协程（不阻塞调用方）。"""
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None

    if running is not None:
        running.create_task(coro)
        return

    loop = _main_loop
    if loop is None or not loop.is_running():
        raise RuntimeError("主事件循环尚未就绪，无法 schedule_async")
    asyncio.run_coroutine_threadsafe(coro, loop)


async def run_cancellable(
    coro: Coroutine[Any, Any, T],
    *,
    run_key: str | None = None,
) -> T:
    """在当前 loop 上执行协程，并按 run_key 登记以便 /chat/cancel 取消。"""
    task = asyncio.ensure_future(coro)
    if run_key:
        _register_run(run_key, task)
    try:
        return await task
    finally:
        if run_key:
            _unregister_run(run_key, task)


def run_async(
    coro: Coroutine[Any, Any, T],
    *,
    timeout: float | None = None,
    run_key: str | None = None,
) -> T:
    """从同步线程阻塞等待协程结果（不可在主事件循环线程内调用）。"""
    try:
        running = asyncio.get_running_loop()
    except RuntimeError:
        running = None

    loop = _main_loop
    if loop is None or not loop.is_running():
        raise RuntimeError("主事件循环尚未就绪，无法 run_async")

    if running is not None and running is loop:
        raise RuntimeError(
            "run_async() 不能在主事件循环内调用，请改用 await / run_cancellable / schedule_async"
        )

    future = asyncio.run_coroutine_threadsafe(coro, loop)
    if run_key:
        _register_run(run_key, future)
    try:
        if timeout is None:
            return future.result()
        return future.result(timeout=timeout)
    finally:
        if run_key:
            _unregister_run(run_key, future)


async def setup_async_services() -> None:
    """应用启动：在主 loop 上初始化 Checkpointer。"""
    from agent.agent_checkpointer import init_checkpointer

    set_main_loop(asyncio.get_running_loop())
    await init_checkpointer()


async def shutdown_async_services() -> None:
    """应用关闭：释放 Checkpointer 与 MCP 客户端。"""
    global _main_loop
    try:
        from agent.agent_checkpointer import shutdown_checkpointer
        from app_mcp.mcp_http_client import get_mcp_manager

        await shutdown_checkpointer()
        await get_mcp_manager().close()
    except Exception:
        pass
    _main_loop = None
