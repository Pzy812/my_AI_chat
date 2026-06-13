"""LangGraph Checkpointer：优先 Postgres，失败时回退 MemorySaver（保证 HITL 可用）。"""
from __future__ import annotations
import asyncio
import logging
from typing import Any, Literal

from config.app_config import AGENT_CHECKPOINT_ENABLED, HITL_ENABLED, POSTGRES_URI

logger = logging.getLogger("ai_chat.checkpointer")

CheckpointerKind = Literal["postgres", "memory", "none"]

_checkpointer: Any | None = None
_pool: Any | None = None
_ready = False
_kind: CheckpointerKind = "none"


def postgres_enabled() -> bool:
    return bool(POSTGRES_URI and AGENT_CHECKPOINT_ENABLED)


def enabled() -> bool:
    return _ready


def checkpointer_kind() -> CheckpointerKind:
    return _kind if _ready else "none"


def get_checkpointer() -> Any | None:
    return _checkpointer if _ready else None


async def _init_postgres_checkpointer(current_loop: asyncio.AbstractEventLoop) -> bool:
    global _checkpointer, _pool, _ready, _kind
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool
    except ImportError as e:
        logger.warning(
            "无法加载 AsyncPostgresSaver，请安装: pip install -r requirements-postgres.txt — %s",
            e,
        )
        return False

    try:
        _pool = AsyncConnectionPool(
            conninfo=POSTGRES_URI,
            min_size=1,
            max_size=10,
            kwargs={
                "autocommit": True,
                "prepare_threshold": 0,
                "row_factory": dict_row,
            },
        )
        await _pool.open()
        _checkpointer = AsyncPostgresSaver(conn=_pool)
        await _checkpointer.setup()
        _ready = True
        _kind = "postgres"
        logger.info("LangGraph AsyncPostgresSaver 已就绪")
        return True
    except Exception as e:
        logger.warning("Postgres Checkpointer 初始化失败：%s", e)
        _checkpointer = None
        if _pool is not None:
            try:
                await _pool.close()
            except Exception:
                pass
        _pool = None
        _ready = False
        _kind = "none"
        return False


async def _init_memory_checkpointer() -> bool:
    global _checkpointer, _ready, _kind
    try:
        from langgraph.checkpoint.memory import MemorySaver
    except ImportError as e:
        logger.warning("无法加载 MemorySaver：%s", e)
        return False
    _checkpointer = MemorySaver()
    _ready = True
    _kind = "memory"
    logger.warning(
        "已回退 MemorySaver（HITL 可用；进程重启后 checkpoint 丢失）。"
        "持久化请配置 POSTGRES_URI 并安装 requirements-postgres.txt"
    )
    return True


async def init_checkpointer() -> bool:
    """启动时创建 checkpoint（须在 async_runner 后台 loop 内调用）。"""
    global _checkpointer, _pool, _ready, _kind
    want_checkpoint = HITL_ENABLED or AGENT_CHECKPOINT_ENABLED
    if not want_checkpoint:
        logger.info("Agent Checkpointer 未启用（HITL_ENABLED 与 AGENT_CHECKPOINT_ENABLED 均为关闭）")
        return False

    current_loop = asyncio.get_running_loop()
    if _ready and _checkpointer is not None:
        bound = getattr(_checkpointer, "loop", None)
        if bound is not None and bound is not current_loop:
            logger.warning("Checkpointer 绑定在旧 event loop，正在重新初始化…")
            await shutdown_checkpointer()
        else:
            return True

    if postgres_enabled():
        if await _init_postgres_checkpointer(current_loop):
            return True

    if HITL_ENABLED:
        return await _init_memory_checkpointer()

    logger.info(
        "Agent Checkpointer 未就绪（Postgres 不可用且 HITL 已关闭，无法使用 Memory 回退）"
    )
    return False


async def reset_agent_thread(session_id: str) -> None:
    """每轮对话前清空 thread checkpoint，避免与 chat_store 最近 N 条重复叠加。"""
    cp = get_checkpointer()
    if cp is None:
        return
    sid = (session_id or "default").strip() or "default"
    try:
        await cp.adelete_thread({"configurable": {"thread_id": sid}})
    except Exception as e:
        logger.debug("adelete_thread(%s) 跳过: %s", sid, e)


async def shutdown_checkpointer() -> None:
    global _checkpointer, _pool, _ready, _kind
    if _pool is not None:
        try:
            await _pool.close()
        except Exception:
            pass
    _checkpointer = None
    _pool = None
    _ready = False
    _kind = "none"
