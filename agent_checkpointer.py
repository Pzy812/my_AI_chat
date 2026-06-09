"""LangGraph AsyncPostgresSaver：Agent 运行 checkpoint（与业务 chat 表分离）。"""
from __future__ import annotations
import asyncio
import logging
from typing import Any

from app_config import AGENT_CHECKPOINT_ENABLED, POSTGRES_URI

logger = logging.getLogger("ai_chat.checkpointer")

_checkpointer: Any | None = None
_pool: Any | None = None
_ready = False


def enabled() -> bool:
    return bool(POSTGRES_URI and AGENT_CHECKPOINT_ENABLED)


def get_checkpointer() -> Any | None:
    return _checkpointer if _ready else None


async def init_checkpointer() -> bool:
    """启动时创建连接池并 setup checkpoint 表（须在 async_runner 后台 loop 内调用）。"""
    global _checkpointer, _pool, _ready
    if not enabled():
        logger.info("Agent Checkpointer 未启用（需 POSTGRES_URI + AGENT_CHECKPOINT_ENABLED=1）")
        return False
    current_loop = asyncio.get_running_loop()
    if _ready and _checkpointer is not None:
        bound = getattr(_checkpointer, "loop", None)
        if bound is not None and bound is not current_loop:
            logger.warning("Checkpointer 绑定在旧 event loop，正在重新初始化…")
            await shutdown_checkpointer()
        else:
            return True
    try:
        from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
        from psycopg.rows import dict_row
        from psycopg_pool import AsyncConnectionPool
    except ImportError as e:
        logger.warning(
            "无法加载 AsyncPostgresSaver，请安装: pip install langgraph-checkpoint-postgres psycopg[binary] psycopg-pool — %s",
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
        logger.info("LangGraph AsyncPostgresSaver 已就绪")
        return True
    except Exception as e:
        logger.warning("Agent Checkpointer 初始化失败，Agent 将无 checkpoint：%s", e)
        _checkpointer = None
        _pool = None
        _ready = False
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
    global _checkpointer, _pool, _ready
    if _pool is not None:
        try:
            await _pool.close()
        except Exception:
            pass
    _checkpointer = None
    _pool = None
    _ready = False
