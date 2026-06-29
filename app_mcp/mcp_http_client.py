"""MCP Streamable HTTP 客户端：本地连接须绕过系统代理。"""
from __future__ import annotations

import asyncio
import logging
import threading
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import AsyncExitStack, asynccontextmanager
from typing import TypeVar

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from config.app_config import MCP_URL

logger = logging.getLogger("ai_chat.mcp")

# Windows 常配置系统代理；httpx 默认 trust_env=True 会把 localhost 也走代理 → ConnectError
_MCP_HTTP_TIMEOUT = httpx.Timeout(30.0, read=300.0)

T = TypeVar("T")

_MCP_HTTP_ERROR_NAMES = frozenset(
    {
        "TransportError",
        "ReadError",
        "WriteError",
        "ConnectError",
        "ConnectTimeout",
        "ReadTimeout",
        "RemoteProtocolError",
        "StreamableHTTPError",
        "ClosedResourceError",
        "McpError",
    }
)


def _looks_like_zhipu_stream_error(exc: BaseException) -> bool:
    """智谱流式在 429/5xx 时也会抛 SSEError（Content-Type 非 event-stream），勿当成 MCP。"""
    name = type(exc).__name__
    if name != "SSEError" and "text/event-stream" not in str(exc).lower():
        return False
    tb = getattr(exc, "__traceback__", None)
    while tb is not None:
        path = (tb.tb_frame.f_code.co_filename or "").lower()
        if "zhipuai" in path or "open.bigmodel.cn" in path:
            return True
        tb = tb.tb_next
    msg = str(exc).lower()
    return "open.bigmodel.cn" in msg or "zhipu" in msg


def _is_zhipu_or_llm_http_error(exc: BaseException) -> bool:
    if _looks_like_zhipu_stream_error(exc):
        return True
    if isinstance(exc, httpx.HTTPStatusError):
        url = str(getattr(getattr(exc, "request", None), "url", "") or "").lower()
        if "open.bigmodel.cn" in url or "bigmodel.cn" in url:
            return True
    msg = str(exc).lower()
    return "open.bigmodel.cn" in msg or "bigmodel.cn" in msg


def is_mcp_transport_error(exc: BaseException) -> bool:
    """判断是否为 MCP Streamable HTTP / SSE 连接类异常（可尝试重连后重试）。"""
    if isinstance(exc, asyncio.CancelledError):
        return False
    if _is_zhipu_or_llm_http_error(exc):
        return False
    if isinstance(exc, BaseExceptionGroup):
        if not exc.exceptions:
            return False
        if any(_is_zhipu_or_llm_http_error(x) for x in exc.exceptions):
            return False
        return all(is_mcp_transport_error(x) for x in exc.exceptions)
    if isinstance(exc, httpx.HTTPError):
        url = str(getattr(getattr(exc, "request", None), "url", "") or "").lower()
        if "open.bigmodel.cn" in url or "bigmodel.cn" in url:
            return False
        if "localhost" in url or ":8090" in url or "/mcp" in url:
            return True
        return False
    if type(exc).__name__ in _MCP_HTTP_ERROR_NAMES:
        return True
    msg = str(exc).lower()
    if "localhost" in msg or ":8090" in msg or "/mcp" in msg:
        return (
            "text/event-stream" in msg
            or "unexpected content type" in msg
            or "session terminated" in msg
            or "connection attempts failed" in msg
        )
    return False


def create_mcp_httpx_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        trust_env=False,
        follow_redirects=True,
        timeout=_MCP_HTTP_TIMEOUT,
    )


class McpSessionManager:
    """进程内复用单条 MCP Streamable HTTP 连接，避免频繁 DELETE/重建导致 SSE 会话失效。"""

    def __init__(self) -> None:
        self._lock = asyncio.Lock()
        self._sync_lock = threading.Lock()
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    def drop_session_sync(self) -> None:
        """丢弃缓存会话（MCP 子进程重启后须调用，避免继续使用失效 SSE）。"""
        with self._sync_lock:
            self._stack = None
            self._session = None

    def has_cached_session(self) -> bool:
        with self._sync_lock:
            return self._session is not None

    async def _close_stack_best_effort(self, stack: AsyncExitStack) -> None:
        try:
            await asyncio.wait_for(stack.aclose(), timeout=3.0)
        except Exception as e:
            logger.debug("orphan MCP stack close ignored: %s", e)

    async def get_session(self) -> ClientSession:
        async with self._lock:
            if self._session is not None:
                return self._session
            return await self._connect_locked()

    async def reconnect(self) -> ClientSession:
        async with self._lock:
            await self._drop_locked()
            return await self._connect_locked()

    async def close(self) -> None:
        async with self._lock:
            await self._drop_locked()

    async def _drop_locked(self) -> None:
        """丢弃引用；不在其它 Task 里 aclose streamable_http（会触发 anyio cancel scope 错误）。"""
        with self._sync_lock:
            self._stack = None
            self._session = None

    async def _connect_locked(self) -> ClientSession:
        stack = AsyncExitStack()
        http_client = create_mcp_httpx_client()
        await stack.enter_async_context(http_client)
        read_stream, write_stream, _ = await stack.enter_async_context(
            streamable_http_client(
                MCP_URL,
                http_client=http_client,
                terminate_on_close=False,
            )
        )
        session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
        await session.initialize()
        self._stack = stack
        self._session = session
        logger.info("MCP 长连接已建立（复用模式，HITL 恢复不再 DELETE 会话）")
        return session


_mcp_manager = McpSessionManager()


def get_mcp_manager() -> McpSessionManager:
    return _mcp_manager


async def get_mcp_session() -> ClientSession:
    return await _mcp_manager.get_session()


async def ensure_mcp_session_healthy() -> ClientSession:
    """校验长连接可用；失败则重建（Agent 每次运行前调用）。"""
    try:
        session = await get_mcp_session()
        await asyncio.wait_for(session.list_tools(), timeout=12.0)
        return session
    except Exception as e:
        if not is_mcp_transport_error(e):
            raise
        logger.warning("MCP 会话探测失败，重建连接: %s", e)
        return await reconnect_mcp_session()


async def reconnect_mcp_session() -> ClientSession:
    logger.warning("MCP 连接异常，正在重连客户端…")
    return await _mcp_manager.reconnect()


async def recover_mcp_server_async() -> bool:
    """重启 MCP 子进程并重建客户端连接。"""
    import app_mcp.mcp_lifecycle as mcp_lifecycle

    _mcp_manager.drop_session_sync()
    ok = await mcp_lifecycle.recover_mcp_server_async()
    if ok:
        try:
            await _mcp_manager.get_session()
        except Exception as e:
            logger.warning("MCP 服务重启后客户端连接失败: %s", e)
            return False
    return ok


async def run_with_mcp_retry(
    fn: Callable[[ClientSession], Awaitable[T]],
    *,
    max_attempts: int = 3,
) -> T:
    """在 MCP 会话上执行协程；连接异常时用新短连接重试，必要时重启 MCP 服务。"""
    last_err: BaseException | None = None
    for attempt in range(max_attempts):
        try:
            async with open_ephemeral_mcp_session() as session:
                return await fn(session)
        except BaseException as e:
            last_err = e
            if attempt + 1 >= max_attempts or not is_mcp_transport_error(e):
                raise
            logger.warning(
                "MCP 调用失败，准备重试 (%s/%s): %s",
                attempt + 1,
                max_attempts,
                e,
            )
            if attempt >= 1 and not await recover_mcp_server_async():
                raise RuntimeError("MCP 重启失败，请手动运行 python mcp_server.py") from e
    assert last_err is not None
    raise last_err


@asynccontextmanager
async def open_ephemeral_mcp_session() -> AsyncIterator[ClientSession]:
    """独立短连接（探测/兼容用），退出时会 DELETE MCP 会话。"""
    async with create_mcp_httpx_client() as http_client:
        async with streamable_http_client(MCP_URL, http_client=http_client) as (
            read_stream,
            write_stream,
            _,
        ):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                yield session


@asynccontextmanager
async def open_mcp_session() -> AsyncIterator[ClientSession]:
    """Agent 主路径：复用进程内长连接。"""
    yield await get_mcp_session()


@asynccontextmanager
async def open_mcp_transport() -> AsyncIterator[tuple]:
    """兼容旧代码：短连接模式（会 DELETE 会话）。Agent 路径请改用 get_mcp_session()。"""
    async with create_mcp_httpx_client() as http_client:
        async with streamable_http_client(MCP_URL, http_client=http_client) as transport:
            yield transport
