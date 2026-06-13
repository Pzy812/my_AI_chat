"""MCP Streamable HTTP 客户端：本地连接须绕过系统代理。"""
from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from config.app_config import MCP_URL

# Windows 常配置系统代理；httpx 默认 trust_env=True 会把 localhost 也走代理 → ConnectError
_MCP_HTTP_TIMEOUT = httpx.Timeout(30.0, read=300.0)


def create_mcp_httpx_client() -> httpx.AsyncClient:
    return httpx.AsyncClient(
        trust_env=False,
        follow_redirects=True,
        timeout=_MCP_HTTP_TIMEOUT,
    )


@asynccontextmanager
async def open_mcp_session() -> AsyncIterator[ClientSession]:
    """连接本机 MCP 并 yield 已 initialize 的 ClientSession。"""
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
async def open_mcp_transport() -> AsyncIterator[tuple]:
    """yield streamable_http_client 三元组 (read, write, get_session_id)。"""
    async with create_mcp_httpx_client() as http_client:
        async with streamable_http_client(MCP_URL, http_client=http_client) as transport:
            yield transport
