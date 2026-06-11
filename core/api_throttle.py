"""智谱 API 限速与 429 重试（遵循 Retry-After，见 RFC 6585 / HTTP 429）。"""
from __future__ import annotations

import asyncio
import logging
import os
import random
import threading
import time
from typing import Awaitable, Callable, TypeVar

import httpx

logger = logging.getLogger("api_throttle")

T = TypeVar("T")

# 两次请求之间的最小间隔（秒）；0 表示不限速（仅 429 时退避重试）
API_REQUEST_INTERVAL_SEC = float(os.getenv("API_REQUEST_INTERVAL_SEC", "0"))
# 429 最大重试次数
API_RETRY_MAX = int(os.getenv("API_RETRY_MAX", "5"))
# 首次退避基数（秒）；服务端未返回 Retry-After 时使用
API_RETRY_BASE_SEC = float(os.getenv("API_RETRY_BASE_SEC", "2.0"))
# 单次等待上限（秒）
API_RETRY_MAX_WAIT_SEC = float(os.getenv("API_RETRY_MAX_WAIT_SEC", "120"))

_lock = threading.Lock()
_last_call_mono = 0.0
_async_lock: asyncio.Lock | None = None


def _get_async_lock() -> asyncio.Lock:
    global _async_lock
    if _async_lock is None:
        _async_lock = asyncio.Lock()
    return _async_lock


def throttle_wait() -> None:
    """可选节流：API_REQUEST_INTERVAL_SEC>0 时生效。"""
    if API_REQUEST_INTERVAL_SEC <= 0:
        return
    global _last_call_mono
    with _lock:
        now = time.monotonic()
        gap = API_REQUEST_INTERVAL_SEC - (now - _last_call_mono)
        if gap > 0:
            time.sleep(gap)
        _last_call_mono = time.monotonic()


async def throttle_wait_async() -> None:
    if API_REQUEST_INTERVAL_SEC <= 0:
        return
    async with _get_async_lock():
        global _last_call_mono
        now = time.monotonic()
        gap = API_REQUEST_INTERVAL_SEC - (now - _last_call_mono)
        if gap > 0:
            await asyncio.sleep(gap)
        _last_call_mono = time.monotonic()


def _parse_retry_after_header(value: str) -> float | None:
    value = (value or "").strip()
    if not value:
        return None
    try:
        return max(float(value), 0.0)
    except ValueError:
        return None


def retry_wait_seconds(exc: BaseException, attempt: int) -> float | None:
    """若可重试则返回应等待秒数，否则 None。"""
    status = None
    retry_after: float | None = None

    if isinstance(exc, httpx.HTTPStatusError):
        status = exc.response.status_code
        retry_after = _parse_retry_after_header(
            exc.response.headers.get("Retry-After", "")
        )
    else:
        resp = getattr(exc, "response", None)
        if resp is not None:
            status = getattr(resp, "status_code", None)
            headers = getattr(resp, "headers", None)
            if headers:
                retry_after = _parse_retry_after_header(
                    headers.get("Retry-After", "")
                )

    msg = str(exc).lower()
    is_429 = status == 429 or "429" in msg or "too many requests" in msg
    is_rate = is_429 or "rate limit" in msg or "频率" in msg or "限流" in msg

    if not is_rate:
        return None

    base = retry_after if retry_after is not None else API_RETRY_BASE_SEC
    # 指数退避 + 抖动，避免并发重试雪崩
    wait = min(
        base * (2**attempt) + random.uniform(0.0, 0.8),
        API_RETRY_MAX_WAIT_SEC,
    )
    if API_REQUEST_INTERVAL_SEC > 0:
        return max(wait, API_REQUEST_INTERVAL_SEC)
    return wait


def is_rate_limit_error(exc: BaseException) -> bool:
    return retry_wait_seconds(exc, 0) is not None


def call_with_retry(
    fn: Callable[[], T],
    *,
    label: str = "zhipu-api",
    throttle: bool = False,
) -> T:
    last_exc: BaseException | None = None
    for attempt in range(API_RETRY_MAX + 1):
        if throttle:
            throttle_wait()
        try:
            return fn()
        except Exception as e:
            last_exc = e
            wait = retry_wait_seconds(e, attempt)
            if wait is None or attempt >= API_RETRY_MAX:
                raise
            logger.warning(
                "%s 触发限速 HTTP 429，第 %s/%s 次重试，等待 %.1fs：%s",
                label,
                attempt + 1,
                API_RETRY_MAX,
                wait,
                e,
            )
            time.sleep(wait)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{label} 重试失败")


async def call_with_retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    label: str = "zhipu-api",
    throttle: bool = False,
) -> T:
    last_exc: BaseException | None = None
    for attempt in range(API_RETRY_MAX + 1):
        if throttle:
            await throttle_wait_async()
        try:
            return await fn()
        except Exception as e:
            last_exc = e
            wait = retry_wait_seconds(e, attempt)
            if wait is None or attempt >= API_RETRY_MAX:
                raise
            logger.warning(
                "%s 触发限速 HTTP 429，第 %s/%s 次重试，等待 %.1fs：%s",
                label,
                attempt + 1,
                API_RETRY_MAX,
                wait,
                e,
            )
            await asyncio.sleep(wait)
    if last_exc is not None:
        raise last_exc
    raise RuntimeError(f"{label} 重试失败")
