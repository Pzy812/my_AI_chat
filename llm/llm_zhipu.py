"""智谱 Chat 模型：可配置 httpx 读超时（langchain 默认仅 60s，长文档易 ReadTimeout）。"""
from __future__ import annotations

import logging
import os
from collections.abc import AsyncIterator, Iterator
from typing import Any, List, Optional

import httpx
from langchain_community.chat_models import ChatZhipuAI as _BaseChatZhipuAI
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.messages import AIMessageChunk, BaseMessage
from langchain_core.outputs import ChatGenerationChunk, ChatResult
from pydantic import Field

from core.api_throttle import call_with_retry, call_with_retry_async

logger = logging.getLogger("ai_chat.llm")

LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "180"))


def _zhipu_stream_fallbackable(exc: BaseException) -> bool:
    """流式 SSE 失败（常见为 429 返回空 Content-Type）时允许降级为非流式。"""
    name = type(exc).__name__
    msg = str(exc).lower()
    if name == "SSEError" or "text/event-stream" in msg:
        return True
    if "429" in msg or "too many requests" in msg or "频率" in msg or "限流" in msg:
        return True
    return False


class ChatZhipuAI(_BaseChatZhipuAI):
    request_timeout: float = Field(default=LLM_REQUEST_TIMEOUT, ge=30.0)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            try:
                return super()._generate(
                    messages, stop=stop, run_manager=run_manager, stream=stream, **kwargs
                )
            except Exception as e:
                if not _zhipu_stream_fallbackable(e):
                    raise
                logger.warning("智谱流式失败，降级为非流式: %s", e)
                return self._generate(
                    messages, stop=stop, run_manager=run_manager, stream=False, **kwargs
                )
        if self.zhipuai_api_key is None:
            raise ValueError("Did not find zhipuai_api_key.")
        message_dicts, params = self._create_message_dicts(messages, stop)
        payload = {**params, **kwargs, "messages": message_dicts, "stream": False}
        from langchain_community.chat_models.zhipuai import (
            _get_jwt_token,
            _truncate_params,
        )

        _truncate_params(payload)
        headers = {
            "Authorization": _get_jwt_token(self.zhipuai_api_key),
            "Accept": "application/json",
        }
        timeout = httpx.Timeout(self.request_timeout)
        with httpx.Client(headers=headers, timeout=timeout, trust_env=False) as client:
            def _post() -> httpx.Response:
                response = client.post(self.zhipuai_api_base, json=payload)  # type: ignore[arg-type]
                response.raise_for_status()
                return response

            response = call_with_retry(_post, label="zhipu-chat")
        return self._create_chat_result(response.json())

    async def _agenerate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        stream: Optional[bool] = None,
        **kwargs: Any,
    ) -> ChatResult:
        should_stream = stream if stream is not None else self.streaming
        if should_stream:
            try:
                return await super()._agenerate(
                    messages, stop=stop, run_manager=run_manager, stream=stream, **kwargs
                )
            except Exception as e:
                if not _zhipu_stream_fallbackable(e):
                    raise
                logger.warning("智谱流式失败，降级为非流式: %s", e)
                return await self._agenerate(
                    messages, stop=stop, run_manager=run_manager, stream=False, **kwargs
                )
        if self.zhipuai_api_key is None:
            raise ValueError("Did not find zhipuai_api_key.")
        message_dicts, params = self._create_message_dicts(messages, stop)
        payload = {**params, **kwargs, "messages": message_dicts, "stream": False}
        from langchain_community.chat_models.zhipuai import (
            _get_jwt_token,
            _truncate_params,
        )

        _truncate_params(payload)
        headers = {
            "Authorization": _get_jwt_token(self.zhipuai_api_key),
            "Accept": "application/json",
        }
        timeout = httpx.Timeout(self.request_timeout)
        async with httpx.AsyncClient(headers=headers, timeout=timeout, trust_env=False) as client:
            async def _post() -> httpx.Response:
                response = await client.post(self.zhipuai_api_base, json=payload)  # type: ignore[arg-type]
                response.raise_for_status()
                return response

            response = await call_with_retry_async(_post, label="zhipu-chat")
        return self._create_chat_result(response.json())

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        try:
            yield from super()._stream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            )
        except Exception as e:
            if not _zhipu_stream_fallbackable(e):
                raise
            logger.warning("智谱流式失败，降级为非流式: %s", e)
            result = self._generate(
                messages, stop=stop, run_manager=run_manager, stream=False, **kwargs
            )
            text = result.generations[0].message.content
            if text:
                yield ChatGenerationChunk(message=AIMessageChunk(content=text))

    async def _astream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[AsyncCallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> AsyncIterator[ChatGenerationChunk]:
        try:
            async for chunk in super()._astream(
                messages, stop=stop, run_manager=run_manager, **kwargs
            ):
                yield chunk
        except Exception as e:
            if not _zhipu_stream_fallbackable(e):
                raise
            logger.warning("智谱流式失败，降级为非流式: %s", e)
            result = await self._agenerate(
                messages, stop=stop, run_manager=run_manager, stream=False, **kwargs
            )
            text = result.generations[0].message.content
            if text:
                yield ChatGenerationChunk(message=AIMessageChunk(content=text))


def make_chat_llm(**kwargs: Any) -> ChatZhipuAI:
    defaults = {"model": "glm-4", "temperature": 0.0}
    defaults.update(kwargs)
    return ChatZhipuAI(**defaults)


def make_summary_llm(**kwargs: Any) -> ChatZhipuAI:
    """会话历史摘要：默认 glm-4-flash，复用 ZHIPUAI_API_KEY。"""
    model = os.getenv("ZHIPU_SUMMARY_MODEL", "glm-4-flash").strip() or "glm-4-flash"
    timeout = float(os.getenv("ZHIPU_SUMMARY_TIMEOUT", "90"))
    defaults = {
        "model": model,
        "temperature": 0.0,
        "request_timeout": timeout,
    }
    defaults.update(kwargs)
    return ChatZhipuAI(**defaults)
