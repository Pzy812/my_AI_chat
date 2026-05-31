"""智谱 Chat 模型：可配置 httpx 读超时（langchain 默认仅 60s，长文档易 ReadTimeout）。"""
from __future__ import annotations

import os
from typing import Any, List, Optional

import httpx
from langchain_community.chat_models import ChatZhipuAI as _BaseChatZhipuAI
from langchain_core.callbacks import AsyncCallbackManagerForLLMRun, CallbackManagerForLLMRun
from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatResult
from pydantic import Field

from api_throttle import call_with_retry, call_with_retry_async

LLM_REQUEST_TIMEOUT = float(os.getenv("LLM_REQUEST_TIMEOUT", "180"))


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
            return super()._generate(
                messages, stop=stop, run_manager=run_manager, stream=stream, **kwargs
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
        with httpx.Client(headers=headers, timeout=timeout) as client:
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
            return await super()._agenerate(
                messages, stop=stop, run_manager=run_manager, stream=stream, **kwargs
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
        async with httpx.AsyncClient(headers=headers, timeout=timeout) as client:
            async def _post() -> httpx.Response:
                response = await client.post(self.zhipuai_api_base, json=payload)  # type: ignore[arg-type]
                response.raise_for_status()
                return response

            response = await call_with_retry_async(_post, label="zhipu-chat")
        return self._create_chat_result(response.json())


def make_chat_llm(**kwargs: Any) -> ChatZhipuAI:
    defaults = {"model": "glm-4", "temperature": 0.0}
    defaults.update(kwargs)
    return ChatZhipuAI(**defaults)
