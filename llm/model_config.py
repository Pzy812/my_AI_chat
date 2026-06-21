"""LLM 配置：智谱预设、OpenAI 兼容模型与工厂方法。"""
from __future__ import annotations

import os
from typing import Any

from config.env_config import get_zhipuai_api_key

ZHIPU_MODEL_PRESETS: list[dict[str, str]] = [
    {"id": "glm-4.7", "label": "GLM-4.7"},
    {"id": "glm-4-plus", "label": "GLM-4 Plus"},
    {"id": "glm-4", "label": "GLM-4"},
    {"id": "glm-4-flash", "label": "GLM-4 Flash"},
    {"id": "glm-4-air", "label": "GLM-4 Air"},
    {"id": "glm-4-long", "label": "GLM-4 Long"},
    {"id": "glm-4v", "label": "GLM-4V（视觉）"},
    {"id": "glm-4v-plus", "label": "GLM-4V Plus"},
]

DEFAULT_ZHIPU_MODEL = os.getenv("ZHIPU_MODEL", "glm-4").strip() or "glm-4"


def mask_api_key(key: str) -> str:
    k = (key or "").strip()
    if not k:
        return ""
    if len(k) <= 8:
        return "*" * len(k)
    return k[:4] + "…" + k[-4:]


def default_llm_config() -> dict[str, str]:
    return {
        "provider": "zhipu",
        "model": DEFAULT_ZHIPU_MODEL,
        "api_key": "",
        "base_url": "",
    }


def normalize_llm_config(raw: dict[str, Any] | None) -> dict[str, str]:
    base = default_llm_config()
    if not raw or not isinstance(raw, dict):
        return base
    provider = str(raw.get("provider") or "zhipu").strip().lower()
    if provider not in ("zhipu", "custom"):
        provider = "zhipu"
    model = str(raw.get("model") or base["model"]).strip() or base["model"]
    api_key = str(raw.get("api_key") or "").strip()
    base_url = str(raw.get("base_url") or "").strip()
    return {
        "provider": provider,
        "model": model,
        "api_key": api_key,
        "base_url": base_url,
    }


def server_llm_defaults() -> dict[str, Any]:
    """供前端展示的服务端默认（不返回完整密钥）。"""
    try:
        env_key = get_zhipuai_api_key()
        key_configured = True
        masked = mask_api_key(env_key)
    except RuntimeError:
        key_configured = False
        masked = ""
    return {
        "provider": "zhipu",
        "model": DEFAULT_ZHIPU_MODEL,
        "api_key_configured": key_configured,
        "api_key_masked": masked,
        "base_url": "",
        "zhipu_presets": ZHIPU_MODEL_PRESETS,
    }


def make_llm_from_config(config: dict[str, Any] | None = None, **kwargs: Any):
    """按前端/请求配置创建 LangChain Chat 模型。"""
    cfg = normalize_llm_config(config)
    provider = cfg["provider"]
    model = cfg["model"]
    api_key = cfg["api_key"]
    base_url = cfg["base_url"]
    temperature = kwargs.pop("temperature", 0.0)

    if provider == "zhipu":
        from llm.llm_zhipu import make_chat_llm

        llm_kwargs: dict[str, Any] = {"model": model, "temperature": temperature}
        if api_key:
            llm_kwargs["api_key"] = api_key
        llm_kwargs.update(kwargs)
        return make_chat_llm(**llm_kwargs)

    from langchain_community.chat_models import ChatOpenAI

    resolved_key = api_key or os.getenv("OPENAI_API_KEY", "").strip()
    if not resolved_key:
        raise RuntimeError("未配置 API Key：请在设置中填写，或在 .env 中设置 OPENAI_API_KEY")
    if not model:
        raise RuntimeError("未指定模型名称")
    llm_kwargs = {
        "model": model,
        "temperature": temperature,
        "api_key": resolved_key,
    }
    if base_url:
        llm_kwargs["base_url"] = base_url
    llm_kwargs.update(kwargs)
    return ChatOpenAI(**llm_kwargs)
