"""LangSmith 环境配置与 Agent run config 增强。"""
from __future__ import annotations

import os
from typing import Any

_CONFIGURED = False


def langsmith_api_key() -> str:
    return (
        os.getenv("LANGSMITH_API_KEY", "").strip()
        or os.getenv("LANGCHAIN_API_KEY", "").strip()
    )


def langsmith_project() -> str:
    return (
        os.getenv("LANGSMITH_PROJECT", "").strip()
        or os.getenv("LANGCHAIN_PROJECT", "").strip()
        or "my_agent"
    )


def langsmith_endpoint() -> str:
    return os.getenv("LANGSMITH_ENDPOINT", "https://api.smith.langchain.com").strip()


def langsmith_web_base() -> str:
    return os.getenv("LANGSMITH_WEB_URL", "https://smith.langchain.com").strip().rstrip("/")


def is_tracing_enabled() -> bool:
    if not langsmith_api_key():
        return False
    flag = os.getenv("LANGSMITH_TRACING", os.getenv("LANGCHAIN_TRACING_V2", "true"))
    return str(flag).strip().lower() not in ("0", "false", "no", "off")


def configure_langsmith() -> bool:
    """在 LangChain / LangGraph 调用前设置 tracing 环境变量。"""
    global _CONFIGURED
    if _CONFIGURED:
        return is_tracing_enabled()
    _CONFIGURED = True
    key = langsmith_api_key()
    if not key or not is_tracing_enabled():
        return False
    os.environ.setdefault("LANGSMITH_API_KEY", key)
    os.environ.setdefault("LANGCHAIN_API_KEY", key)
    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ.setdefault("LANGSMITH_TRACING", "true")
    project = langsmith_project()
    os.environ.setdefault("LANGSMITH_PROJECT", project)
    os.environ.setdefault("LANGCHAIN_PROJECT", project)
    endpoint = langsmith_endpoint()
    if endpoint:
        os.environ.setdefault("LANGSMITH_ENDPOINT", endpoint)
    return True


def public_status() -> dict[str, Any]:
    """供 /service/status 与前端展示（不含 API Key）。"""
    enabled = configure_langsmith()
    return {
        "enabled": enabled,
        "project": langsmith_project() if enabled else "",
        "web_url": langsmith_web_base() if enabled else "",
    }


def thread_metadata(session_id: str, *, turn: int | None = None) -> dict[str, str | int]:
    """LangSmith Threads 分组所需 metadata（thread_id + session_id）。"""
    sid = (session_id or "default").strip() or "default"
    meta: dict[str, str | int] = {
        "thread_id": sid,
        "session_id": sid,
        "app": "ai_chat_agent",
    }
    if turn is not None:
        meta["turn"] = turn
    return meta


def build_thread_url(thread_id: str) -> str:
    """LangSmith 项目 Threads 视图（按 thread_id 过滤）。"""
    project = langsmith_project()
    base = langsmith_web_base()
    tid = (thread_id or "default").strip() or "default"
    return f"{base}/o/default/projects/p/{project}?tab=1&searchModel=%7B%22searchFilter%22%3A%22eq%28metadata.thread_id%2C%20%22{tid}%22%29%22%7D"


def enrich_agent_config(
    config: dict[str, Any],
    *,
    session_id: str,
    rag_mode: str | None = None,
    harness_enabled: bool | None = None,
    stream: bool = False,
    resume_action: str | None = None,
    turn_index: int | None = None,
) -> dict[str, Any]:
    """为 LangGraph invoke / astream_events 注入 metadata 与 tags。"""
    if not is_tracing_enabled():
        return config
    out = dict(config)
    metadata = dict(out.get("metadata") or {})
    metadata.update(thread_metadata(session_id, turn=turn_index))
    if rag_mode:
        metadata["rag_mode"] = rag_mode
    if harness_enabled is not None:
        metadata["harness_enabled"] = harness_enabled
    if resume_action:
        metadata["hitl_action"] = resume_action
    out["metadata"] = metadata
    tags = list(out.get("tags") or [])
    for tag in ("agent", "langgraph"):
        if tag not in tags:
            tags.append(tag)
    sid = (session_id or "default").strip() or "default"
    tags.append(f"session:{sid[:12]}")
    if turn_index is not None:
        tags.append(f"turn-{turn_index}")
    if stream and "stream" not in tags:
        tags.append("stream")
    if resume_action:
        tags.append("hitl_resume")
    out["tags"] = tags
    if turn_index is not None:
        out["run_name"] = out.get("run_name") or f"turn-{turn_index}"
    else:
        out["run_name"] = out.get("run_name") or f"agent:{sid[:12]}"
    return out
