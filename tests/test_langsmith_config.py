"""LangSmith 配置与会话 trace 单元测试（不调用外部 API）。"""
from __future__ import annotations

import os

from observability.langsmith_config import (
    build_thread_url,
    enrich_agent_config,
    is_tracing_enabled,
    thread_metadata,
)
from observability.langsmith_session import (
    clear_session_trace,
    extract_last_user_text,
    peek_turn_index,
)


def test_enrich_agent_config_adds_thread_metadata(monkeypatch):
    monkeypatch.setenv("LANGSMITH_API_KEY", "test-key")
    monkeypatch.setenv("LANGSMITH_TRACING", "1")
    base = {"configurable": {"thread_id": "s1"}, "recursion_limit": 50}
    out = enrich_agent_config(
        base,
        session_id="s1",
        rag_mode="graphrag",
        stream=True,
        turn_index=2,
    )
    assert out["metadata"]["session_id"] == "s1"
    assert out["metadata"]["thread_id"] == "s1"
    assert out["metadata"]["turn"] == 2
    assert out["metadata"]["rag_mode"] == "graphrag"
    assert "stream" in out["tags"]
    assert "turn-2" in out["tags"]
    assert out["run_name"] == "turn-2"


def test_tracing_disabled_without_key(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    assert is_tracing_enabled() is False


def test_thread_metadata_includes_both_keys():
    meta = thread_metadata("abc", turn=1)
    assert meta["thread_id"] == "abc"
    assert meta["session_id"] == "abc"
    assert meta["turn"] == 1


def test_build_thread_url_contains_thread_id():
    url = build_thread_url("my-session")
    assert "my-session" in url
    assert "projects/p/" in url


def test_extract_last_user_text():
    from langchain_core.messages import AIMessage, HumanMessage

    msgs = [HumanMessage(content="你好"), AIMessage(content="嗨"), HumanMessage(content="再问")]
    assert extract_last_user_text(msgs) == "再问"


def test_peek_turn_index_without_tracing(monkeypatch):
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    clear_session_trace("s-peek")
    assert peek_turn_index("s-peek") is None
