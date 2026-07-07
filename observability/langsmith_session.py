"""LangSmith 会话级 trace：一个 session 一个根 trace，每轮用户消息一个 turn 子 run。"""
from __future__ import annotations

import logging
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any, Iterator

from observability.langsmith_config import is_tracing_enabled, langsmith_project, thread_metadata

logger = logging.getLogger("ai_chat.langsmith.session")


@dataclass
class SessionTraceState:
    session_id: str
    root_run_id: str
    root: Any
    turn_index: int = 0
    current_turn: Any | None = None


_sessions: dict[str, SessionTraceState] = {}


def _sid(session_id: str) -> str:
    return (session_id or "default").strip() or "default"


def _short_label(session_id: str, n: int = 16) -> str:
    s = _sid(session_id)
    return s if len(s) <= n else s[:n]


def extract_last_user_text(messages: list | None) -> str:
    """从 LangChain messages 或 dict 列表提取最后一条用户文本。"""
    if not messages:
        return ""
    for m in reversed(messages):
        if isinstance(m, dict):
            role = str(m.get("role") or m.get("type") or "").lower()
            if role not in ("human", "user", "humanmessage"):
                continue
            content = m.get("content")
        else:
            type_name = type(m).__name__.lower()
            if "human" not in type_name:
                role = getattr(m, "type", "") or ""
                if str(role).lower() not in ("human", "user"):
                    continue
            content = getattr(m, "content", None)
        if isinstance(content, str) and content.strip():
            return content.strip()[:500]
    return ""


def _get_or_create_session_root(session_id: str) -> SessionTraceState | None:
    if not is_tracing_enabled():
        return None
    sid = _sid(session_id)
    existing = _sessions.get(sid)
    if existing is not None:
        return existing
    try:
        from langsmith.run_trees import RunTree

        meta = thread_metadata(sid)
        root = RunTree(
            name=f"session:{_short_label(sid)}",
            run_type="chain",
            inputs={"session_id": sid},
            extra={"metadata": meta},
            tags=["session", "chat"],
            project_name=langsmith_project(),
        )
        root.post()
        state = SessionTraceState(session_id=sid, root_run_id=str(root.id), root=root)
        _sessions[sid] = state
        return state
    except Exception as e:
        logger.warning("创建 session trace 失败 session=%s: %s", sid, e)
        return None


def session_root_run_id(session_id: str) -> str | None:
    state = _sessions.get(_sid(session_id))
    return state.root_run_id if state else None


def current_turn_index(session_id: str) -> int | None:
    state = _sessions.get(_sid(session_id))
    if not state:
        return None
    if state.current_turn is not None:
        return state.turn_index
    return state.turn_index or None


def peek_turn_index(session_id: str, *, is_resume: bool = False) -> int | None:
    """供 enrich_agent_config 使用：resume 复用当前 turn，否则为下一 turn。"""
    state = _sessions.get(_sid(session_id))
    if not state:
        return None
    if is_resume and state.current_turn is not None:
        return state.turn_index
    return state.turn_index + 1


@contextmanager
def agent_turn_trace(
    session_id: str,
    *,
    user_input: str = "",
    is_resume: bool = False,
) -> Iterator[Any | None]:
    """为 Agent 调用创建/复用 turn 子 run，LangGraph 及其 LLM/Tool 子 span 均挂在其下。"""
    if not is_tracing_enabled():
        yield None
        return

    sid = _sid(session_id)
    state = _get_or_create_session_root(sid)
    if state is None:
        yield None
        return

    from langsmith.run_helpers import tracing_context

    turn = None
    reuse = is_resume and state.current_turn is not None

    if reuse:
        turn = state.current_turn
    else:
        state.turn_index += 1
        n = state.turn_index
        preview = (user_input or "").strip()[:500]
        meta = thread_metadata(sid, turn=n)
        try:
            turn = state.root.create_child(
                name=f"turn-{n}",
                run_type="chain",
                inputs={"user_input": preview} if preview else {"turn": n},
                extra={"metadata": meta},
                tags=[f"turn-{n}"],
            )
            turn.post()
            state.current_turn = turn
        except Exception as e:
            logger.warning("创建 turn trace 失败 session=%s turn=%s: %s", sid, n, e)
            yield None
            return

    try:
        with tracing_context(parent=turn):
            yield turn
    finally:
        pass


def finish_agent_turn(
    session_id: str,
    *,
    turn_run: Any | None,
    reply: str | None = None,
    hitl_pending: bool = False,
    error: str | None = None,
) -> None:
    """结束当前 turn；HITL 等待时不关闭 turn，便于 resume 续跑在同一 turn 下。"""
    if not turn_run or not is_tracing_enabled():
        return
    if hitl_pending:
        return

    sid = _sid(session_id)
    state = _sessions.get(sid)
    try:
        if error:
            turn_run.end(error=error)
        else:
            outputs: dict[str, Any] = {"status": "completed"}
            if reply:
                outputs["reply"] = reply[:2000]
            turn_run.end(outputs=outputs)
        turn_run.post()
    except Exception as e:
        logger.debug("结束 turn trace 失败: %s", e)
    finally:
        if state:
            state.current_turn = None


def clear_session_trace(session_id: str) -> None:
    """清空/删除会话时结束 session 根 trace。"""
    sid = _sid(session_id)
    state = _sessions.pop(sid, None)
    if not state:
        return
    try:
        if state.current_turn is not None:
            state.current_turn.end(outputs={"status": "interrupted"})
            state.current_turn.post()
        state.root.end(outputs={"turns": state.turn_index, "status": "closed"})
        state.root.post()
    except Exception as e:
        logger.debug("清理 session trace 失败: %s", e)
