"""LangSmith run 捕获与 trace 树（供 API / 前端瀑布流展示）。"""
from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any
from uuid import UUID

from langchain_core.callbacks.base import BaseCallbackHandler

from observability.langsmith_config import (
    build_thread_url,
    is_tracing_enabled,
    langsmith_project,
    langsmith_web_base,
)
from observability.langsmith_session import (
    current_turn_index,
    session_root_run_id,
)

logger = logging.getLogger("ai_chat.langsmith")

_run_captures: dict[str, RunIdCapture] = {}
_IO_PREVIEW_MAX = int(__import__("os").getenv("LANGSMITH_IO_PREVIEW_MAX", "8000"))


class RunIdCapture(BaseCallbackHandler):
    """捕获 LangGraph 根 chain 的 run_id（同步 + 异步链均支持）。"""

    _ROOT_NAMES = frozenset({"LangGraph", "agent", "RunnableSequence"})

    def __init__(self) -> None:
        super().__init__()
        self.root_run_id: str | None = None
        self._fallback_run_id: str | None = None

    def _record(self, run_id: UUID, parent_run_id: UUID | None, name: str = "") -> None:
        rid = str(run_id)
        label = (name or "").strip()
        if label in self._ROOT_NAMES or label.startswith("chat:"):
            self.root_run_id = rid
            return
        if parent_run_id is None and self.root_run_id is None:
            if self._fallback_run_id is None:
                self._fallback_run_id = rid
            if self.root_run_id is None:
                self.root_run_id = rid

    def on_chain_start(
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        name = ""
        if isinstance(serialized, dict):
            name = str(serialized.get("name") or serialized.get("id") or "")
        self._record(run_id, parent_run_id, name)

    async def on_chain_start(  # type: ignore[override]
        self,
        serialized: dict[str, Any],
        inputs: dict[str, Any],
        *,
        run_id: UUID,
        parent_run_id: UUID | None = None,
        **kwargs: Any,
    ) -> None:
        name = ""
        if isinstance(serialized, dict):
            name = str(serialized.get("name") or serialized.get("id") or "")
        self._record(run_id, parent_run_id, name)


def attach_run_capture(config: dict[str, Any], session_id: str) -> RunIdCapture | None:
    """向 Agent config 附加 callback，并登记 session → capture。"""
    if not is_tracing_enabled():
        return None
    sid = session_id or "default"
    capture = RunIdCapture()
    _run_captures[sid] = capture
    callbacks = list(config.get("callbacks") or [])
    callbacks.append(capture)
    config["callbacks"] = callbacks
    return capture


def build_run_url(run_id: str) -> str:
    return f"{langsmith_web_base()}/?peek={run_id}"


def build_trace_payload(
    run_id: str | None,
    *,
    session_id: str = "",
    include_summary: bool = False,
) -> dict[str, Any] | None:
    """构造可返回前端的 LangSmith 摘要（不含密钥）。"""
    if not is_tracing_enabled():
        return None
    sid = session_id or "default"
    session_root = session_root_run_id(sid)
    thread_id = sid
    primary_run_id = session_root or run_id
    if not run_id and not session_root:
        return {
            "enabled": True,
            "project": langsmith_project(),
            "session_id": sid,
            "thread_id": thread_id,
            "run_id": "",
            "session_root_run_id": "",
            "url": build_thread_url(thread_id),
            "thread_url": build_thread_url(thread_id),
            "status": "pending",
            "message": "未捕获到 run_id（可能 tracing 尚未就绪）",
        }
    payload: dict[str, Any] = {
        "enabled": True,
        "project": langsmith_project(),
        "session_id": sid,
        "thread_id": thread_id,
        "run_id": primary_run_id or run_id or "",
        "session_root_run_id": session_root or "",
        "turn_run_id": run_id if session_root and run_id and run_id != session_root else "",
        "turn": current_turn_index(sid),
        "url": build_run_url(primary_run_id or run_id or ""),
        "thread_url": build_thread_url(thread_id),
        "status": "ready",
    }
    if include_summary and primary_run_id:
        summary = fetch_trace_summary(primary_run_id)
        if summary:
            payload["summary"] = summary
    return payload


def _pick_best_run_id(*candidates: str | None) -> str | None:
    """优先使用 astream_events 捕获的根 run（第一个参数），其次 callback。"""
    cleaned = [str(c).strip() for c in candidates if c and str(c).strip()]
    return cleaned[0] if cleaned else None


def finalize_trace_for_session(
    session_id: str,
    *,
    event_root_run_id: str | None = None,
    include_summary: bool = False,
) -> dict[str, Any] | None:
    """读取并清理 session 对应的 run capture，优先返回会话根 trace。"""
    sid = session_id or "default"
    capture = _run_captures.pop(sid, None)
    capture_id = capture.root_run_id if capture else None
    if capture and not capture_id and capture._fallback_run_id:
        capture_id = capture._fallback_run_id
    session_root = session_root_run_id(sid)
    run_id = _pick_best_run_id(session_root, event_root_run_id, capture_id)
    return build_trace_payload(
        run_id,
        session_id=sid,
        include_summary=include_summary,
    )


def _resolve_trace_root(client: Any, run_id: str) -> Any:
    """沿 parent_run_id 上溯到 trace 根 run（避免误用 ChatZhipuAI 等子 run）。"""
    run = client.read_run(run_id)
    seen: set[str] = set()
    while getattr(run, "parent_run_id", None):
        pid = str(run.parent_run_id)
        if pid in seen:
            break
        seen.add(str(run.id))
        run = client.read_run(pid)
    return run


def _latency_ms(run: Any) -> int | None:
    start = getattr(run, "start_time", None)
    end = getattr(run, "end_time", None)
    if start is None or end is None:
        return None
    try:
        return max(0, int((end - start).total_seconds() * 1000))
    except Exception:
        return None


def _latency_sec_str(run: Any) -> str | None:
    ms = _latency_ms(run)
    if ms is None:
        return None
    return f"{ms / 1000:.2f}s"


def _run_tokens(run: Any) -> int | None:
    usage = getattr(run, "total_tokens", None)
    if usage is not None:
        return int(usage)
    extra = getattr(run, "extra", None) or {}
    if isinstance(extra, dict):
        meta = extra.get("metadata") or {}
        if isinstance(meta, dict) and meta.get("total_tokens") is not None:
            return int(meta["total_tokens"])
    outputs = getattr(run, "outputs", None)
    if isinstance(outputs, dict):
        for key in ("token_usage", "usage", "llm_output"):
            block = outputs.get(key)
            if isinstance(block, dict):
                total = block.get("total_tokens") or block.get("total")
                if total is not None:
                    return int(total)
    return None


def _clip_text(text: str, cap: int = _IO_PREVIEW_MAX) -> str:
    s = (text or "").strip()
    if len(s) <= cap:
        return s
    return s[:cap] + f"\n…（已截断，全文 {len(s)} 字符）"


def _serialize_value(value: Any, depth: int = 0) -> Any:
    if depth > 6:
        return "…"
    if value is None or isinstance(value, (bool, int, float)):
        return value
    if isinstance(value, str):
        return _clip_text(value, min(_IO_PREVIEW_MAX, 4000))
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, UUID):
        return str(value)
    if isinstance(value, dict):
        return {str(k): _serialize_value(v, depth + 1) for k, v in list(value.items())[:40]}
    if isinstance(value, (list, tuple)):
        return [_serialize_value(v, depth + 1) for v in list(value)[:30]]
    return _clip_text(str(value), 2000)


def _extract_message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict):
                if block.get("type") == "text":
                    parts.append(str(block.get("text") or ""))
                elif "text" in block:
                    parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
        return "\n".join(p for p in parts if p)
    return str(content)


def _preview_io(run: Any) -> tuple[Any | None, Any | None, str | None, str | None]:
    """返回 (inputs, outputs, input_text, output_text)。"""
    inputs = _serialize_value(getattr(run, "inputs", None))
    outputs = _serialize_value(getattr(run, "outputs", None))
    input_text = ""
    output_text = ""

    if isinstance(inputs, dict):
        if "messages" in inputs:
            msgs = inputs.get("messages") or []
            if isinstance(msgs, list) and msgs:
                last = msgs[-1] if isinstance(msgs[-1], dict) else {}
                input_text = _extract_message_text(
                    last.get("content") if isinstance(last, dict) else last
                )
        elif "input" in inputs:
            input_text = _extract_message_text(inputs.get("input"))
        else:
            input_text = _clip_text(json.dumps(inputs, ensure_ascii=False, default=str), 3000)

    if isinstance(outputs, dict):
        if "generations" in outputs:
            gens = outputs.get("generations") or []
            if gens and isinstance(gens[0], list) and gens[0]:
                g0 = gens[0][0]
                if isinstance(g0, dict):
                    msg = g0.get("message") or g0.get("text") or g0
                    if isinstance(msg, dict):
                        output_text = _extract_message_text(msg.get("content"))
                    else:
                        output_text = str(msg)
        elif "messages" in outputs:
            msgs = outputs.get("messages") or []
            if isinstance(msgs, list) and msgs:
                last = msgs[-1]
                if isinstance(last, dict):
                    output_text = _extract_message_text(last.get("content"))
        elif "output" in outputs:
            output_text = _extract_message_text(outputs.get("output"))
        else:
            output_text = _clip_text(json.dumps(outputs, ensure_ascii=False, default=str), 3000)
    elif outputs is not None:
        output_text = _clip_text(str(outputs), 3000)

    return inputs, outputs, input_text or None, output_text or None


def _run_to_node(run: Any) -> dict[str, Any]:
    rtype = getattr(run, "run_type", "") or "run"
    name = getattr(run, "name", "") or rtype
    inputs, outputs, input_text, output_text = _preview_io(run)
    start = getattr(run, "start_time", None)
    node: dict[str, Any] = {
        "id": str(getattr(run, "id", "")),
        "parent_id": str(run.parent_run_id) if getattr(run, "parent_run_id", None) else None,
        "trace_id": str(getattr(run, "trace_id", "") or ""),
        "name": name,
        "run_type": rtype,
        "status": getattr(run, "status", "") or "",
        "latency_ms": _latency_ms(run),
        "latency": _latency_sec_str(run),
        "start_time": start.isoformat() if start else None,
        "tokens": _run_tokens(run) if rtype == "llm" else None,
        "inputs": inputs,
        "outputs": outputs,
        "input_text": input_text,
        "output_text": output_text,
        "error": getattr(run, "error", None),
    }
    order = getattr(run, "dotted_order", None)
    if order is not None:
        node["dotted_order"] = str(order)
    return node


def _build_tree(nodes: list[dict[str, Any]], root_id: str) -> list[dict[str, Any]]:
    by_id = {n["id"]: {**n, "children": []} for n in nodes if n.get("id")}
    roots: list[dict[str, Any]] = []
    for node in by_id.values():
        pid = node.get("parent_id")
        if pid and pid in by_id:
            by_id[pid]["children"].append(node)
        elif node["id"] == root_id or not pid:
            roots.append(node)
    if not roots and root_id in by_id:
        roots = [by_id[root_id]]

    def _sort_branch(branch: list[dict[str, Any]]) -> None:
        branch.sort(key=lambda n: n.get("start_time") or n.get("dotted_order") or n.get("id") or "")
        for child in branch:
            if child.get("children"):
                _sort_branch(child["children"])

    _sort_branch(roots)
    return roots


def _flatten_waterfall(tree: list[dict[str, Any]], depth: int = 0) -> list[dict[str, Any]]:
    """深度优先展开为瀑布流行（类似 LangSmith Waterfall）。"""
    rows: list[dict[str, Any]] = []
    for node in tree:
        row = {k: v for k, v in node.items() if k != "children"}
        row["depth"] = depth
        rows.append(row)
        children = node.get("children") or []
        if children:
            rows.extend(_flatten_waterfall(children, depth + 1))
    return rows


def _list_trace_runs(client: Any, root: Any, project: str) -> list[Any]:
    trace_id = str(getattr(root, "trace_id", "") or getattr(root, "id", ""))
    runs: list[Any] = []
    kwargs_list = [
        {"trace_id": trace_id, "limit": 200},
        {"trace_id": trace_id, "project_name": project, "limit": 200},
        {"project_name": project, "filter": f'eq(trace_id, "{trace_id}")', "limit": 200},
        {"project_name": project, "filter": f'has(trace_id, "{trace_id}")', "limit": 200},
    ]
    best: list[Any] = []
    for kwargs in kwargs_list:
        try:
            batch = list(client.list_runs(**kwargs))
            if len(batch) > len(best):
                best = batch
        except Exception as e:
            logger.debug("list_runs 失败 kwargs=%s: %s", kwargs, e)
    runs = best
    if not runs:
        try:
            runs = list(
                client.list_runs(
                    project_name=project,
                    filter=f'eq(parent_run_id, "{root.id}")',
                    limit=200,
                )
            )
        except Exception as e:
            logger.debug("list_runs children 失败: %s", e)
    if not runs:
        runs = [root]
    run_map = {str(r.id): r for r in runs}
    if str(root.id) not in run_map:
        run_map[str(root.id)] = root
    # 递归补齐父链，避免 waterfall 缺层
    for rid in list(run_map.keys()):
        try:
            r = run_map[rid]
            pid = getattr(r, "parent_run_id", None)
            while pid and str(pid) not in run_map:
                parent = client.read_run(str(pid))
                run_map[str(parent.id)] = parent
                pid = getattr(parent, "parent_run_id", None)
        except Exception:
            pass
    return list(run_map.values())


def fetch_trace_summary(run_id: str) -> dict[str, Any] | None:
    """从 LangSmith API 拉取 trace 瀑布流树（供前端展示）。"""
    if not is_tracing_enabled() or not run_id:
        return None
    try:
        from langsmith import Client

        client = Client()
        root = _resolve_trace_root(client, run_id)
        project = langsmith_project()
        all_runs = _list_trace_runs(client, root, project)
        nodes = [_run_to_node(r) for r in all_runs]
        root_id = str(root.id)
        tree = _build_tree(nodes, root_id)
        waterfall = _flatten_waterfall(tree)

        # 若树构建失败，按开始时间平铺（至少展示 LLM/Tool 步骤）
        if not waterfall and nodes:
            ordered = sorted(
                nodes,
                key=lambda n: n.get("start_time") or n.get("dotted_order") or n.get("id") or "",
            )
            waterfall = [{**n, "depth": 0} for n in ordered]

        llm_runs = [n for n in nodes if n.get("run_type") == "llm"]
        tool_runs = [n for n in nodes if n.get("run_type") == "tool"]
        chain_runs = [n for n in nodes if n.get("run_type") == "chain"]
        total_tokens = sum(n.get("tokens") or 0 for n in llm_runs) or None

        return {
            "run_id": root_id,
            "requested_run_id": run_id,
            "trace_id": str(getattr(root, "trace_id", "") or root_id),
            "name": getattr(root, "name", "") or "agent",
            "session_id": (
                (getattr(root, "extra", None) or {}).get("metadata", {}).get("session_id")
                if isinstance(getattr(root, "extra", None), dict)
                else None
            ),
            "status": getattr(root, "status", "") or "",
            "latency_ms": _latency_ms(root),
            "latency": _latency_sec_str(root),
            "total_tokens": total_tokens,
            "llm_calls": len(llm_runs),
            "tool_calls": len(tool_runs),
            "chain_calls": len(chain_runs),
            "step_count": len(nodes),
            "waterfall": waterfall,
            "tree": tree,
            "steps": waterfall,
            "url": build_run_url(root_id),
            "project": project,
        }
    except Exception as e:
        logger.warning("LangSmith trace 摘要拉取失败 run_id=%s: %s", run_id, e)
        return None
