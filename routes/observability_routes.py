"""LangSmith 可观测性 API。"""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Request

from observability.langsmith_config import is_tracing_enabled, public_status
from observability.langsmith_trace import fetch_trace_summary

router = APIRouter(tags=["observability"])


@router.api_route("/observability/langsmith/status", methods=["GET", "POST"])
async def langsmith_status():
    return {"code": 0, **public_status()}


@router.get("/observability/langsmith/trace")
async def langsmith_trace_detail(request: Request):
    run_id = (request.query_params.get("run_id") or "").strip()
    if not run_id:
        return {"code": -1, "msg": "缺少 run_id 参数"}
    if not is_tracing_enabled():
        return {"code": -1, "msg": "LangSmith 未启用，请在 .env 中配置 LANGSMITH_API_KEY"}

    try:
        retries = max(1, min(int(request.query_params.get("retries") or 5), 8))
    except (TypeError, ValueError):
        retries = 5
    try:
        delay_sec = max(0.5, min(float(request.query_params.get("delay") or 1.0), 5.0))
    except (TypeError, ValueError):
        delay_sec = 1.0

    summary = None
    for attempt in range(retries):
        summary = await asyncio.to_thread(fetch_trace_summary, run_id)
        if summary and summary.get("step_count", 0) > 0:
            break
        if attempt + 1 < retries:
            await asyncio.sleep(delay_sec)

    if not summary:
        return {
            "code": -1,
            "msg": "无法读取 trace（LangSmith 可能仍在写入，请稍后重试或点击外链查看）",
            "run_id": run_id,
        }
    return {"code": 0, "trace": summary}
