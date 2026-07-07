"""LangSmith 可观测性 API。"""
from __future__ import annotations

import time

from flask import Blueprint, jsonify, request

from observability.langsmith_config import is_tracing_enabled, public_status
from observability.langsmith_trace import fetch_trace_summary

bp = Blueprint("observability", __name__)


@bp.route("/observability/langsmith/status", methods=["GET", "POST"])
def langsmith_status():
    return jsonify({"code": 0, **public_status()})


@bp.route("/observability/langsmith/trace", methods=["GET"])
def langsmith_trace_detail():
    run_id = (request.args.get("run_id") or "").strip()
    if not run_id:
        return jsonify({"code": -1, "msg": "缺少 run_id 参数"})
    if not is_tracing_enabled():
        return jsonify({"code": -1, "msg": "LangSmith 未启用，请在 .env 中配置 LANGSMITH_API_KEY"})

    retries = max(1, min(int(request.args.get("retries") or 5), 8))
    delay_sec = max(0.5, min(float(request.args.get("delay") or 1.0), 5.0))
    summary = None
    for attempt in range(retries):
        summary = fetch_trace_summary(run_id)
        if summary and summary.get("step_count", 0) > 0:
            break
        if attempt + 1 < retries:
            time.sleep(delay_sec)

    if not summary:
        return jsonify(
            {
                "code": -1,
                "msg": "无法读取 trace（LangSmith 可能仍在写入，请稍后重试或点击外链查看）",
                "run_id": run_id,
            }
        )
    return jsonify({"code": 0, "trace": summary})
