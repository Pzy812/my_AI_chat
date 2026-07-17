from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean
from typing import Any


def build_markdown_report(results: list[dict[str, Any]]) -> str:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for result in results:
        grouped[str(result["variant"])].append(result)
    lines = ["# Agent Eval：Baseline vs Harness", "", "## 总体结果", ""]
    lines.append("| 版本 | 运行数 | 完成率 | 平均分 | 平均工具调用 | 平均延迟 |")
    lines.append("|---|---:|---:|---:|---:|---:|")
    for variant in ("baseline", "harness"):
        rows = grouped.get(variant) or []
        if not rows:
            continue
        rate = 100 * sum(bool(r["success"]) for r in rows) / len(rows)
        lines.append(
            f"| {variant} | {len(rows)} | {rate:.1f}% | "
            f"{mean(float(r['score']) for r in rows):.1f} | "
            f"{mean(int(r['tool_calls']) for r in rows):.2f} | "
            f"{mean(int(r['latency_ms']) for r in rows):.0f} ms |"
        )

    lines.extend(["", "## 分类别完成率", "", "| 类别 | Baseline | Harness |", "|---|---:|---:|"])
    categories = sorted({str(r["category"]) for r in results})
    for category in categories:
        values = []
        for variant in ("baseline", "harness"):
            rows = [r for r in grouped.get(variant, []) if r["category"] == category]
            values.append(f"{100 * sum(bool(r['success']) for r in rows) / len(rows):.1f}%" if rows else "-")
        lines.append(f"| {category} | {values[0]} | {values[1]} |")

    lines.extend(["", "## 失败类型", "", "| 失败类型 | Baseline | Harness |", "|---|---:|---:|"])
    counters = {
        variant: Counter(f for r in rows for f in r.get("failures") or [])
        for variant, rows in grouped.items()
    }
    failure_types = sorted(set(counters.get("baseline", {})) | set(counters.get("harness", {})))
    for failure in failure_types:
        lines.append(
            f"| {failure} | {counters.get('baseline', {}).get(failure, 0)} | "
            f"{counters.get('harness', {}).get(failure, 0)} |"
        )
    return "\n".join(lines) + "\n"
