"""Deterministic, side-effect-free tools used by agent evals."""
from __future__ import annotations

from dataclasses import dataclass, field
from time import perf_counter
from typing import Any, Callable

from langchain_core.tools import StructuredTool


@dataclass
class FakeToolRuntime:
    fixtures: dict[str, Any]
    calls: list[dict[str, Any]] = field(default_factory=list)
    _counts: dict[str, int] = field(default_factory=dict)

    def _behavior(self, name: str, default: str) -> tuple[bool, str]:
        sequence = (self.fixtures.get("tool_behaviors") or {}).get(name) or []
        index = self._counts.get(name, 0)
        self._counts[name] = index + 1
        if index < len(sequence):
            item = sequence[index]
            return bool(item.get("success")), str(item.get("output") or "")
        success = not any(marker in default for marker in ("❌", "失败", "不存在", "请求超时"))
        return success, default

    def invoke(self, name: str, args: dict[str, Any], operation: Callable[[], str]) -> str:
        started = perf_counter()
        try:
            default = operation()
            success, output = self._behavior(name, default)
        except Exception as exc:
            success, output = False, f"❌ {type(exc).__name__}: {exc}"
        self.calls.append(
            {
                "name": name,
                "arguments": args,
                "success": success,
                "output": output[:2000],
                "latency_ms": int((perf_counter() - started) * 1000),
            }
        )
        return output if success else (output if output.startswith("❌") else f"❌ {output}")

    def search_result(self, query: str) -> str:
        results = self.fixtures.get("search_results") or {}
        query_lower = query.lower()
        matched = [str(v) for k, v in results.items() if str(k).lower() in query_lower]
        if matched:
            return "\n".join(matched)
        if results:
            return "\n".join(str(v) for v in results.values())
        return "未找到结果"


def build_fake_tools(runtime: FakeToolRuntime) -> list[StructuredTool]:
    def get_current_time() -> str:
        return runtime.invoke(
            "get_current_time", {}, lambda: str(runtime.fixtures.get("current_time") or "2026-07-17 10:00:00")
        )

    def web_search(query: str, max_results: int = 5) -> str:
        args = {"query": query, "max_results": max_results}
        return runtime.invoke("web_search", args, lambda: runtime.search_result(query))

    def web_search_batch(queries: list[str], max_results: int = 5) -> str:
        args = {"queries": queries, "max_results": max_results}
        return runtime.invoke(
            "web_search_batch",
            args,
            lambda: "\n\n".join(f"### {q}\n{runtime.search_result(q)}" for q in queries),
        )

    def list_local_directory(directory: str) -> str:
        args = {"directory": directory}
        files = runtime.fixtures.get("files") or {}
        return runtime.invoke("list_local_directory", args, lambda: "\n".join(sorted(files)))

    def glob_local_files(directory: str, pattern: str = "*", recursive: bool = True) -> str:
        args = {"directory": directory, "pattern": pattern, "recursive": recursive}
        files = runtime.fixtures.get("files") or {}
        return runtime.invoke("glob_local_files", args, lambda: "\n".join(sorted(files)))

    def read_local_file(file_path: str) -> str:
        args = {"file_path": file_path}
        files = runtime.fixtures.get("files") or {}

        def operation() -> str:
            if file_path in files:
                return str(files[file_path])
            for path, body in files.items():
                if file_path.endswith(str(path)):
                    return str(body)
            return "❌ 文件不存在"

        return runtime.invoke("read_local_file", args, operation)

    def format_pretty_table(headers: list, rows: list, align: str = "c", format_type: str = "markdown") -> str:
        args = {"headers": headers, "rows": rows, "align": align, "format_type": format_type}
        return runtime.invoke(
            "format_pretty_table",
            args,
            lambda: "| " + " | ".join(map(str, headers)) + " |\n" + "\n".join(
                "| " + " | ".join(map(str, row)) + " |" for row in rows
            ),
        )

    def export_to_excel(headers: list, rows: list, filename: str | None = None) -> str:
        args = {"headers": headers, "rows": rows, "filename": filename}
        safe = filename or "eval_output.xlsx"
        return runtime.invoke("export_to_excel", args, lambda: f"✅ Excel 已保存：/fake/exports/{safe}")

    def send_email(to_email: str, content: str) -> str:
        args = {"to_email": to_email, "content": content}
        return runtime.invoke("send_email", args, lambda: f"✅ 邮件已发送到：{to_email}")

    specs = [
        (get_current_time, "获取固定的评测时间。"),
        (web_search, "搜索固定评测数据。"),
        (web_search_batch, "并行搜索多个固定评测主题。"),
        (list_local_directory, "列出评测目录。"),
        (glob_local_files, "匹配评测文件。"),
        (read_local_file, "读取评测文件。"),
        (format_pretty_table, "将数据格式化为 Markdown 表格。"),
        (export_to_excel, "模拟导出 Excel，不写真实文件。"),
        (send_email, "模拟发送邮件，不产生真实外发。"),
    ]
    return [StructuredTool.from_function(func=func, description=description) for func, description in specs]
