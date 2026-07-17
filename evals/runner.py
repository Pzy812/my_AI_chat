"""Run deterministic fake-tool evals against the real ReAct agent.

Usage:
    python -m evals.runner --limit 12 --repeats 1
    python -m evals.runner --repeats 3

The command calls the configured LLM, but never calls the network/file/email MCP
tools.  Fake tools are injected directly into the agent.
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
from pathlib import Path
from time import perf_counter
from typing import Any

# Eval traces can be large (30 cases × repeats × variants); keep them opt-in.
if os.getenv("EVAL_LANGSMITH_TRACING", "0").strip().lower() not in ("1", "true", "yes"):
    os.environ["LANGSMITH_TRACING"] = "false"
    os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_core.messages import HumanMessage

from agent.agent_service import _create_agent
from agent.harness import (
    clear_run_context,
    make_progress_control_tool,
    sync_run_context_from_values,
    wrap_tools_with_phase_gate,
)
from agent.task_checklist import MAX_TASK_CONTINUATIONS, should_continue_task
from agent.task_runtime import build_step_states
from chat.chat_helpers import last_assistant_text
from evals.fake_tools import FakeToolRuntime, build_fake_tools
from evals.graders import grade_case
from evals.report import build_markdown_report
from evals.schemas import EvalCase, EvalResult
from llm.model_config import make_llm_from_config


ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = ROOT / "datasets" / "agent_tasks.jsonl"
DEFAULT_RESULTS = ROOT / "results"
CRITICAL_FAILURES = {"forbidden_tool_called", "duplicate_delivery", "false_success_claim"}


def load_cases(path: Path) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        body = line.strip()
        if not body or body.startswith("#"):
            continue
        try:
            cases.append(EvalCase.from_dict(json.loads(body)))
        except Exception as exc:
            raise ValueError(f"{path}:{line_number}: {exc}") from exc
    return cases


def initial_state(case: EvalCase, variant: str) -> dict[str, Any]:
    harness = variant == "harness"
    plan = list(case.plan) if harness else []
    return {
        "messages": [HumanMessage(content=case.input)],
        "user_goal": case.input,
        "plan": plan,
        "plan_index": 0,
        "task_phase": "gather" if harness else "deliver",
        "harness_enabled": harness,
        "completed_steps": [],
        "step_checklist": [],
        "task_status": "executing",
        "step_states": build_step_states(plan),
        "tool_events": [],
    }


def trajectory_from_state(state: dict[str, Any], runtime: FakeToolRuntime, variant: str) -> list[dict[str, Any]]:
    if variant == "baseline":
        return list(runtime.calls)
    return [
        {
            "name": event.get("tool_name"),
            "arguments": event.get("arguments") or {},
            "success": bool(event.get("success")),
            "output": event.get("output_summary") or "",
            "latency_ms": int(event.get("latency_ms") or 0),
            "step_id": event.get("step_id") or "",
            "executed": event.get("executed", True),
        }
        for event in state.get("tool_events") or []
    ]


async def run_case(case: EvalCase, variant: str, repeat: int, llm: Any) -> EvalResult:
    session_id = f"eval-{variant}-{case.id}-{repeat}"
    clear_run_context(session_id)
    runtime = FakeToolRuntime(case.fixtures)
    tools = [*build_fake_tools(runtime), make_progress_control_tool()]
    tools = wrap_tools_with_phase_gate(tools)
    agent = await _create_agent(llm, tools, None)
    state_input = initial_state(case, variant)
    sync_run_context_from_values(session_id, state_input)
    config = {"configurable": {"thread_id": session_id}, "recursion_limit": 100}

    started = perf_counter()
    state = await agent.ainvoke(state_input, config=config)
    if variant == "harness":
        continuations = 0
        while continuations < MAX_TASK_CONTINUATIONS:
            answer = last_assistant_text(state.get("messages") or [])
            nudge = should_continue_task(state, state.get("messages") or [], answer)
            if not nudge:
                break
            continuations += 1
            next_input = dict(state)
            next_input["messages"] = [*(state.get("messages") or []), HumanMessage(content=nudge)]
            sync_run_context_from_values(session_id, next_input)
            state = await agent.ainvoke(next_input, config=config)

    latency_ms = int((perf_counter() - started) * 1000)
    answer = last_assistant_text(state.get("messages") or [])
    trajectory = trajectory_from_state(state, runtime, variant)
    score, failures = grade_case(case, trajectory, answer)
    success = score >= 80 and not CRITICAL_FAILURES.intersection(failures)
    result = EvalResult(
        task_id=case.id,
        category=case.category,
        variant=variant,
        repeat=repeat,
        success=success,
        score=score,
        tool_calls=len(trajectory),
        latency_ms=latency_ms,
        failures=failures,
        trajectory=trajectory,
        final_answer=answer,
        task_state={
            "plan_index": state.get("plan_index"),
            "task_status": state.get("task_status"),
            "step_states": state.get("step_states") or [],
        },
    )
    clear_run_context(session_id)
    return result


async def async_main(args: argparse.Namespace) -> int:
    cases = load_cases(Path(args.dataset))
    if args.limit:
        cases = cases[: args.limit]
    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    invalid = set(variants) - {"baseline", "harness"}
    if invalid:
        raise ValueError(f"未知 variant: {sorted(invalid)}")
    llm_config = {
        "provider": args.provider,
        "model": args.model,
        "api_key": args.api_key,
        "base_url": args.base_url,
    }
    llm = make_llm_from_config(llm_config)
    results: list[dict[str, Any]] = []
    for repeat in range(1, args.repeats + 1):
        for case in cases:
            for variant in variants:
                print(f"[{repeat}/{args.repeats}] {variant}: {case.id}", flush=True)
                try:
                    result = await run_case(case, variant, repeat, llm)
                    results.append(result.as_dict())
                except Exception as exc:
                    results.append(
                        EvalResult(
                            task_id=case.id,
                            category=case.category,
                            variant=variant,
                            repeat=repeat,
                            success=False,
                            score=0,
                            tool_calls=0,
                            latency_ms=0,
                            failures=["runtime_error"],
                            trajectory=[],
                            final_answer=f"{type(exc).__name__}: {exc}",
                            task_state={},
                        ).as_dict()
                    )

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "runs.jsonl"
    jsonl_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in results),
        encoding="utf-8",
    )
    report_path = output_dir / "comparison.md"
    report_path.write_text(build_markdown_report(results), encoding="utf-8")
    print(f"结果：{jsonl_path}")
    print(f"报告：{report_path}")
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseline vs Task Harness eval")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_RESULTS))
    parser.add_argument("--variants", default="baseline,harness")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--provider", default="zhipu")
    parser.add_argument("--model", default="glm-4-flash")
    parser.add_argument("--api-key", default="")
    parser.add_argument("--base-url", default="")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(async_main(parse_args())))
