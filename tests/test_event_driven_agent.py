from __future__ import annotations

import asyncio

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent.agent_service import _create_agent
from agent.harness import (
    clear_run_context,
    get_abandoned_tools,
    make_progress_control_tool,
    reconcile_task_runtime,
    sync_run_context_from_values,
    wrap_tools_with_phase_gate,
)
from agent.task_runtime import build_step_states
from evals.fake_tools import FakeToolRuntime, build_fake_tools


class StubToolModel(BaseChatModel):
    calls: int = 0

    @property
    def _llm_type(self) -> str:
        return "stub-tool-model"

    def bind_tools(self, tools, **kwargs):
        return self

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        if self.calls == 1:
            message = AIMessage(
                content="",
                tool_calls=[
                    {
                        "id": "tc1",
                        "name": "web_search",
                        "args": {"query": "产品A", "max_results": 5},
                    }
                ],
            )
        else:
            message = AIMessage(content="产品A价格3999元。")
        return ChatResult(generations=[ChatGeneration(message=message)])


def test_real_langgraph_agent_advances_from_tool_event():
    async def run() -> None:
        session_id = "event-driven-agent-test"
        plan = ["搜索产品A的价格"]
        state = {
            "messages": [HumanMessage(content="搜索产品A价格")],
            "user_goal": "搜索产品A价格",
            "plan": plan,
            "plan_index": 0,
            "task_phase": "gather",
            "harness_enabled": True,
            "completed_steps": [],
            "step_checklist": [],
            "task_status": "executing",
            "step_states": build_step_states(plan),
            "tool_events": [],
        }
        runtime = FakeToolRuntime({"search_results": {"产品A": "产品A价格3999元。"}})
        tools = wrap_tools_with_phase_gate(
            [*build_fake_tools(runtime), make_progress_control_tool()]
        )
        sync_run_context_from_values(session_id, state)
        try:
            agent = await _create_agent(StubToolModel(), tools, None)
            output = await agent.ainvoke(
                state,
                config={"configurable": {"thread_id": session_id}},
            )
            assert output["plan_index"] == 1
            assert output["task_status"] == "done"
            assert output["step_states"][0]["status"] == "succeeded"
            assert len(output["tool_events"]) == 1
        finally:
            clear_run_context(session_id)

    asyncio.run(run())


def _runtime_tools(runtime: FakeToolRuntime):
    return wrap_tools_with_phase_gate(
        [*build_fake_tools(runtime), make_progress_control_tool()]
    )


def test_delivery_preflight_fast_forwards_reasoning_only_step():
    session_id = "delivery-fast-forward-test"
    plan = ["搜索产品A资料", "整理并总结资料", "发送邮件到指定邮箱"]
    state = {
        "harness_enabled": True,
        "plan": plan,
        "plan_index": 0,
        "task_phase": "gather",
        "step_states": build_step_states(plan),
        "tool_events": [],
    }
    runtime = FakeToolRuntime({"search_results": {"产品A": "产品A价格3999元。"}})
    tools = _runtime_tools(runtime)
    config = {"configurable": {"thread_id": session_id}}
    sync_run_context_from_values(session_id, state)
    try:
        web = next(t for t in tools if t.name == "web_search")
        web.invoke({"query": "产品A", "max_results": 5}, config=config)
        progress = reconcile_task_runtime(state, session_id)
        state = {**state, **progress}
        assert state["plan_index"] == 1
        sync_run_context_from_values(session_id, state)

        email = next(t for t in tools if t.name == "send_email")
        result = email.invoke(
            {
                "to_email": "a@example.com",
                "content": "产品A资料已经完成整理。核心价格为3999元，并附有完整的分析结论、来源说明和推荐建议。",
            },
            config=config,
        )
        assert "邮件已发送" in result
        final = reconcile_task_runtime(state, session_id)
        assert final["plan_index"] == 3
        assert final["task_status"] == "done"
        assert final["step_states"][1]["status"] == "succeeded"
    finally:
        clear_run_context(session_id)


def test_delivery_wait_does_not_abandon_or_fail_required_process_step():
    session_id = "delivery-wait-test"
    plan = ["搜索产品A资料", "整理成表格", "发送邮件到指定邮箱"]
    state = {
        "harness_enabled": True,
        "plan": plan,
        "plan_index": 0,
        "task_phase": "gather",
        "step_states": build_step_states(plan),
        "tool_events": [],
    }
    runtime = FakeToolRuntime({"search_results": {"产品A": "产品A价格3999元。"}})
    tools = _runtime_tools(runtime)
    config = {"configurable": {"thread_id": session_id}}
    sync_run_context_from_values(session_id, state)
    try:
        web = next(t for t in tools if t.name == "web_search")
        web.invoke({"query": "产品A", "max_results": 5}, config=config)
        progress = reconcile_task_runtime(state, session_id)
        state = {**state, **progress}
        sync_run_context_from_values(session_id, state)
        email = next(t for t in tools if t.name == "send_email")
        for _ in range(4):
            result = email.invoke(
                {
                    "to_email": "a@example.com",
                    "content": "产品A资料已经整理完成，包含价格、参数、来源和分析结论，准备发送给指定收件人。",
                },
                config=config,
            )
            assert "尚未执行" in result
            assert "已放弃" not in result
        final = reconcile_task_runtime(state, session_id)
        assert final["plan_index"] == 1
        assert final["step_states"][1]["status"] == "running"
        assert final["step_states"][1]["attempts"] == 0
        assert "send_email" not in get_abandoned_tools(session_id)
    finally:
        clear_run_context(session_id)


def test_parallel_duplicate_delivery_executes_side_effect_once():
    async def run() -> None:
        session_id = "parallel-delivery-dedup-test"
        plan = ["发送邮件到指定邮箱"]
        state = {
            "harness_enabled": True,
            "user_goal": "发送邮件到 a@example.com",
            "plan": plan,
            "plan_index": 0,
            "task_phase": "deliver",
            "step_states": build_step_states(plan),
            "tool_events": [],
        }
        runtime = FakeToolRuntime({})
        email = next(t for t in _runtime_tools(runtime) if t.name == "send_email")
        config = {"configurable": {"thread_id": session_id}}
        sync_run_context_from_values(session_id, state)
        try:
            args = {
                "to_email": "a@example.com",
                "content": "这是一封包含完整资料、分析结论、来源说明以及推荐建议的测试邮件正文。",
            }
            results = await asyncio.gather(
                email.ainvoke(args, config=config),
                email.ainvoke(args, config=config),
                email.ainvoke(args, config=config),
            )
            assert sum("邮件已发送" in result for result in results) == 1
            assert sum(call["name"] == "send_email" for call in runtime.calls) == 1
        finally:
            clear_run_context(session_id)

    asyncio.run(run())
