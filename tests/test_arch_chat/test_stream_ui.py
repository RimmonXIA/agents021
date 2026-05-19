"""Tests for streaming TUI."""

from __future__ import annotations

import pytest

from arch_chat.config import Settings
from arch_chat.models.context import RunContext, RunResult
from arch_chat.tui.stream_ui import NullStreamUI


@pytest.fixture
def fake_settings():
    return Settings(DEEPSEEK_API_KEY="test-key")  # type: ignore[call-arg]


class FakeModel:
    id = "fake"


def test_add_trace_emits_step_end(fake_settings: Settings):
    sink = NullStreamUI()
    ctx = RunContext(
        message="hi",
        settings=fake_settings,
        session_id="s1",
        user_id="u1",
        model=FakeModel(),  # type: ignore[arg-type]
        stream=sink,
    )
    ctx.add_trace("test_step", "hello world")
    assert ("step_end", "test_step", "hello world") in sink.events


def test_emit_step_start(fake_settings: Settings):
    sink = NullStreamUI()
    ctx = RunContext(
        message="hi",
        settings=fake_settings,
        session_id="s1",
        user_id="u1",
        model=FakeModel(),  # type: ignore[arg-type]
        stream=sink,
    )
    ctx.emit_step_start("generator", "generator")
    assert ("step_start", "generator", "generator") in sink.events


def test_null_stream_records_lifecycle(fake_settings: Settings):
    sink = NullStreamUI()
    from arch_chat.models.state import RoutingDecision

    sink.start_turn("who are you?")
    sink.phase("routing")
    sink.routing(
        RoutingDecision(architecture="meta_controller", confidence=0.9, reasoning="test"),
    )
    sink.token("I am ")
    sink.token("an assistant.")
    sink.finalize(RunResult(content="I am an assistant."))

    event_types = [e[0] for e in sink.events]
    assert event_types == [
        "start_turn",
        "phase",
        "routing",
        "token",
        "token",
        "finalize",
    ]


@pytest.mark.asyncio
async def test_run_agent_stream_events(fake_settings: Settings):
    from arch_chat.runner import AgentRunResult, run_agent

    sink = NullStreamUI()
    ctx = RunContext(
        message="hello",
        settings=fake_settings,
        session_id="s1",
        user_id="u1",
        model=FakeModel(),  # type: ignore[arg-type]
        stream=sink,
    )

    class FakeAgent:
        name = "test_agent"

    async def fake_run_agent_stream(agent, prompt, **kwargs):
        on_chunk = kwargs.get("on_chunk")
        on_reasoning = kwargs.get("on_reasoning")

        class Chunk:
            content = "Hello"
            reasoning_content = "thinking..."

        if on_reasoning:
            on_reasoning("thinking...")
        if on_chunk:
            await on_chunk(Chunk())
        return AgentRunResult(success=True, content="Hello")

    import arch_chat.runner as runner_mod

    original = runner_mod.run_agent_stream
    runner_mod.run_agent_stream = fake_run_agent_stream  # type: ignore[assignment]
    try:
        result = await run_agent(FakeAgent(), "hello", ctx=ctx, step_name="test_agent")  # type: ignore[arg-type]
    finally:
        runner_mod.run_agent_stream = original

    assert result.success
    assert result.content == "Hello"
    types = [e[0] for e in sink.events]
    assert "step_start" in types
    assert "token" in types
    assert "reasoning" in types
