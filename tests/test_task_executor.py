import asyncio
from unittest.mock import patch

import pytest

from core.agents.runner import AgentRunResult
from core.engine.task_executor import TaskExecutor
from core.memory.blackboard import Blackboard
from core.models import AtomicTask


class _Synthesizer:
    def __init__(self) -> None:
        self.last_context: dict[str, object] = {}

    def synthesize(self, capability: str, context: dict[str, object]):  # noqa: ANN001
        self.last_context = context

        class _Agent:
            name = capability

        return _Agent()


@pytest.mark.asyncio
async def test_task_executor_applies_changeset_and_world_context(
    temp_db: str, temp_lancedb: str
) -> None:
    bb = Blackboard(session_id="task_executor", original_intent="execute")
    await bb.update_context("seed", "value")

    synth = _Synthesizer()
    executor = TaskExecutor(bb, synth, ui_callback=None, concurrency_limit=asyncio.Semaphore(2))
    task = AtomicTask(
        id="task1",
        description="Task 1",
        required_capabilities=["search"],
        context_keys=["seed"],
        expected_output="done",
    )

    with patch(
        "core.engine.task_executor.run_agent",
        return_value=AgentRunResult(success=True, content="result"),
    ):
        await executor.execute(task, {})

    assert bb.state.shared_memory["task1_result"] == "result"
    assert bb.state.completed_tasks[0].id == "task1"
    assert "current_date" in synth.last_context
    assert synth.last_context["seed"] == "value"
