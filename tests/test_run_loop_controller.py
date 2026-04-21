import asyncio

import pytest

from core.engine.run_loop_controller import RunLoopController
from core.memory.blackboard import Blackboard
from core.models import AtomicTask


class _Compactor:
    async def compact(self, shared_memory):  # noqa: ANN001
        return shared_memory


@pytest.mark.asyncio
async def test_run_loop_marks_failed_when_pending_tasks_never_become_ready(
    temp_db: str, temp_lancedb: str
) -> None:
    bb = Blackboard(session_id="run_loop_failed", original_intent="blocked")

    async def decompose() -> None:
        await bb.add_todo(
            AtomicTask(
                id="blocked",
                description="Blocked",
                required_capabilities=["search"],
                expected_output="x",
                depends_on=["missing"],
            )
        )

    async def execute_task(task, running):  # noqa: ANN001
        del task, running
        await asyncio.sleep(0)

    async def on_terminate() -> None:
        return None

    loop = RunLoopController(
        bb,
        decompose,
        execute_task,
        _Compactor(),  # type: ignore[arg-type]
        ui_callback=None,
        on_terminate=on_terminate,
    )

    await loop.run()
    assert bb.state.status == "failed"
