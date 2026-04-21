"""Per-task sub-agent synthesis and blackboard updates."""
from __future__ import annotations

import asyncio
from typing import Any

from core.agents.runner import run_agent
from core.agents.synthesizer import AgentSynthesizer
from core.memory.blackboard import Blackboard
from core.models import AtomicTask, SubAgentResult, TrajectoryStep
from core.utils.logging import get_logger

logger = get_logger(__name__)


class TaskExecutor:
    def __init__(
        self,
        blackboard: Blackboard,
        synthesizer: AgentSynthesizer,
        ui_callback: Any,
        concurrency_limit: asyncio.Semaphore,
    ) -> None:
        self.bb = blackboard
        self.asynth = synthesizer
        self.ui_callback = ui_callback
        self.concurrency_limit = concurrency_limit
        self.step_counter = 0

    async def execute(
        self,
        task: AtomicTask,
        running_tasks_map: dict[str, tuple[AtomicTask, asyncio.Task[None]]],
    ) -> None:
        async with self.concurrency_limit:
            self.step_counter += 1
            local_step_id = self.step_counter
            logger.info(f"Executing Task {local_step_id}: {task.description}")

            sibling_ids = [tid for tid in running_tasks_map if tid != task.id]
            await self.bb.sync_world_state()
            context = await self.bb.get_context(task.context_keys + task.required_keys)
            # Clock/calendar live on world_state, not shared_memory; always merge so grounding and templates see today.
            context.update(self.bb.world_context_patch())

            if self.ui_callback:
                self.ui_callback("task_status", {"task_id": task.id, "status": "Synthesizing Agent..."})

            primary_capability = task.required_capabilities[0] if task.required_capabilities else "search"
            try:
                sub_agent = self.asynth.synthesize(primary_capability, context)
            except ValueError as e:
                logger.error(f"Synthesizer error: {e}")
                await self._record_failure(task, str(e), local_step_id)
                return

            if self.ui_callback:
                self.ui_callback("task_status", {"task_id": task.id, "status": "Running Sub-Agent..."})

            prompt = f"Task: {task.description}\nExpected Output: {task.expected_output}"
            output = await run_agent(sub_agent, prompt, augment_prompt_on_parse_retry=True)

            if isinstance(output, str):
                if self.ui_callback:
                    self.ui_callback("task_status", {"task_id": task.id, "status": "finished"})

                result = SubAgentResult(task_id=task.id, status="success", output=output)
                changeset = {f"{task.id}_result": output}
                await self.bb.apply_changeset(task, changeset)
                await self.bb.record_step(
                    TrajectoryStep(
                        step_id=local_step_id,
                        task=task,
                        result=result,
                        parent_ids=task.depends_on,
                        sibling_ids=sibling_ids,
                    )
                )
                if self.ui_callback:
                    self.ui_callback(
                        "task_result",
                        {"task_id": task.id, "output": output},
                    )
            else:
                error_msg = f"Task execution failed: {output}"
                logger.error(error_msg)
                await self._record_failure(task, error_msg, local_step_id)

    async def _record_failure(self, task: AtomicTask, error_msg: str, step_id: int) -> None:
        result = SubAgentResult(task_id=task.id, status="error", output=error_msg)
        await self.bb.record_step(TrajectoryStep(step_id=step_id, task=task, result=result))
