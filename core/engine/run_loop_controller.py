"""Main asyncio loop: schedule ready tasks, wait for completion, compaction, EO hook."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

from core.engine.ports import StatePort
from core.memory.compactor import SemanticCompactor
from core.models import AtomicTask
from core.utils.logging import get_logger

logger = get_logger(__name__)


class RunLoopController:
    def __init__(
        self,
        blackboard: StatePort,
        decompose: Callable[[], Awaitable[None]],
        execute_task: Callable[
            [AtomicTask, dict[str, tuple[AtomicTask, asyncio.Task[None]]]],
            Awaitable[None],
        ],
        compactor: SemanticCompactor,
        ui_callback: Any,
        on_terminate: Callable[[], Awaitable[None]],
    ) -> None:
        self.bb = blackboard
        self._decompose = decompose
        self._execute_task = execute_task
        self.compactor = compactor
        self.ui_callback = ui_callback
        self._on_terminate = on_terminate
        self.running_tasks: dict[str, tuple[AtomicTask, asyncio.Task[None]]] = {}
        self._eo_task: asyncio.Task[None] | None = None

    async def _run_eo_safe(self) -> None:
        try:
            await self._on_terminate()
        except Exception:
            logger.exception("EvolutionaryOptimizer (background) failed")

    async def run(self) -> None:
        decomposition_task = asyncio.create_task(self._decompose())

        async with self.bb.lock:
            if self.bb.state.status != "running":
                self.bb.state.status = "running"

        while self.bb.state.status == "running":
            ready_tasks = await self.bb.get_ready_tasks()

            for task in ready_tasks:
                if task.id not in self.running_tasks:
                    logger.info(f"Scheduling Task: {task.description} (ID: {task.id})")
                    self.running_tasks[task.id] = (
                        task,
                        asyncio.create_task(self._execute_task(task, self.running_tasks)),
                    )

            if self.ui_callback:
                async with self.bb.lock:
                    pending = len(self.bb.state.todo_list)
                    completed = len(self.bb.state.completed_tasks)
                self.ui_callback(
                    "stats",
                    {"running": len(self.running_tasks), "pending": pending, "completed": completed},
                )

            if not self.running_tasks:
                async with self.bb.lock:
                    has_todo = len(self.bb.state.todo_list) > 0

                if decomposition_task.done():
                    async with self.bb.lock:
                        current_status = self.bb.state.status

                    if current_status == "failed":
                        logger.error("Decomposition failed. Loop exiting with failure.")
                        break

                    if not has_todo:
                        logger.info("All tasks completed. Execution finished.")
                        await self.bb.mark_completed()
                        async with self.bb.lock:
                            self.bb.state.shared_memory = await self.compactor.compact(
                                self.bb.state.shared_memory
                            )
                    else:
                        pending_ids = [t.id for t in self.bb.state.todo_list]
                        logger.error(
                            f"TRINITY LOOP FAILED: No ready tasks. Pending IDs: {pending_ids}"
                        )
                        await self.bb.mark_failed()
                    break
                await asyncio.sleep(0.5)
                continue

            try:
                done, _ = await asyncio.wait(
                    [v[1] for v in self.running_tasks.values()] + [decomposition_task],
                    return_when=asyncio.FIRST_COMPLETED,
                    timeout=1.0,
                )
            except asyncio.TimeoutError:
                continue

            for completed in done:
                if completed == decomposition_task:
                    logger.debug("Decomposition task heartbeat/completion.")
                    continue

                matching_tid = [k for k, v in self.running_tasks.items() if v[1] == completed]
                if matching_tid:
                    tid = matching_tid[0]
                    del self.running_tasks[tid]

        if self.bb.state.status in ["completed", "failed"]:
            logger.info("Loop Terminated. Scheduling EO distillation in background...")
            self._eo_task = asyncio.create_task(self._run_eo_safe())
