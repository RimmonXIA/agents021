"""In-memory session state store for Blackboard."""
from __future__ import annotations

import asyncio
from datetime import datetime
from typing import Any

from core.models import AtomicTask, GlobalState, TrajectoryStep, WorldState
from core.utils.logging import get_logger

logger = get_logger(__name__)


class SessionStateStore:
    """Pure in-memory state manager with locking and scheduling helpers."""

    def __init__(self, session_id: str, original_intent: str = "") -> None:
        now = datetime.now()
        self.state = GlobalState(
            session_id=session_id,
            original_intent=original_intent,
            world_state=WorldState(
                current_date=now.strftime("%Y-%m-%d"),
                current_time=now.strftime("%H:%M:%S"),
                day_of_week=now.strftime("%A"),
            ),
        )
        self.lock = asyncio.Lock()

    async def add_todo(self, task: AtomicTask) -> None:
        async with self.lock:
            self.state.todo_list.append(task)

    async def pop_todo(self) -> AtomicTask | None:
        async with self.lock:
            if not self.state.todo_list:
                return None
            return self.state.todo_list.pop(0)

    async def get_ready_tasks(self) -> list[AtomicTask]:
        ready = []
        async with self.lock:
            completed_ids = {t.id for t in self.state.completed_tasks}
            shared_keys = set(self.state.shared_memory.keys())

            for task in self.state.todo_list:
                static_met = all(dep_id in completed_ids for dep_id in task.depends_on)
                dynamic_met = all(key in shared_keys for key in task.required_keys)
                if static_met and dynamic_met:
                    ready.append(task)

            for task in ready:
                self.state.todo_list.remove(task)
        return ready

    async def update_context(self, key: str, value: Any) -> None:
        async with self.lock:
            logger.debug("Updating shared memory: %s", key)
            self.state.shared_memory[key] = value

    async def apply_changeset(self, task: AtomicTask, changes: dict[str, Any]) -> None:
        async with self.lock:
            for key, value in changes.items():
                if task.branch_policy == "overwrite" or key not in self.state.shared_memory:
                    self.state.shared_memory[key] = value
                elif task.branch_policy == "append":
                    existing = self.state.shared_memory.get(key)
                    if isinstance(existing, list) and isinstance(value, list):
                        existing.extend(value)
                    elif isinstance(existing, str) and isinstance(value, str):
                        self.state.shared_memory[key] = f"{existing}\n\n{value}"
                    else:
                        self.state.shared_memory[key] = value
                elif task.branch_policy == "semantic_merge":
                    logger.warning(
                        "Semantic merge requested for %s but not yet fully implemented. Overwriting.",
                        key,
                    )
                    self.state.shared_memory[key] = value

    async def get_context(self, keys: list[str], filter_query: str | None = None) -> dict[str, Any]:
        async with self.lock:
            context = {k: self.state.shared_memory.get(k) for k in keys}
            if filter_query:
                logger.debug("Applying context filter: %s", filter_query)
            return context

    def world_context_patch(self) -> dict[str, Any]:
        ws = self.state.world_state
        if not ws:
            return {}
        return {
            "current_date": ws.current_date,
            "current_time": ws.current_time,
            "day_of_week": ws.day_of_week,
            "system_era": ws.system_era,
            "knowledge_cutoff": ws.knowledge_cutoff,
        }

    async def record_step_in_memory(self, step: TrajectoryStep) -> None:
        async with self.lock:
            self.state.trajectory.append(step)
            self.state.completed_tasks.append(step.task)

    async def mark_completed(self) -> None:
        async with self.lock:
            self.state.status = "completed"

    async def mark_failed(self) -> None:
        async with self.lock:
            self.state.status = "failed"

    async def sync_world_state(self) -> None:
        now = datetime.now()
        async with self.lock:
            if self.state.world_state:
                self.state.world_state.current_date = now.strftime("%Y-%m-%d")
                self.state.world_state.current_time = now.strftime("%H:%M:%S")
                self.state.world_state.day_of_week = now.strftime("%A")

    async def get_full_plan(self) -> Any:
        from core.models import IOPlan

        async with self.lock:
            return IOPlan(tasks=self.state.todo_list.copy())
