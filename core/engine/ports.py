"""Runtime boundary protocols for Trinity orchestration."""
from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import Any, Protocol

from core.models import AtomicTask, GlobalState, Skill, TrajectoryStep


class StatePort(Protocol):
    """Mutable session-state boundary used by orchestration components."""

    state: GlobalState
    lock: asyncio.Lock

    async def add_todo(self, task: AtomicTask) -> None: ...
    async def get_ready_tasks(self) -> list[AtomicTask]: ...
    async def get_context(self, keys: list[str], filter_query: str | None = None) -> dict[str, Any]: ...
    async def update_context(self, key: str, value: Any) -> None: ...
    async def apply_changeset(self, task: AtomicTask, changes: dict[str, Any]) -> None: ...
    async def record_step(self, step: TrajectoryStep) -> None: ...
    async def mark_completed(self) -> None: ...
    async def mark_failed(self) -> None: ...
    async def sync_world_state(self) -> None: ...
    def world_context_patch(self) -> dict[str, Any]: ...
    async def get_full_plan(self) -> Any: ...
    def fetch_relevant_skills(self) -> list[dict[str, str]]: ...


class TrajectoryPort(Protocol):
    """Persistence boundary for execution trajectories."""

    def append(
        self,
        session_id: str,
        step_id: int,
        task_json: str,
        result_json: str,
        timestamp: str,
    ) -> None: ...

    def fetch_session(self, session_id: str) -> list[tuple[int, str, str]]: ...


class SkillPort(Protocol):
    """Persistence boundary for skill retrieval and storage."""

    def fetch_for_intent(self, original_intent: str) -> list[dict[str, str]]: ...
    def persist_skill(self, skill: Skill) -> None: ...


class EvolutionPort(Protocol):
    """Boundary for EO trigger behavior."""

    async def process_session(self, session_id: str) -> None: ...


class PlannerPort(Protocol):
    """Boundary for decomposition behavior."""

    async def decompose_intent(self) -> None: ...


class TaskExecutionPort(Protocol):
    """Boundary for sub-task execution behavior."""

    async def execute(
        self,
        task: AtomicTask,
        running_tasks_map: dict[str, tuple[AtomicTask, asyncio.Task[None]]],
    ) -> None: ...


OnTerminateHook = Awaitable[None]
