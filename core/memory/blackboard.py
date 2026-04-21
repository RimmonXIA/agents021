import asyncio
from typing import Any

from core.config import settings
from core.memory.skill_index import SkillIndex
from core.memory.trajectory_store import TrajectoryStore
from core.models import AtomicTask, GlobalState, TrajectoryStep
from core.utils.logging import get_logger

logger = get_logger(__name__)


class Blackboard:
    """
    Manages the global state of the IntentOrchestrator's execution loop.
    Persists trajectory and state to SQLite for EO to analyze later.
    """

    def __init__(self, session_id: str, original_intent: str = "") -> None:
        from datetime import datetime

        now = datetime.now()
        world_state = {
            "current_date": now.strftime("%Y-%m-%d"),
            "current_time": now.strftime("%H:%M:%S"),
            "day_of_week": now.strftime("%A"),
        }
        from core.models import WorldState as WorldStateModel

        self.state = GlobalState(
            session_id=session_id,
            original_intent=original_intent,
            world_state=WorldStateModel(**world_state),
        )
        self.db_path = settings.sqlite_db_path
        self.lancedb_dir = settings.lancedb_dir
        self.lock = asyncio.Lock()
        self._trajectory_store = TrajectoryStore(self.db_path)
        self._skill_index = SkillIndex(self.lancedb_dir)

    async def add_todo(self, task: AtomicTask) -> None:
        async with self.lock:
            self.state.todo_list.append(task)

    async def pop_todo(self) -> AtomicTask | None:
        async with self.lock:
            if not self.state.todo_list:
                return None
            return self.state.todo_list.pop(0)

    async def get_ready_tasks(self) -> list[AtomicTask]:
        """Identifies tasks that meet both static and dynamic dependencies."""
        ready = []
        async with self.lock:
            completed_ids = {t.id for t in self.state.completed_tasks}
            shared_keys = set(self.state.shared_memory.keys())

            for task in self.state.todo_list:
                static_met = all(dep_id in completed_ids for dep_id in task.depends_on)
                dynamic_met = all(key in shared_keys for key in task.required_keys)

                if static_met and dynamic_met:
                    ready.append(task)

            for t in ready:
                self.state.todo_list.remove(t)
        return ready

    async def update_context(self, key: str, value: Any) -> None:
        async with self.lock:
            logger.debug(f"Updating shared memory: {key}")
            self.state.shared_memory[key] = value

    async def apply_changeset(self, task: AtomicTask, changes: dict[str, Any]) -> None:
        """MergeGate: Applies updates from an isolated workspace using the task's policy."""
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
        """
        Retrieves context from shared memory.
        SOTA: Supports basic filtering to reduce token pressure on sub-agents.
        """
        async with self.lock:
            context = {k: self.state.shared_memory.get(k) for k in keys}
            if filter_query:
                logger.debug(f"Applying context filter: {filter_query}")
            return context

    def world_context_patch(self) -> dict[str, Any]:
        """Authoritative clock/calendar fields from world_state (not shared_memory)."""
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

    async def record_step(self, step: TrajectoryStep) -> None:
        """Records a step in memory and persists to SQLite."""
        async with self.lock:
            self.state.trajectory.append(step)
            self.state.completed_tasks.append(step.task)

        self._trajectory_store.append(
            self.state.session_id,
            step.step_id,
            step.task.model_dump_json(),
            step.result.model_dump_json(),
            step.timestamp.isoformat(),
        )

    async def mark_completed(self) -> None:
        async with self.lock:
            self.state.status = "completed"

    async def mark_failed(self) -> None:
        async with self.lock:
            self.state.status = "failed"

    async def sync_world_state(self) -> None:
        """Syncs the world state with current system time."""
        from datetime import datetime

        now = datetime.now()
        async with self.lock:
            if self.state.world_state:
                self.state.world_state.current_date = now.strftime("%Y-%m-%d")
                self.state.world_state.current_time = now.strftime("%H:%M:%S")
                self.state.world_state.day_of_week = now.strftime("%A")

    def fetch_relevant_skills(self) -> list[dict[str, str]]:
        """Retrieves semantic relevant skills (SOPs) from LanceDB."""
        return self._skill_index.fetch_for_intent(self.state.original_intent)

    async def get_full_plan(self) -> Any:
        """Returns the current todo_list wrapped in an IOPlan."""
        from core.models import IOPlan

        async with self.lock:
            return IOPlan(tasks=self.state.todo_list.copy())
