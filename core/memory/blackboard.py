from core.memory.skill_index import SkillIndex
from core.memory.state_store import SessionStateStore
from core.memory.trajectory_store import TrajectoryStore
from core.models import AtomicTask, GlobalState, TrajectoryStep


class Blackboard:
    """
    Manages the global state of the IntentOrchestrator's execution loop.
    Persists trajectory and state to SQLite for EO to analyze later.
    """

    def __init__(
        self,
        session_id: str,
        original_intent: str = "",
        *,
        state_store: SessionStateStore | None = None,
        trajectory_store: TrajectoryStore | None = None,
        skill_index: SkillIndex | None = None,
    ) -> None:
        from core.config import settings

        self._state_store = state_store or SessionStateStore(session_id, original_intent)
        self._trajectory_store = trajectory_store or TrajectoryStore(settings.sqlite_db_path)
        self._skill_index = skill_index or SkillIndex(settings.lancedb_dir)

    @property
    def state(self) -> GlobalState:
        return self._state_store.state

    @property
    def lock(self):
        return self._state_store.lock

    @property
    def trajectory_store(self) -> TrajectoryStore:
        return self._trajectory_store

    @property
    def skill_index(self) -> SkillIndex:
        return self._skill_index

    async def add_todo(self, task: AtomicTask) -> None:
        await self._state_store.add_todo(task)

    async def pop_todo(self) -> AtomicTask | None:
        return await self._state_store.pop_todo()

    async def get_ready_tasks(self) -> list[AtomicTask]:
        return await self._state_store.get_ready_tasks()

    async def update_context(self, key: str, value: object) -> None:
        await self._state_store.update_context(key, value)

    async def apply_changeset(self, task: AtomicTask, changes: dict[str, object]) -> None:
        await self._state_store.apply_changeset(task, changes)

    async def get_context(self, keys: list[str], filter_query: str | None = None) -> dict[str, object]:
        return await self._state_store.get_context(keys, filter_query)

    def world_context_patch(self) -> dict[str, object]:
        return self._state_store.world_context_patch()

    async def record_step(self, step: TrajectoryStep) -> None:
        await self._state_store.record_step_in_memory(step)
        self._trajectory_store.append(
            self.state.session_id,
            step.step_id,
            step.task.model_dump_json(),
            step.result.model_dump_json(),
            step.timestamp.isoformat(),
        )

    async def mark_completed(self) -> None:
        await self._state_store.mark_completed()

    async def mark_failed(self) -> None:
        await self._state_store.mark_failed()

    async def sync_world_state(self) -> None:
        await self._state_store.sync_world_state()

    def fetch_relevant_skills(self) -> list[dict[str, str]]:
        return self._skill_index.fetch_for_intent(self.state.original_intent)

    async def get_full_plan(self):
        return await self._state_store.get_full_plan()
