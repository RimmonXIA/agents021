import asyncio

from core.memory.skill_index import SkillIndex
from core.memory.state_store import SessionStateStore
from core.memory.trajectory_store import TrajectoryStore
from core.models import AtomicTask, GlobalState, TrajectoryStep
from core.utils.logging import get_logger

logger = get_logger(__name__)


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
        self._trajectory_queue: asyncio.Queue[tuple[str, int, str, str, str]] = asyncio.Queue()
        self._trajectory_worker_task: asyncio.Task[None] | None = None
        self._trajectory_worker_lock = asyncio.Lock()
        self._memory_tiers: dict[str, dict[str, object]] = {"hot": {}, "warm": {}, "cold": {}}
        self._memory_tier_order: list[str] = []

    async def _ensure_trajectory_worker(self) -> None:
        if self._trajectory_worker_task and not self._trajectory_worker_task.done():
            return
        async with self._trajectory_worker_lock:
            if self._trajectory_worker_task and not self._trajectory_worker_task.done():
                return
            self._trajectory_worker_task = asyncio.create_task(self._trajectory_worker())

    async def _trajectory_worker(self) -> None:
        batch: list[tuple[str, int, str, str, str]] = []
        try:
            while True:
                item = await self._trajectory_queue.get()
                batch.append(item)
                while len(batch) < 32:
                    try:
                        batch.append(self._trajectory_queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                await asyncio.to_thread(self._trajectory_store.append_batch, batch.copy())
                for _ in batch:
                    self._trajectory_queue.task_done()
                batch.clear()
        except asyncio.CancelledError:
            if batch:
                try:
                    await asyncio.to_thread(self._trajectory_store.append_batch, batch.copy())
                finally:
                    for _ in batch:
                        self._trajectory_queue.task_done()
            raise

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
        from core.config import settings

        await self._state_store.update_context(key, value)
        if not settings.memory_tiering_enabled:
            return
        tier = self._route_memory_tier(key, value)
        self._memory_tiers[tier][key] = value
        if tier == "hot":
            if key in self._memory_tier_order:
                self._memory_tier_order.remove(key)
            self._memory_tier_order.append(key)
            while len(self._memory_tier_order) > settings.memory_hot_max_items:
                evicted_key = self._memory_tier_order.pop(0)
                evicted_value = self._memory_tiers["hot"].pop(evicted_key, None)
                if evicted_value is not None:
                    self._memory_tiers["warm"][evicted_key] = evicted_value

    async def apply_changeset(self, task: AtomicTask, changes: dict[str, object]) -> None:
        await self._state_store.apply_changeset(task, changes)

    async def get_context(self, keys: list[str], filter_query: str | None = None) -> dict[str, object]:
        return await self._state_store.get_context(keys, filter_query)

    def world_context_patch(self) -> dict[str, object]:
        return self._state_store.world_context_patch()

    async def record_step(self, step: TrajectoryStep) -> None:
        await self._state_store.record_step_in_memory(step)
        await self._ensure_trajectory_worker()
        await self._trajectory_queue.put(
            (
                self.state.session_id,
                step.step_id,
                step.task.model_dump_json(),
                step.result.model_dump_json(),
                step.timestamp.isoformat(),
            )
        )

    async def flush_trajectory(self) -> None:
        await self._trajectory_queue.join()
        task = self._trajectory_worker_task
        if task and task.done():
            exc = task.exception()
            if exc:
                raise exc

    async def mark_completed(self) -> None:
        await self._state_store.mark_completed()

    async def mark_failed(self) -> None:
        await self._state_store.mark_failed()

    async def sync_world_state(self) -> None:
        await self._state_store.sync_world_state()

    def fetch_relevant_skills(self) -> list[dict[str, str]]:
        from core.config import settings

        if settings.memory_tiering_enabled:
            return self._skill_index.retrieve_reflect_answer(
                self.state.original_intent,
                stage=settings.normalized_memory_router_stage,
            )
        return self._skill_index.fetch_for_intent(self.state.original_intent)

    @staticmethod
    def _route_memory_tier(key: str, value: object) -> str:
        normalized_key = key.lower()
        if normalized_key.startswith("task_") or normalized_key.endswith("_result"):
            return "hot"
        if isinstance(value, str) and len(value) > 512:
            return "cold"
        if isinstance(value, (dict, list)):
            return "warm"
        return "hot"

    def memory_tiers_snapshot(self) -> dict[str, dict[str, object]]:
        return {
            "hot": dict(self._memory_tiers["hot"]),
            "warm": dict(self._memory_tiers["warm"]),
            "cold": dict(self._memory_tiers["cold"]),
        }

    def list_sessions(self, limit: int = 50) -> list[str]:
        return self._trajectory_store.list_sessions(limit=limit)

    def trajectory_port(self) -> TrajectoryStore | None:
        return self._trajectory_store

    def skill_port(self) -> SkillIndex | None:
        return self._skill_index

    async def get_full_plan(self):
        return await self._state_store.get_full_plan()
