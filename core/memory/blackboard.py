import asyncio
import sqlite3
from typing import Any

import lancedb

from core.config import settings
from core.models import AtomicTask, GlobalState, TrajectoryStep
from core.utils.logging import get_logger

logger = get_logger(__name__)

class Blackboard:
    """
    Manages the global state of the IntentOrchestrator's execution loop.
    Persists trajectory and state to SQLite for EO to analyze later.
    """
    def __init__(self, session_id: str, original_intent: str):
        self.state = GlobalState(session_id=session_id, original_intent=original_intent)
        self.db_path = settings.sqlite_db_path
        self.lancedb_dir = settings.lancedb_dir
        self.lock = asyncio.Lock()
        self._init_db()

    def _init_db(self) -> None:
        """Initializes SQLite tables for trajectory persistence and connects to LanceDB."""
        # Ensure directories exist
        from pathlib import Path
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.lancedb_dir).mkdir(parents=True, exist_ok=True)

        logger.info(f"Initializing SQLite memory at {self.db_path}")
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS trajectories (
                    session_id TEXT,
                    step_id INTEGER,
                    task_json TEXT,
                    result_json TEXT,
                    timestamp TEXT
                )
            ''')
            conn.commit()
        
        logger.info(f"Connecting to LanceDB at {self.lancedb_dir}")
        self.vector_db = lancedb.connect(self.lancedb_dir)

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
                # 1. Check depends_on (Static)
                static_met = all(dep_id in completed_ids for dep_id in task.depends_on)
                # 2. Check required_keys (Dynamic)
                dynamic_met = all(key in shared_keys for key in task.required_keys)
                
                if static_met and dynamic_met:
                    ready.append(task)
            
            # Remove ready tasks from TODO
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
                    # Placeholder for advanced merging; default to overwrite for now
                    logger.warning(f"Semantic merge requested for {key} but not yet fully implemented. Overwriting.")
                    self.state.shared_memory[key] = value

    async def get_context(self, keys: list[str], filter_query: str | None = None) -> dict[str, Any]:
        """
        Retrieves context from shared memory. 
        SOTA: Supports basic filtering to reduce token pressure on sub-agents.
        """
        async with self.lock:
            context = {k: self.state.shared_memory.get(k) for k in keys}
            if filter_query:
                # Placeholder for JSONPath-like filtering
                # For now, we just log the optimization intent
                logger.debug(f"Applying context filter: {filter_query}")
            return context

    async def record_step(self, step: TrajectoryStep) -> None:
        """Records a step in memory and persists to SQLite."""
        async with self.lock:
            self.state.trajectory.append(step)
            self.state.completed_tasks.append(step.task)
        
        # SQLite WAL mode handles concurrency for the write
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT INTO trajectories (session_id, step_id, task_json, result_json, timestamp) VALUES (?, ?, ?, ?, ?)',
                (
                    self.state.session_id,
                    step.step_id,
                    step.task.model_dump_json(),
                    step.result.model_dump_json(),
                    step.timestamp.isoformat()
                )
            )
            conn.commit()

    async def mark_completed(self) -> None:
        async with self.lock:
            self.state.status = "completed"

    async def mark_failed(self) -> None:
        async with self.lock:
            self.state.status = "failed"

    def fetch_relevant_skills(self) -> list[dict[str, str]]:
        """Retrieves semantic relevant skills (SOPs) from LanceDB."""
        logger.info(f"Searching for relevant skills for intent: '{self.state.original_intent}'")
        try:
            table = self.vector_db.open_table("skills")
            results = table.search(self.state.original_intent).limit(3).to_list()
            
            skills = []
            for r in results:
                skills.append({
                    "title": r.get("title", ""),
                    "description": r.get("description", ""),
                    "content": r.get("content_markdown", "")
                })
            
            if skills:
                logger.info(f"Found {len(skills)} relevant skills in memory.")
            else:
                logger.info("No relevant skills found.")
            return skills
        except Exception as e:
            logger.debug(f"Skills table not found or error during retrieval: {e}")
            return []

    async def get_full_plan(self) -> Any:
        """Returns the current todo_list wrapped in an IOPlan."""
        from core.models import IOPlan
        async with self.lock:
            # Note: tasks might be empty if decomposition hasn't happened or they were already popped.
            # But decompose_intent adds them before this is usually called.
            return IOPlan(tasks=self.state.todo_list.copy())
