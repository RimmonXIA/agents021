"""SQLite persistence for execution trajectories (EO input)."""
from __future__ import annotations

import sqlite3
from pathlib import Path


class TrajectoryStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("PRAGMA journal_mode=WAL")
            cursor.execute(
                """
                CREATE TABLE IF NOT EXISTS trajectories (
                    session_id TEXT,
                    step_id INTEGER,
                    task_json TEXT,
                    result_json TEXT,
                    timestamp TEXT
                )
                """
            )
            conn.commit()

    def append(
        self,
        session_id: str,
        step_id: int,
        task_json: str,
        result_json: str,
        timestamp: str,
    ) -> None:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "INSERT INTO trajectories (session_id, step_id, task_json, result_json, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, step_id, task_json, result_json, timestamp),
            )
            conn.commit()

    def fetch_session(self, session_id: str) -> list[tuple[int, str, str]]:
        """Load a session trajectory ordered by step id."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT step_id, task_json, result_json FROM trajectories WHERE session_id = ? ORDER BY step_id ASC",
                (session_id,),
            )
            return [(int(step_id), str(task_json), str(result_json)) for step_id, task_json, result_json in cursor.fetchall()]
