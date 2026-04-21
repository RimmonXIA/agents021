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
            cursor.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_trajectories_session_step "
                "ON trajectories(session_id, step_id)"
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
                "INSERT OR REPLACE INTO trajectories (session_id, step_id, task_json, result_json, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                (session_id, step_id, task_json, result_json, timestamp),
            )
            conn.commit()

    def append_batch(self, rows: list[tuple[str, int, str, str, str]]) -> None:
        if not rows:
            return
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.executemany(
                "INSERT OR REPLACE INTO trajectories (session_id, step_id, task_json, result_json, timestamp) "
                "VALUES (?, ?, ?, ?, ?)",
                rows,
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

    def fetch_session_detailed(self, session_id: str) -> list[tuple[int, str, str, str]]:
        """Load detailed session trajectory ordered by step id."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT step_id, task_json, result_json, timestamp "
                "FROM trajectories WHERE session_id = ? ORDER BY step_id ASC",
                (session_id,),
            )
            return [
                (int(step_id), str(task_json), str(result_json), str(timestamp))
                for step_id, task_json, result_json, timestamp in cursor.fetchall()
            ]

    def list_sessions(self, limit: int = 50) -> list[str]:
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT session_id FROM trajectories GROUP BY session_id ORDER BY MAX(timestamp) DESC LIMIT ?",
                (limit,),
            )
            return [str(row[0]) for row in cursor.fetchall()]
