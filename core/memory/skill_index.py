"""LanceDB-backed skill retrieval for planner context."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import lancedb

from core.models import Skill
from core.utils.logging import get_logger

logger = get_logger(__name__)


class SkillIndex:
    """Semantic search over distilled skills (SOPs) for injection into planner prompts."""

    def __init__(self, lancedb_dir: str) -> None:
        Path(lancedb_dir).mkdir(parents=True, exist_ok=True)
        self._lancedb_dir = lancedb_dir
        self.vector_db: Any = lancedb.connect(lancedb_dir)

    def fetch_for_intent(self, original_intent: str) -> list[dict[str, str]]:
        logger.info(f"Searching for relevant skills for intent: '{original_intent}'")
        try:
            table = self.vector_db.open_table("skills")
            results = table.search(original_intent).limit(3).to_list()

            skills: list[dict[str, str]] = []
            for r in results:
                skills.append(
                    {
                        "title": r.get("title", ""),
                        "description": r.get("description", ""),
                        "content": r.get("content_markdown", ""),
                    }
                )

            if skills:
                logger.info(f"Found {len(skills)} relevant skills in memory.")
            else:
                logger.info("No relevant skills found.")
            return skills
        except Exception as e:
            logger.debug(f"Skills table not found or error during retrieval: {e}")
            return []

    def persist_skill(self, skill: Skill) -> None:
        """Persist a distilled skill for future semantic retrieval."""
        data = [
            {
                "id": skill.id,
                "title": skill.title,
                "description": skill.description,
                "content_markdown": skill.content_markdown,
                "text": f"{skill.title} {skill.description}",
            }
        ]
        try:
            table = self.vector_db.open_table("skills")
            table.add(data)
            logger.info("Skill '%s' added to existing table.", skill.title)
        except Exception:
            self.vector_db.create_table("skills", data=data)
            logger.info("Skill '%s' persisted to new table.", skill.title)
