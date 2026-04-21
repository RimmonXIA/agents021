import json
import sqlite3
from typing import Any

import lancedb
from agno.agent import Agent
from agno.models.deepseek import DeepSeek

from core.agents.runner import run_agent
from core.config import settings
from core.models import EOSkillExtract, Skill
from core.utils.logging import get_logger

logger = get_logger(__name__)


def _eo_on_failed_attempt(agent: Agent, attempt: int, e: Exception) -> None:
    logger.error(f"[EO] Attempt {attempt} failed: {e}")
    if agent.model and hasattr(agent.model, "id") and agent.model.id == "deepseek-reasoner":
        agent.model = DeepSeek(id="deepseek-chat")


class EvolutionaryOptimizer:
    """
    (EO) - Evolutionary Optimizer.
    Extracts successful trajectories and refines them into reusable Skills (SOPs).
    """

    def __init__(self) -> None:
        self.db_path = settings.sqlite_db_path
        self.lancedb_dir = settings.lancedb_dir
        self.reflection_agent = Agent(
            model=DeepSeek(id=settings.eo_model),
            name="ReflectionEngine",
            description="Analyzes trajectories to extract reusable SOPs (Skills).",
            instructions=[
                "You are the Evolutionary Optimizer.",
                "Review the provided execution trajectory of a multi-agent system.",
                "Identify what worked well and what failed.",
                "If the trajectory was successful or contained valuable learning, extract a generalized 'Skill' (SOP).",
                "Output MUST be strict JSON matching the schema.",
                "If the trajectory provides no reusable value, set 'skip' to true.",
            ],
        )

    async def process_session(self, session_id: str) -> None:
        """Processes a session trajectory to distill new skills."""
        logger.info(f"Starting Reflection on session: {session_id}")

        trajectory_text: list[str] = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT step_id, task_json, result_json FROM trajectories WHERE session_id = ? ORDER BY step_id ASC",
                (session_id,),
            )
            rows = cursor.fetchall()

            if not rows:
                logger.warning(f"No trajectory found for session {session_id}")
                return

            for row in rows:
                step_id, t_json, r_json = row
                task = json.loads(t_json)
                result = json.loads(r_json)
                trajectory_text.append(
                    f"Step {step_id}: Task: {task.get('description')} -> "
                    f"Status: {result.get('status')} -> Output: {result.get('output')[:200]}..."
                )

        trajectory_str = "\n".join(trajectory_text)
        logger.info("Analyzing Trajectory for skill distillation...")

        skill_data = await run_agent(
            self.reflection_agent,
            f"Trajectory to analyze:\n{trajectory_str}",
            response_model=EOSkillExtract,
            on_failed_attempt=_eo_on_failed_attempt,
            on_reasoning=lambda rc: logger.debug(f"[EO Reflection Thinking]\n{rc}"),
            return_none_on_failure=True,
        )

        if isinstance(skill_data, EOSkillExtract):
            if skill_data.skip:
                logger.info("Trajectory yielded no new skills. Skipping.")
            else:
                skill = Skill(
                    title=skill_data.title,
                    description=skill_data.description,
                    content_markdown=skill_data.content_markdown,
                    vector_embedding=None,
                )
                logger.info(f"Distilling skill: {skill.title}")
                self._persist_skill(skill)
        else:
            logger.error("Reflection failed to produce valid skill data.")

    def _persist_skill(self, skill: Skill) -> None:
        """Saves the skill to LanceDB for semantic retrieval."""
        db = lancedb.connect(self.lancedb_dir)
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
            table = db.open_table("skills")
            table.add(data)
            logger.info(f"Skill '{skill.title}' added to existing table.")
        except Exception:
            db.create_table("skills", data=data)
            logger.info(f"Skill '{skill.title}' persisted to new table.")
