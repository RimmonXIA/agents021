import asyncio
import json
import sqlite3
from typing import Any

import lancedb
from agno.agent import Agent
from agno.models.deepseek import DeepSeek

from core.config import settings
from core.models import EOSkillExtract, Skill
from core.utils.logging import get_logger

logger = get_logger(__name__)

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
                "If the trajectory provides no reusable value, set 'skip' to true."
            ]
        )

    async def _safe_run(self, agent: Agent, prompt: str, response_model: type | None = None) -> Any:
        """Helper for elegant agent execution in Optimizer."""
        current_attempt = 0
        max_retries = settings.max_retries
        while current_attempt < max_retries:
            current_attempt += 1
            try:
                response = await asyncio.to_thread(agent.run, prompt, response_model=response_model)
                content = response.content

                if hasattr(response, "reasoning_content") and response.reasoning_content:
                    logger.debug(f"[EO Reflection Thinking]\n{response.reasoning_content}")

                if not content:
                    raise ValueError("Empty response.")
                
                if response_model and isinstance(content, str):
                    from agno.utils.string import parse_response_model_str
                    parsed = parse_response_model_str(content, response_model)
                    if parsed: 
                        return parsed
                    raise ValueError("Parsing failed.")
                return content
            except Exception as e:
                logger.error(f"[EO] Attempt {current_attempt} failed: {e}")
                if agent.model and hasattr(agent.model, "id") and agent.model.id == "deepseek-reasoner":
                    agent.model = DeepSeek(id="deepseek-chat")
        return None

    async def process_session(self, session_id: str) -> None:
        """Processes a session trajectory to distill new skills."""
        logger.info(f"Starting Reflection on session: {session_id}")
        
        trajectory_text = []
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                "SELECT step_id, task_json, result_json FROM trajectories WHERE session_id = ? ORDER BY step_id ASC", 
                (session_id,)
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
        
        skill_data = await self._safe_run(
            self.reflection_agent, 
            f"Trajectory to analyze:\n{trajectory_str}", 
            response_model=EOSkillExtract
        )

        if isinstance(skill_data, EOSkillExtract):
            if skill_data.skip:
                logger.info("Trajectory yielded no new skills. Skipping.")
            else:
                skill = Skill(
                    title=skill_data.title,
                    description=skill_data.description,
                    content_markdown=skill_data.content_markdown,
                    vector_embedding=None
                )
                logger.info(f"Distilling skill: {skill.title}")
                self._persist_skill(skill)
        else:
            logger.error("Reflection failed to produce valid skill data.")

    def _persist_skill(self, skill: Skill) -> None:
        """Saves the skill to LanceDB for semantic retrieval."""
        db = lancedb.connect(self.lancedb_dir)
        data = [{
            "id": skill.id,
            "title": skill.title,
            "description": skill.description,
            "content_markdown": skill.content_markdown,
            "text": f"{skill.title} {skill.description}"
        }]
        try:
            table = db.open_table("skills")
            table.add(data)
            logger.info(f"Skill '{skill.title}' added to existing table.")
        except Exception:
            db.create_table("skills", data=data)
            logger.info(f"Skill '{skill.title}' persisted to new table.")
