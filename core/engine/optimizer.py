import json

from agno.agent import Agent
from agno.models.deepseek import DeepSeek

from core.agents.runner import run_agent
from core.config import settings
from core.engine.ports import SkillPort, TrajectoryPort
from core.memory.skill_index import SkillIndex
from core.memory.trajectory_store import TrajectoryStore
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

    def __init__(
        self,
        trajectory_store: TrajectoryPort | None = None,
        skill_store: SkillPort | None = None,
    ) -> None:
        self._trajectory_store = trajectory_store or TrajectoryStore(settings.sqlite_db_path)
        self._skill_store = skill_store or SkillIndex(settings.lancedb_dir)
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
        rows = self._trajectory_store.fetch_session(session_id)
        if not rows:
            logger.warning(f"No trajectory found for session {session_id}")
            return

        for step_id, t_json, r_json in rows:
            task = json.loads(t_json)
            result = json.loads(r_json)
            output = str(result.get("output", ""))
            trajectory_text.append(
                f"Step {step_id}: Task: {task.get('description')} -> "
                f"Status: {result.get('status')} -> Output: {output[:200]}..."
            )

        trajectory_str = "\n".join(trajectory_text)
        logger.info("Analyzing Trajectory for skill distillation...")

        run_result = await run_agent(
            self.reflection_agent,
            f"Trajectory to analyze:\n{trajectory_str}",
            response_model=EOSkillExtract,
            on_failed_attempt=_eo_on_failed_attempt,
            on_reasoning=lambda rc: logger.debug(f"[EO Reflection Thinking]\n{rc}"),
        )
        skill_data = run_result.parsed if run_result.success else None

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
                self._skill_store.persist_skill(skill)
        else:
            logger.error("Reflection failed to produce valid skill data.")
