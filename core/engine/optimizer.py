import json
from collections.abc import Iterable

from agno.agent import Agent
from agno.models.deepseek import DeepSeek

from core.agents.runner import run_agent
from core.config import settings
from core.engine.ports import SkillPort, TrajectoryPort
from core.memory.skill_index import SkillIndex
from core.memory.trajectory_store import TrajectoryStore
from core.models import (
    EOCandidateList,
    EOGateResult,
    EOReviewedSkillList,
    EOSkillExtract,
    ReflectionPack,
    ReflectionPackStep,
    Skill,
)
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
        self._last_gate_distribution: dict[str, int] = {"accept": 0, "revise": 0, "reject": 0}
        self.reflection_agent = Agent(
            model=DeepSeek(id=settings.eo_model),
            name="EOCandidateEngine",
            description="Analyzes reflection packs and proposes candidate reusable SOPs.",
            instructions=[
                "You are the Evolutionary Optimizer.",
                "Review the provided reflection pack from a multi-agent run.",
                "Extract practical candidate SOPs grounded in evidence.",
                "Only include candidates with clear evidence steps.",
                "Output MUST be strict JSON matching the schema.",
            ],
        )
        self.critic_agent = Agent(
            model=DeepSeek(id=settings.eo_model),
            name="EOCriticEngine",
            description="Critiques candidate skills for faithfulness, novelty, and utility.",
            instructions=[
                "You are the EO quality critic.",
                "Review candidate skills against evidence from the reflection pack.",
                "Score each candidate for faithfulness, novelty, and utility in [0, 1].",
                "Call out concerns and reject hallucinated claims.",
                "Output MUST be strict JSON matching the schema.",
            ],
        )
        self.gate_agent = Agent(
            model=DeepSeek(id=settings.eo_model),
            name="EOGateEngine",
            description="Applies final quality gate to reviewed skills.",
            instructions=[
                "You are the EO quality gate.",
                "Decide for each reviewed skill: accept, revise, or reject.",
                "Set quality_score in [0, 1] and justify rationale.",
                "Reject skills that are weakly grounded or low utility.",
                "Output MUST be strict JSON matching the schema.",
            ],
        )

    def _chunk_output(self, text: str) -> list[str]:
        max_chars = max(100, settings.eo_max_output_chars)
        if len(text) <= max_chars:
            return [text] if text else []
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    def _parse_list(self, raw: object) -> list[str]:
        if isinstance(raw, list):
            return [str(x) for x in raw if str(x)]
        return []

    def _build_reflection_pack(
        self,
        session_id: str,
        original_intent: str,
        rows: Iterable[tuple[int, str, str, str]],
    ) -> ReflectionPack:
        steps: list[ReflectionPackStep] = []
        success_steps = 0
        failed_steps = 0
        parallelism_observed = False

        for step_id, t_json, r_json, timestamp in rows:
            task = json.loads(t_json)
            result = json.loads(r_json)
            status = str(result.get("status", "unknown"))
            if status == "success":
                success_steps += 1
            else:
                failed_steps += 1
            output = str(result.get("output", ""))
            output_chunks = self._chunk_output(output)
            artifacts = result.get("artifacts", {})
            if not isinstance(artifacts, dict):
                artifacts = {}

            parent_ids = self._parse_list(artifacts.get("parent_ids", task.get("depends_on", [])))
            sibling_ids = self._parse_list(artifacts.get("sibling_ids", []))
            if sibling_ids:
                parallelism_observed = True

            steps.append(
                ReflectionPackStep(
                    step_id=step_id,
                    task_id=str(task.get("id", "")),
                    task_description=str(task.get("description", "")),
                    status=status,
                    context_keys=self._parse_list(task.get("context_keys", [])),
                    required_keys=self._parse_list(task.get("required_keys", [])),
                    parent_ids=parent_ids,
                    sibling_ids=sibling_ids,
                    output_excerpt=output[:400],
                    output_chunks=output_chunks,
                    artifacts=artifacts,
                    timestamp=timestamp,
                )
            )

        return ReflectionPack(
            session_id=session_id,
            original_intent=original_intent,
            total_steps=len(steps),
            success_steps=success_steps,
            failed_steps=failed_steps,
            parallelism_observed=parallelism_observed,
            steps=steps,
        )

    async def _run_single_pass(self, pack: ReflectionPack) -> list[Skill]:
        run_result = await run_agent(
            self.reflection_agent,
            f"Reflection pack JSON:\n{pack.model_dump_json(indent=2)}",
            response_model=EOSkillExtract,
            on_failed_attempt=_eo_on_failed_attempt,
            on_reasoning=lambda rc: logger.debug(f"[EO Reflection Thinking]\n{rc}"),
        )
        skill_data = run_result.parsed if run_result.success else None
        if isinstance(skill_data, EOSkillExtract) and not skill_data.skip:
            return [
                Skill(
                    title=skill_data.title,
                    description=skill_data.description,
                    content_markdown=skill_data.content_markdown,
                    source_session_ids=[pack.session_id],
                    quality_score=0.5,
                )
            ]
        return []

    async def _run_multi_pass(self, pack: ReflectionPack) -> list[Skill]:
        candidate_result = await run_agent(
            self.reflection_agent,
            f"Reflection pack JSON:\n{pack.model_dump_json(indent=2)}",
            response_model=EOCandidateList,
            on_failed_attempt=_eo_on_failed_attempt,
            on_reasoning=lambda rc: logger.debug(f"[EO Candidate Thinking]\n{rc}"),
        )
        candidate_data = candidate_result.parsed if candidate_result.success else None
        if not isinstance(candidate_data, EOCandidateList) or not candidate_data.candidates:
            return []

        critic_result = await run_agent(
            self.critic_agent,
            (
                "Reflection pack JSON:\n"
                f"{pack.model_dump_json(indent=2)}\n\n"
                "Candidate JSON:\n"
                f"{candidate_data.model_dump_json(indent=2)}"
            ),
            response_model=EOReviewedSkillList,
            on_failed_attempt=_eo_on_failed_attempt,
            on_reasoning=lambda rc: logger.debug(f"[EO Critic Thinking]\n{rc}"),
        )
        reviewed_data = critic_result.parsed if critic_result.success else None
        if not isinstance(reviewed_data, EOReviewedSkillList) or not reviewed_data.reviewed_candidates:
            return []

        gate_result = await run_agent(
            self.gate_agent,
            (
                "Reviewed candidates JSON:\n"
                f"{reviewed_data.model_dump_json(indent=2)}\n\n"
                f"Minimum quality score: {settings.eo_min_quality_score}"
            ),
            response_model=EOGateResult,
            on_failed_attempt=_eo_on_failed_attempt,
            on_reasoning=lambda rc: logger.debug(f"[EO Gate Thinking]\n{rc}"),
        )
        gated_data = gate_result.parsed if gate_result.success else None
        if not isinstance(gated_data, EOGateResult):
            return []
        self._last_gate_distribution = {"accept": 0, "revise": 0, "reject": 0}
        for cand in gated_data.gated_candidates:
            if cand.decision in self._last_gate_distribution:
                self._last_gate_distribution[cand.decision] += 1
        logger.info("EO gate distribution: %s", self._last_gate_distribution)

        accepted: list[Skill] = []
        for cand in gated_data.gated_candidates:
            has_evidence = bool(cand.evidence_step_ids)
            meets_quality = cand.quality_score >= settings.eo_min_quality_score
            eligible = has_evidence and meets_quality
            stage = settings.normalized_skill_write_gate_stage
            if settings.eo_quality_gate:
                if cand.decision != "accept":
                    continue
                if cand.quality_score < settings.eo_min_quality_score:
                    continue
            if stage == "hard" and not eligible:
                continue
            status = "active"
            gate_decision = "accept" if eligible else "observe"
            if stage == "soft" and not eligible:
                status = "deprecated"
                gate_decision = "revise"
            accepted.append(
                Skill(
                    title=cand.title,
                    description=cand.description,
                    content_markdown=cand.content_markdown,
                    tags=cand.tags,
                    source_session_ids=[pack.session_id],
                    evidence_step_ids=cand.evidence_step_ids,
                    quality_score=cand.quality_score,
                    status=status,
                    gate_decision=gate_decision,
                    gate_rationale=cand.rationale,
                    provenance={
                        "session_id": pack.session_id,
                        "original_intent": pack.original_intent,
                        "eo_stage": stage,
                        "has_evidence": has_evidence,
                        "meets_quality": meets_quality,
                    },
                )
            )
        return accepted

    @property
    def last_gate_distribution(self) -> dict[str, int]:
        return dict(self._last_gate_distribution)

    async def process_session(self, session_id: str, original_intent: str = "") -> None:
        """Processes a session trajectory to distill new skills."""
        logger.info(f"Starting Reflection on session: {session_id}")

        rows = self._trajectory_store.fetch_session_detailed(session_id)
        if not rows:
            logger.warning(f"No trajectory found for session {session_id}")
            return

        pack = self._build_reflection_pack(session_id, original_intent, rows)
        logger.info("Analyzing ReflectionPack for skill distillation...")

        if settings.eo_multi_pass:
            skills = await self._run_multi_pass(pack)
        else:
            skills = await self._run_single_pass(pack)

        if not skills:
            logger.info("Trajectory yielded no accepted skills. Skipping.")
            return

        for skill in skills:
            logger.info("Distilling skill: %s", skill.title)
            self._skill_store.persist_skill(skill)
