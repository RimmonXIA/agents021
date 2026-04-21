from unittest.mock import patch

import pytest
from agno.models.deepseek import DeepSeek

from core.agents.runner import AgentRunResult
from core.config import settings
from core.engine.optimizer import EvolutionaryOptimizer, _eo_on_failed_attempt
from core.models import EOCandidateList, EOGateResult, EOReviewedSkillList, EOSkillExtract


class _TrajectoryStore:
    def fetch_session(self, session_id: str) -> list[tuple[int, str, str]]:
        del session_id
        return [
            (
                1,
                '{"description":"collect data"}',
                '{"status":"success","output":"large output"}',
            )
        ]

    def fetch_session_detailed(self, session_id: str) -> list[tuple[int, str, str, str]]:
        del session_id
        return [
            (
                1,
                '{"id":"t1","description":"collect data","depends_on":[],"required_keys":[],"context_keys":[]}',
                '{"status":"success","output":"large output","artifacts":{"parent_ids":[],"sibling_ids":[]}}',
                "2026-01-01T00:00:00",
            )
        ]


class _SkillStore:
    def __init__(self) -> None:
        self.persisted_titles: list[str] = []

    def fetch_for_intent(self, original_intent: str) -> list[dict[str, str]]:
        del original_intent
        return []

    def persist_skill(self, skill):  # noqa: ANN001
        self.persisted_titles.append(skill.title)


def test_eo_failed_attempt_falls_back_model() -> None:
    class _Agent:
        def __init__(self) -> None:
            self.model = DeepSeek(id="deepseek-reasoner")

    agent = _Agent()
    _eo_on_failed_attempt(agent, 1, RuntimeError("x"))
    assert isinstance(agent.model, DeepSeek)
    assert agent.model.id == "deepseek-chat"


@pytest.mark.asyncio
async def test_optimizer_persists_skill_when_reflection_succeeds() -> None:
    skill_store = _SkillStore()
    eo = EvolutionaryOptimizer(trajectory_store=_TrajectoryStore(), skill_store=skill_store)
    old_multi_pass = settings.eo_multi_pass
    settings.eo_multi_pass = False
    with patch(
        "core.engine.optimizer.run_agent",
        return_value=AgentRunResult(
            success=True,
            parsed=EOSkillExtract(
                skip=False,
                title="Useful SOP",
                description="When to use it",
                content_markdown="steps",
            ),
        ),
    ):
        await eo.process_session("s1")
    settings.eo_multi_pass = old_multi_pass

    assert skill_store.persisted_titles == ["Useful SOP"]


@pytest.mark.asyncio
async def test_optimizer_multi_pass_persists_only_accepted_skills() -> None:
    skill_store = _SkillStore()
    eo = EvolutionaryOptimizer(trajectory_store=_TrajectoryStore(), skill_store=skill_store)
    with patch(
        "core.engine.optimizer.run_agent",
        side_effect=[
            AgentRunResult(
                success=True,
                parsed=EOCandidateList.model_validate(
                    {
                        "candidates": [
                            {
                                "title": "Validated SOP",
                                "description": "Use when data collection fails intermittently.",
                                "content_markdown": "1. Retry\n2. Fallback",
                                "evidence_step_ids": [1],
                                "tags": ["retry"],
                            }
                        ]
                    }
                ),
            ),
            AgentRunResult(
                success=True,
                parsed=EOReviewedSkillList.model_validate(
                    {
                        "reviewed_candidates": [
                            {
                                "title": "Validated SOP",
                                "description": "Use when data collection fails intermittently.",
                                "content_markdown": "1. Retry\n2. Fallback",
                                "evidence_step_ids": [1],
                                "tags": ["retry"],
                                "faithfulness_score": 0.9,
                                "novelty_score": 0.7,
                                "utility_score": 0.8,
                                "concerns": [],
                            }
                        ]
                    }
                ),
            ),
            AgentRunResult(
                success=True,
                parsed=EOGateResult.model_validate(
                    {
                        "gated_candidates": [
                            {
                                "title": "Validated SOP",
                                "description": "Use when data collection fails intermittently.",
                                "content_markdown": "1. Retry\n2. Fallback",
                                "evidence_step_ids": [1],
                                "tags": ["retry"],
                                "quality_score": 0.88,
                                "decision": "accept",
                                "rationale": "Strong evidence alignment.",
                            },
                            {
                                "title": "Weak SOP",
                                "description": "Not enough evidence.",
                                "content_markdown": "Do something",
                                "evidence_step_ids": [],
                                "tags": [],
                                "quality_score": 0.2,
                                "decision": "reject",
                                "rationale": "Ungrounded.",
                            },
                        ]
                    }
                ),
            ),
        ],
    ):
        await eo.process_session("s1")

    assert skill_store.persisted_titles == ["Validated SOP"]


@pytest.mark.asyncio
async def test_optimizer_hard_gate_requires_evidence_and_quality() -> None:
    skill_store = _SkillStore()
    eo = EvolutionaryOptimizer(trajectory_store=_TrajectoryStore(), skill_store=skill_store)
    old_stage = settings.skill_write_gate_stage
    settings.skill_write_gate_stage = "hard"
    try:
        with patch(
            "core.engine.optimizer.run_agent",
            side_effect=[
                AgentRunResult(
                    success=True,
                    parsed=EOCandidateList.model_validate(
                        {
                            "candidates": [
                                {
                                    "title": "Weak SOP",
                                    "description": "missing evidence",
                                    "content_markdown": "do x",
                                    "evidence_step_ids": [],
                                    "tags": ["weak"],
                                }
                            ]
                        }
                    ),
                ),
                AgentRunResult(
                    success=True,
                    parsed=EOReviewedSkillList.model_validate(
                        {
                            "reviewed_candidates": [
                                {
                                    "title": "Weak SOP",
                                    "description": "missing evidence",
                                    "content_markdown": "do x",
                                    "evidence_step_ids": [],
                                    "tags": ["weak"],
                                    "faithfulness_score": 0.5,
                                    "novelty_score": 0.4,
                                    "utility_score": 0.4,
                                    "concerns": ["no grounding"],
                                }
                            ]
                        }
                    ),
                ),
                AgentRunResult(
                    success=True,
                    parsed=EOGateResult.model_validate(
                        {
                            "gated_candidates": [
                                {
                                    "title": "Weak SOP",
                                    "description": "missing evidence",
                                    "content_markdown": "do x",
                                    "evidence_step_ids": [],
                                    "tags": ["weak"],
                                    "quality_score": 0.2,
                                    "decision": "accept",
                                    "rationale": "accepted by model but should fail hard gate",
                                }
                            ]
                        }
                    ),
                ),
            ],
        ):
            await eo.process_session("s1")
    finally:
        settings.skill_write_gate_stage = old_stage

    assert skill_store.persisted_titles == []
