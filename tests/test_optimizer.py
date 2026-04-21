from unittest.mock import patch

import pytest
from agno.models.deepseek import DeepSeek

from core.agents.runner import AgentRunResult
from core.engine.optimizer import EvolutionaryOptimizer, _eo_on_failed_attempt
from core.models import EOSkillExtract


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

    assert skill_store.persisted_titles == ["Useful SOP"]
