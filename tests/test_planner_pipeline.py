import pytest

from core.engine.planner_pipeline import PlannerPipeline
from core.memory.blackboard import Blackboard


class _Chunk:
    def __init__(self, content: str, reasoning: str = "") -> None:
        self.content = content
        self.reasoning_content = reasoning


class _Planner:
    def __init__(self, chunks: list[_Chunk]) -> None:
        self._chunks = chunks

    async def arun(self, prompt: str, stream: bool = False):  # noqa: ANN001
        del prompt, stream
        for chunk in self._chunks:
            yield chunk


@pytest.mark.asyncio
async def test_streaming_planner_parses_partial_json_chunks(temp_db: str, temp_lancedb: str) -> None:
    bb = Blackboard(session_id="test_partial_json", original_intent="plan this")
    planner = _Planner(
        [
            _Chunk('{"tasks": ['),
            _Chunk(
                '{"id":"t1","description":"d","required_capabilities":["search"],'
                '"context_keys":[],"depends_on":[],"required_keys":[],"expected_output":"o"}'
            ),
            _Chunk("]}"),
        ]
    )
    pipeline = PlannerPipeline(bb, planner)  # type: ignore[arg-type]

    await pipeline.decompose_intent()

    plan = await bb.get_full_plan()
    assert len(plan.tasks) == 1
    assert plan.tasks[0].id == "t1"


@pytest.mark.asyncio
async def test_streaming_planner_keeps_running_when_chunk_is_malformed(
    temp_db: str, temp_lancedb: str
) -> None:
    bb = Blackboard(session_id="test_bad_chunk", original_intent="plan this")
    planner = _Planner(
        [
            _Chunk('{"tasks": ['),
            _Chunk("{not valid json"),
            _Chunk(
                ',{"id":"t2","description":"d2","required_capabilities":["search"],'
                '"context_keys":[],"depends_on":[],"required_keys":[],"expected_output":"o2"}'
            ),
            _Chunk("]}"),
        ]
    )
    pipeline = PlannerPipeline(bb, planner)  # type: ignore[arg-type]

    await pipeline.decompose_intent()

    plan = await bb.get_full_plan()
    assert len(plan.tasks) == 0
    assert bb.state.status == "running"
