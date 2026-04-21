import sqlite3

import pytest

from core.memory.blackboard import Blackboard
from core.config import settings
from core.models import AtomicTask, SubAgentResult, TrajectoryStep


@pytest.mark.asyncio
async def test_blackboard_init(temp_db: str, temp_lancedb: str) -> None:
    """Test that Blackboard initializes correctly and creates database tables."""
    Blackboard(session_id="test_session", original_intent="Test Intent")
    
    # Verify SQLite table creation
    with sqlite3.connect(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='trajectories'")
        assert cursor.fetchone() is not None

@pytest.mark.asyncio
async def test_blackboard_todo_management(temp_db: str, temp_lancedb: str) -> None:
    """Test TODO list operations (add/pop)."""
    bb = Blackboard(session_id="test_session", original_intent="Test Intent")
    task = AtomicTask(
        id="t1", 
        description="d", 
        required_capabilities=["search"], 
        expected_output="o"
    )
    
    await bb.add_todo(task)
    assert len(bb.state.todo_list) == 1
    
    popped = await bb.pop_todo()
    assert popped is not None
    assert popped.id == "t1"
    assert len(bb.state.todo_list) == 0

@pytest.mark.asyncio
async def test_blackboard_persistence(temp_db: str, temp_lancedb: str) -> None:
    """Test that recording a step persists it to the SQLite database."""
    bb = Blackboard(session_id="test_session", original_intent="Test Intent")
    task = AtomicTask(id="t1", description="d", required_capabilities=["search"], expected_output="o")
    result = SubAgentResult(task_id="t1", status="success", output="done")
    step = TrajectoryStep(step_id=1, task=task, result=result)
    
    await bb.record_step(step)
    await bb.flush_trajectory()
    
    # Verify in SQLite
    with sqlite3.connect(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT step_id, task_json FROM trajectories WHERE session_id='test_session'")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == 1
        assert '"id":"t1"' in row[1]


@pytest.mark.asyncio
async def test_blackboard_flush_persists_all_steps_in_order(temp_db: str, temp_lancedb: str) -> None:
    bb = Blackboard(session_id="ordered_session", original_intent="ordering")
    task = AtomicTask(id="t-order", description="d", required_capabilities=["search"], expected_output="o")

    for step_id in range(1, 6):
        result = SubAgentResult(task_id="t-order", status="success", output=f"done-{step_id}")
        await bb.record_step(TrajectoryStep(step_id=step_id, task=task, result=result))

    await bb.flush_trajectory()

    with sqlite3.connect(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT step_id FROM trajectories WHERE session_id='ordered_session' ORDER BY step_id ASC"
        )
        rows = [int(r[0]) for r in cursor.fetchall()]
    assert rows == [1, 2, 3, 4, 5]


@pytest.mark.asyncio
async def test_blackboard_list_sessions_returns_recent_ids(temp_db: str, temp_lancedb: str) -> None:
    bb = Blackboard(session_id="session-alpha", original_intent="a")
    task = AtomicTask(id="t1", description="d", required_capabilities=["search"], expected_output="o")
    await bb.record_step(TrajectoryStep(step_id=1, task=task, result=SubAgentResult(task_id="t1", status="success", output="ok")))
    await bb.flush_trajectory()

    bb2 = Blackboard(session_id="session-beta", original_intent="b")
    await bb2.record_step(TrajectoryStep(step_id=1, task=task, result=SubAgentResult(task_id="t1", status="success", output="ok")))
    await bb2.flush_trajectory()

    sessions = bb2.list_sessions(limit=10)
    assert "session-alpha" in sessions
    assert "session-beta" in sessions


@pytest.mark.asyncio
async def test_blackboard_detailed_trajectory_contains_timestamp(temp_db: str, temp_lancedb: str) -> None:
    bb = Blackboard(session_id="session-detail", original_intent="detail")
    task = AtomicTask(id="t-detail", description="d", required_capabilities=["search"], expected_output="o")
    await bb.record_step(
        TrajectoryStep(
            step_id=1,
            task=task,
            result=SubAgentResult(task_id="t-detail", status="success", output="ok"),
        )
    )
    await bb.flush_trajectory()

    rows = bb.trajectory_store.fetch_session_detailed("session-detail")
    assert len(rows) == 1
    assert rows[0][0] == 1
    assert rows[0][3]


@pytest.mark.asyncio
async def test_blackboard_memory_tiers_route_and_bound_hot(temp_db: str, temp_lancedb: str) -> None:
    old_enabled = settings.memory_tiering_enabled
    old_hot_max = settings.memory_hot_max_items
    settings.memory_tiering_enabled = True
    settings.memory_hot_max_items = 1
    try:
        bb = Blackboard(session_id="session-tiered", original_intent="tiering")
        await bb.update_context("task_one_result", "small")
        await bb.update_context("task_two_result", "small-2")
        await bb.update_context("analysis", {"k": "v"})
        tiers = bb.memory_tiers_snapshot()
        assert "task_two_result" in tiers["hot"]
        assert "task_one_result" in tiers["warm"]
        assert "analysis" in tiers["warm"]
    finally:
        settings.memory_tiering_enabled = old_enabled
        settings.memory_hot_max_items = old_hot_max


class _StageProbeSkillIndex:
    def __init__(self) -> None:
        self.stages: list[str] = []

    def retrieve_reflect_answer(self, original_intent: str, stage: str = "observe") -> list[dict[str, str]]:
        del original_intent
        self.stages.append(stage)
        return [{"title": f"stage:{stage}"}]

    def fetch_for_intent(self, original_intent: str) -> list[dict[str, str]]:
        del original_intent
        return [{"title": "fallback"}]


@pytest.mark.asyncio
async def test_blackboard_routes_retrieval_by_stage_when_tiering_enabled(temp_db: str, temp_lancedb: str) -> None:
    old_enabled = settings.memory_tiering_enabled
    old_stage = settings.memory_router_stage
    settings.memory_tiering_enabled = True
    probe = _StageProbeSkillIndex()
    try:
        for stage in ("observe", "soft", "hard"):
            settings.memory_router_stage = stage
            bb = Blackboard(session_id=f"stage-{stage}", original_intent="intent", skill_index=probe)
            skills = bb.fetch_relevant_skills()
            assert skills and skills[0]["title"] == f"stage:{stage}"
        settings.memory_router_stage = "invalid-stage"
        bb = Blackboard(session_id="stage-invalid", original_intent="intent", skill_index=probe)
        skills = bb.fetch_relevant_skills()
        assert skills and skills[0]["title"] == "stage:observe"
    finally:
        settings.memory_tiering_enabled = old_enabled
        settings.memory_router_stage = old_stage
