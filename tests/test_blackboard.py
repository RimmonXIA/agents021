import sqlite3

import pytest

from core.memory.blackboard import Blackboard
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
    
    # Verify in SQLite
    with sqlite3.connect(temp_db) as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT step_id, task_json FROM trajectories WHERE session_id='test_session'")
        row = cursor.fetchone()
        assert row is not None
        assert row[0] == 1
        assert '"id":"t1"' in row[1]
