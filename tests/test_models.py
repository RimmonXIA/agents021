import pytest
from pydantic import ValidationError

from core.models import AtomicTask, GlobalState, SubAgentResult


def test_atomic_task_validation() -> None:
    """Verify that AtomicTask requires mandatory fields and validates types."""
    # Valid task
    task = AtomicTask(
        id="task_1",
        description="Search for something",
        required_capabilities=["search"],
        expected_output="A summary"
    )
    assert task.id == "task_1"
    assert task.context_keys == []  # Default factory

    # Invalid task (missing field)
    with pytest.raises(ValidationError):
        AtomicTask(id="task_2")  # type: ignore[call-arg]

def test_subagent_result_validation() -> None:
    """Verify SubAgentResult data structure."""
    result = SubAgentResult(
        task_id="task_1",
        status="success",
        output="Found it"
    )
    assert result.status == "success"
    assert result.artifacts == {}

def test_global_state_initialization() -> None:
    """Verify GlobalState default values."""
    state = GlobalState(session_id="session_123", original_intent="Goal")
    assert state.status == "running"
    assert len(state.todo_list) == 0
    assert isinstance(state.shared_memory, dict)
