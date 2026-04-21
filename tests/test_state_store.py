import pytest

from core.memory.state_store import SessionStateStore
from core.models import AtomicTask, SubAgentResult, TrajectoryStep


@pytest.mark.asyncio
async def test_get_ready_tasks_preserves_dependency_and_key_semantics() -> None:
    store = SessionStateStore("s1", "intent")
    task_a = AtomicTask(id="A", description="a", required_capabilities=["search"], expected_output="x")
    task_b = AtomicTask(
        id="B",
        description="b",
        required_capabilities=["search"],
        expected_output="y",
        depends_on=["A"],
        required_keys=["k1"],
    )
    await store.add_todo(task_a)
    await store.add_todo(task_b)

    ready = await store.get_ready_tasks()
    assert [t.id for t in ready] == ["A"]
    assert [t.id for t in store.state.todo_list] == ["B"]

    await store.update_context("k1", 1)
    step = TrajectoryStep(
        step_id=1,
        task=task_a,
        result=SubAgentResult(task_id="A", status="success", output="ok"),
    )
    await store.record_step_in_memory(step)
    ready_after = await store.get_ready_tasks()
    assert [t.id for t in ready_after] == ["B"]


@pytest.mark.asyncio
async def test_semantic_merge_policy_is_deterministic() -> None:
    store = SessionStateStore("s2", "intent")
    task = AtomicTask(
        id="M",
        description="merge",
        required_capabilities=["search"],
        expected_output="o",
        branch_policy="semantic_merge",
    )
    await store.update_context("data", {"x": [1], "z": "old"})
    await store.apply_changeset(task, {"data": {"x": [1, 2], "y": "new"}})
    assert store.state.shared_memory["data"] == {"x": [1, 2], "z": "old", "y": "new"}


@pytest.mark.asyncio
async def test_get_context_filter_query_applies_matching() -> None:
    store = SessionStateStore("s3", "intent")
    await store.update_context("weather_report", "sunny in paris")
    await store.update_context("stock_report", "semiconductor trend")
    filtered = await store.get_context(["weather_report", "stock_report"], filter_query="paris")
    assert filtered == {"weather_report": "sunny in paris"}
