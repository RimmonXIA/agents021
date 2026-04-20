import asyncio
from typing import Any
from unittest.mock import AsyncMock, patch

import pytest
from agno.agent import Agent

from core.engine.orchestrator import IntentOrchestrator
from core.memory.blackboard import Blackboard
from core.models import AtomicTask, IOPlan


@patch.object(IntentOrchestrator, 'trigger_eo', new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_concurrent_execution(mock_eo: AsyncMock, temp_db: str, temp_lancedb: str) -> None:
    """
    Verifies that independent tasks run concurrently and dependencies are respected.
    Task A (0.2s) and Task B (0.1s) should run in parallel (~0.2s total).
    Task C should only start after A and B are done.
    """
    bb = Blackboard(session_id="test_concurrency", original_intent="Solve A and B")
    orchestrator = IntentOrchestrator(bb)
    
    # Define tasks
    task_a = AtomicTask(id="A", description="Task A", required_capabilities=["search"], expected_output="Result A")
    task_b = AtomicTask(id="B", description="Task B", required_capabilities=["search"], expected_output="Result B")
    task_c = AtomicTask(id="C", description="Task C", required_capabilities=["search"], expected_output="Result C", depends_on=["A", "B"])
    
    # Track execution order
    execution_log: list[str] = []

    async def mocked_safe_run(agent: Agent, prompt: str, **kwargs: Any) -> Any:
        now = asyncio.get_event_loop().time() - start_time
        if "Task A" in prompt:
            execution_log.append(f"A_start@{now:.2f}")
            await asyncio.sleep(0.2)
            execution_log.append(f"A_end@{asyncio.get_event_loop().time() - start_time:.2f}")
            return "Result A"
        if "Task B" in prompt:
            execution_log.append(f"B_start@{now:.2f}")
            await asyncio.sleep(0.1)
            execution_log.append(f"B_end@{asyncio.get_event_loop().time() - start_time:.2f}")
            return "Result B"
        if "Task C" in prompt:
            execution_log.append(f"C_start@{now:.2f}")
            execution_log.append(f"C_end@{asyncio.get_event_loop().time() - start_time:.2f}")
            return "Result C"
        # Planner decomposition
        return IOPlan(tasks=[task_a, task_b, task_c])

    with patch.object(IntentOrchestrator, '_safe_run', side_effect=mocked_safe_run):
        start_time = asyncio.get_event_loop().time()
        await orchestrator.run_loop()
        end_time = asyncio.get_event_loop().time()
        
        total_time = end_time - start_time
        print(f"\nExecution Log: {execution_log}")
        print(f"Total Time: {total_time}")
        
        # Time analysis:
        # A and B are parallel: max(0.2, 0.1) = 0.2
        # C is sequential after A and B: 0.2 + 0 = 0.2
        # If it were sequential: 0.2 + 0.1 + 0 = 0.3
        # We allow some overhead (0.05s)
        assert total_time < 0.28, f"Execution too slow: {total_time}s"
        
        # Verify DAG order in log
        def get_index(prefix: str) -> int:
            for i, entry in enumerate(execution_log):
                if entry.startswith(prefix):
                    return i
            raise ValueError(f"{prefix} not found in {execution_log}")

        assert get_index("A_start") < get_index("C_start")
        assert get_index("B_start") < get_index("C_start")
        assert get_index("A_end") < get_index("C_start")
        assert get_index("B_end") < get_index("C_start")
        
        assert bb.state.status == "completed"
        assert bb.state.shared_memory["A_result"] == "Result A"
        assert bb.state.shared_memory["B_result"] == "Result B"
        assert bb.state.shared_memory["C_result"] == "Result C"

@patch.object(IntentOrchestrator, 'trigger_eo', new_callable=AsyncMock)
@pytest.mark.asyncio
async def test_dynamic_key_scheduling(mock_eo: AsyncMock, temp_db: str, temp_lancedb: str) -> None:
    """
    Verifies that tasks wait for required_keys to appear in shared memory.
    """
    bb = Blackboard(session_id="test_keys", original_intent="Dynamic Keys")
    orchestrator = IntentOrchestrator(bb)
    
    # Task A produces 'data_x'
    task_a = AtomicTask(id="A", description="Task A", required_capabilities=["search"], expected_output="X")
    # Task B requires 'data_x'
    task_b = AtomicTask(id="B", description="Task B", required_capabilities=["search"], expected_output="Y", required_keys=["data_x"])
    
    async def mocked_safe_run(agent: Agent, prompt: str, **kwargs: Any) -> Any:
        if "Task A" in prompt:
            await asyncio.sleep(0.1)
            # Manually update blackboard with the required key
            await bb.update_context("data_x", "value_x")
            return "Result A"
        if "Task B" in prompt:
            return "Result B"
        return IOPlan(tasks=[task_a, task_b])

    with patch.object(IntentOrchestrator, '_safe_run', side_effect=mocked_safe_run):
        await orchestrator.run_loop()
        
        assert bb.state.status == "completed"
        assert "B_result" in bb.state.shared_memory
