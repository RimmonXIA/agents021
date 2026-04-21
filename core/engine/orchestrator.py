import asyncio
from typing import Any

from agno.agent import Agent

from core.agents.synthesizer import AgentSynthesizer
from core.config import settings
from core.engine.ports import StatePort
from core.engine.optimizer import EvolutionaryOptimizer
from core.engine.planner_pipeline import PlannerPipeline
from core.engine.run_loop_controller import RunLoopController
from core.engine.task_executor import TaskExecutor
from core.memory.blackboard import Blackboard
from core.memory.compactor import SemanticCompactor
from core.models import AtomicTask


class IntentOrchestrator:
    """
    The Intent Orchestrator (IO) - The "nO Master Loop".
    Composes planner streaming, task execution, and the run loop.
    """

    def __init__(self, blackboard: StatePort, ui_callback: Any = None) -> None:
        self.bb = blackboard
        self.ui_callback = ui_callback
        self.asynth = AgentSynthesizer()
        self.planner: Agent = self.asynth.synthesize("planner", {})
        self.compactor = SemanticCompactor(self.asynth)
        self.concurrency_limit = asyncio.Semaphore(settings.max_concurrency)
        self._planner_pipeline = PlannerPipeline(self.bb, self.planner)
        self._task_executor = TaskExecutor(
            self.bb, self.asynth, self.ui_callback, self.concurrency_limit
        )
        self._run_loop = RunLoopController(
            self.bb,
            self._invoke_planner_decompose,
            self._task_executor.execute,
            self.compactor,
            self.ui_callback,
            self.trigger_eo,
        )

    @property
    def running_tasks(self) -> dict[str, tuple[AtomicTask, asyncio.Task[None]]]:
        """Exposes in-flight asyncio tasks for CLI live UI (compat with pre-refactor IO)."""
        return self._run_loop.running_tasks

    async def _invoke_planner_decompose(self) -> None:
        """Resolves `decompose_intent` at call time (tests may replace the method)."""
        await self._planner_pipeline.decompose_intent()

    async def decompose_intent(self) -> None:
        """Entry point for HITL plan review: stream planner output into todos only."""
        await self._planner_pipeline.decompose_intent()

    async def run_loop(self) -> None:
        """Full orchestration: decomposition + scheduling + compaction; EO runs in background."""
        await self._run_loop.run()

    async def trigger_eo(self) -> None:
        """Runs EvolutionaryOptimizer (invoked from run loop as a background task)."""
        eo_kwargs: dict[str, object] = {}
        if hasattr(self.bb, "trajectory_store"):
            eo_kwargs["trajectory_store"] = getattr(self.bb, "trajectory_store")
        if hasattr(self.bb, "skill_index"):
            eo_kwargs["skill_store"] = getattr(self.bb, "skill_index")
        eo = EvolutionaryOptimizer(**eo_kwargs)
        await eo.process_session(self.bb.state.session_id)
