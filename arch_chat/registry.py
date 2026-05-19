from __future__ import annotations

from dataclasses import dataclass

from arch_chat.workflows.base import ArchitectureRunner
from arch_chat.workflows.blackboard import BlackboardRunner
from arch_chat.workflows.cellular_automata import CellularAutomataRunner
from arch_chat.workflows.dry_run import DryRunRunner
from arch_chat.workflows.ensemble import EnsembleRunner
from arch_chat.workflows.episodic_memory import EpisodicMemoryRunner
from arch_chat.workflows.graph_memory import GraphMemoryRunner
from arch_chat.workflows.mental_loop import MentalLoopRunner
from arch_chat.workflows.meta_controller import MetaControllerRunner
from arch_chat.workflows.metacognitive import MetacognitiveRunner
from arch_chat.workflows.multi_agent import MultiAgentRunner
from arch_chat.workflows.pev import PEVRunner
from arch_chat.workflows.planning import PlanningRunner
from arch_chat.workflows.react import ReActRunner
from arch_chat.workflows.reflection import ReflectionRunner
from arch_chat.workflows.self_improve import SelfImproveRunner
from arch_chat.workflows.tool_use import ToolUseRunner
from arch_chat.workflows.tot import ToTRunner


@dataclass(frozen=True)
class ArchEntry:
    name: str
    essence: str
    number: int
    runner: ArchitectureRunner


def _build_registry() -> dict[str, ArchEntry]:
    runners: list[ArchitectureRunner] = [
        ReflectionRunner(),
        ToolUseRunner(),
        ReActRunner(),
        PlanningRunner(),
        PEVRunner(),
        MultiAgentRunner(),
        BlackboardRunner(),
        EpisodicMemoryRunner(),
        GraphMemoryRunner(),
        ToTRunner(),
        MentalLoopRunner(),
        MetaControllerRunner(),
        EnsembleRunner(),
        DryRunRunner(),
        MetacognitiveRunner(),
        SelfImproveRunner(),
        CellularAutomataRunner(),
    ]
    return {
        r.name: ArchEntry(name=r.name, essence=r.essence, number=r.number, runner=r)
        for r in runners
    }


ARCH_REGISTRY: dict[str, ArchEntry] = _build_registry()


def list_architectures() -> list[ArchEntry]:
    return sorted(ARCH_REGISTRY.values(), key=lambda e: e.number)


def get_runner(name: str) -> ArchitectureRunner | None:
    entry = ARCH_REGISTRY.get(name.lower().replace("-", "_").replace(" ", "_"))
    if entry:
        return entry.runner
    for key, entry in ARCH_REGISTRY.items():
        if key.replace("_", "") == name.lower().replace("-", "").replace("_", "").replace(" ", ""):
            return entry.runner
    return None
