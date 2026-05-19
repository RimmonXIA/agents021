from __future__ import annotations

from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from agno.models.deepseek import DeepSeek

from arch_chat.config import Settings

if TYPE_CHECKING:
    from arch_chat.tools.in_memory_graph import InMemoryGraph
    from arch_chat.tui.stream_ui import StreamSink


ApprovalCallback = Callable[[str], Awaitable[bool]]


def _default_graph() -> InMemoryGraph:
    from arch_chat.tools.in_memory_graph import InMemoryGraph

    return InMemoryGraph()


@dataclass
class TraceStep:
    name: str
    content: str
    state_snapshot: dict[str, Any] | None = None


@dataclass
class RunResult:
    content: str
    trace: list[TraceStep] = field(default_factory=list)
    session_state: dict[str, Any] = field(default_factory=dict)
    refused: bool = False


@dataclass
class RunContext:
    message: str
    settings: Settings
    session_id: str
    user_id: str
    model: DeepSeek
    session_state: dict[str, Any] = field(default_factory=dict)
    trace: list[TraceStep] = field(default_factory=list)
    approval_callback: ApprovalCallback | None = None
    graph: InMemoryGraph = field(default_factory=_default_graph)
    dry_run_mode: bool = True
    episodic_facts: list[str] = field(default_factory=list)
    semantic_facts: list[str] = field(default_factory=list)
    stream: StreamSink | None = None

    def emit_step_start(self, name: str, agent: str = "") -> None:
        if self.stream:
            self.stream.step_start(name, agent)

    def add_trace(self, name: str, content: str, state: dict[str, Any] | None = None) -> None:
        self.trace.append(TraceStep(name=name, content=content, state_snapshot=state))
        if self.stream:
            self.stream.step_end(name, content)

    def snapshot_state(self) -> dict[str, Any]:
        return dict(self.session_state)
