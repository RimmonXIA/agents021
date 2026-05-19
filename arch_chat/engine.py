"""Architecture chat engine — routes and runs workflows."""

from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from agno.models.deepseek import DeepSeek

from arch_chat.config import Settings
from arch_chat.memory.episodic import SessionMemory
from arch_chat.models.context import RunContext, RunResult
from arch_chat.models.state import RoutingDecision
from arch_chat.registry import ARCH_REGISTRY, get_runner
from arch_chat.router.meta_controller import route_message
from arch_chat.tools.in_memory_graph import InMemoryGraph

if TYPE_CHECKING:
    from arch_chat.tui.stream_ui import StreamSink


class ChatEngine:
    def __init__(
        self,
        settings: Settings,
        session_id: str | None = None,
        user_id: str = "default",
    ):
        self.settings = settings
        self.session_id = session_id or str(uuid.uuid4())[:8]
        self.user_id = user_id
        self.model = DeepSeek(id=settings.chat_model, api_key=settings.deepseek_api_key)
        self.memory = SessionMemory()
        self.graph = InMemoryGraph()
        self.session_state: dict = {}
        self.arch_override: str | None = None
        self.auto_route = True
        self.trace_verbose = True
        self.stream_enabled = True
        self.last_routing: RoutingDecision | None = None
        self.approval_callback = None
        self.stream_sink: StreamSink | None = None

    async def handle_message(self, message: str) -> RunResult:
        sink = self.stream_sink if self.stream_enabled else None
        if sink:
            sink.start_turn(message)
            sink.phase("routing")

        if self.auto_route and not self.arch_override:
            decision = await route_message(message, self.model, stream=sink)
            arch_name = decision.architecture
            overridden = False
        else:
            arch_name = self.arch_override or "react"
            entry = ARCH_REGISTRY.get(arch_name)
            decision = RoutingDecision(
                architecture=arch_name,
                confidence=1.0,
                reasoning=f"Manual override to {arch_name}" + (f": {entry.essence}" if entry else ""),
            )
            overridden = bool(self.arch_override)

        self.last_routing = decision
        if sink:
            sink.routing(decision, overridden=overridden)
            sink.phase("running", arch_name)

        runner = get_runner(arch_name)
        if not runner:
            result = RunResult(content=f"Unknown architecture: {arch_name}")
            if sink:
                sink.error(result.content)
            return result

        needs_metacog = arch_name not in ("metacognitive", "dry_run", "cellular_automata", "tot")
        if needs_metacog:
            from arch_chat.router.classifier import RISK_KEYWORDS

            if any(k in message.lower() for k in RISK_KEYWORDS):
                metacog = get_runner("metacognitive")
                if metacog:
                    pre_ctx = self._make_context(message)
                    pre_result = await metacog.run(pre_ctx)
                    if pre_result.refused:
                        pre_result.trace = pre_ctx.trace
                        if sink:
                            sink.finalize(pre_result)
                        return pre_result

        ctx = self._make_context(message)
        result = await runner.run(ctx)
        result.trace = ctx.trace

        if arch_name == "episodic_memory" and "remember" in message.lower():
            self.memory.episodic_facts = ctx.episodic_facts

        return result

    def _make_context(self, message: str) -> RunContext:
        sink = self.stream_sink if self.stream_enabled else None
        return RunContext(
            message=message,
            settings=self.settings,
            session_id=self.session_id,
            user_id=self.user_id,
            model=self.model,
            session_state=dict(self.session_state),
            approval_callback=self.approval_callback,
            graph=self.graph,
            episodic_facts=list(self.memory.episodic_facts),
            semantic_facts=list(self.memory.semantic_facts),
            stream=sink,
        )

    def reset_session(self) -> None:
        self.session_state.clear()

    def clear_memory(self) -> None:
        self.memory.clear()
