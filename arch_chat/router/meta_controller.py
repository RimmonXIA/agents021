"""Meta-Controller routing: heuristics + LLM classifier."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agno.agent import Agent
from agno.models.deepseek import DeepSeek

from arch_chat.models.state import RoutingDecision
from arch_chat.registry import ARCH_REGISTRY, list_architectures
from arch_chat.router.classifier import (
    SPECIALIZED_ARCHS,
    has_code_intent,
    heuristic_route,
)
from arch_chat.runner import run_agent

if TYPE_CHECKING:
    from arch_chat.tui.stream_ui import StreamSink

ROUTER_INSTRUCTIONS = """
Route the user message to exactly one architecture.

Rules:
- reflection: ONLY for explicit code generation or refactoring requests
- meta_controller: greetings, identity questions, general Q&A, ambiguous messages
- Never pick specialized architectures (tot, cellular_automata, dry_run, pev, mental_loop,
  graph_memory, ensemble, blackboard) unless the message clearly matches that pattern
- Prefer meta_controller when uncertain
"""


def validate_routing(message: str, decision: RoutingDecision) -> RoutingDecision:
    arch = decision.architecture.lower().replace("-", "_")

    if arch == "reflection" and not has_code_intent(message):
        return RoutingDecision(
            architecture="meta_controller",
            confidence=0.9,
            reasoning="Override: reflection requires code intent; routing to general chat.",
        )

    if decision.confidence < 0.6:
        return RoutingDecision(
            architecture="meta_controller",
            confidence=0.8,
            reasoning=f"Override: low confidence ({decision.confidence:.0%}); defaulting to general chat.",
        )

    if arch in SPECIALIZED_ARCHS:
        from arch_chat.router.classifier import _matches_specialized

        if not _matches_specialized(message, arch):
            return RoutingDecision(
                architecture="meta_controller",
                confidence=0.85,
                reasoning=f"Override: {arch} requires matching keywords; routing to general chat.",
            )

    return decision


async def route_message(
    message: str,
    model: DeepSeek,
    *,
    stream: StreamSink | None = None,
) -> RoutingDecision:
    hint = heuristic_route(message)
    if hint and hint in ARCH_REGISTRY:
        entry = ARCH_REGISTRY[hint]
        return RoutingDecision(
            architecture=hint,
            confidence=0.85,
            reasoning=f"Heuristic match: {entry.essence}",
        )

    if stream:
        stream.spinner("Classifying architecture...")

    arch_list = "\n".join(f"- {e.name}: {e.essence}" for e in list_architectures())
    classifier = Agent(
        name="router",
        model=model,
        output_schema=RoutingDecision,
        instructions=(
            ROUTER_INSTRUCTIONS
            + f"\nValid names: {', '.join(ARCH_REGISTRY.keys())}.\n"
            + f"Architectures:\n{arch_list}"
        ),
    )

    result = await run_agent(
        classifier,
        f"User message: {message}\nPick the best architecture.",
        stream=stream,
        step_name="router",
        response_model=RoutingDecision,
    )
    if result.success and result.parsed:
        decision: RoutingDecision = result.parsed
        if decision.architecture.lower().replace("-", "_") in ARCH_REGISTRY:
            decision.architecture = decision.architecture.lower().replace("-", "_")
            return validate_routing(message, decision)

    return RoutingDecision(
        architecture="meta_controller",
        confidence=0.5,
        reasoning="Default general chat path.",
    )
