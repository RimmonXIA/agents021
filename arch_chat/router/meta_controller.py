"""LLM-native architecture routing via structured API call."""

from __future__ import annotations

from typing import TYPE_CHECKING

from agno.agent import Agent
from agno.models.deepseek import DeepSeek

from arch_chat.models.state import RoutingDecision
from arch_chat.registry import ARCH_REGISTRY, get_runner
from arch_chat.router.prompts import build_router_system_prompt, build_router_user_prompt
from arch_chat.runner import run_agent

if TYPE_CHECKING:
    from arch_chat.tui.stream_ui import StreamSink

DEFAULT_ARCHITECTURE = "meta_controller"


def normalize_decision(decision: RoutingDecision) -> RoutingDecision:
    """Map LLM output to a valid registry key; fallback only on invalid names."""
    raw = decision.architecture.lower().replace("-", "_").replace(" ", "_").strip()
    if raw in ARCH_REGISTRY:
        return RoutingDecision(
            architecture=raw,
            confidence=decision.confidence,
            reasoning=decision.reasoning,
        )
    if get_runner(raw):
        normalized = raw
        for key in ARCH_REGISTRY:
            if key.replace("_", "") == raw.replace("_", ""):
                normalized = key
                break
        return RoutingDecision(
            architecture=normalized,
            confidence=decision.confidence,
            reasoning=decision.reasoning,
        )
    return RoutingDecision(
        architecture=DEFAULT_ARCHITECTURE,
        confidence=0.5,
        reasoning=(
            f"LLM returned unknown architecture '{decision.architecture}'; "
            f"defaulting to {DEFAULT_ARCHITECTURE}. Original: {decision.reasoning}"
        ),
    )


async def route_message(
    message: str,
    model: DeepSeek,
    *,
    stream: StreamSink | None = None,
) -> RoutingDecision:
    """Route exclusively via LLM structured output — no heuristic pre-filter."""
    if stream:
        stream.spinner("LLM routing architecture...")

    classifier = Agent(
        name="architecture_router",
        model=model,
        output_schema=RoutingDecision,
        instructions=build_router_system_prompt(),
    )

    result = await run_agent(
        classifier,
        build_router_user_prompt(message),
        stream=stream,
        step_name="router",
        response_model=RoutingDecision,
    )

    if result.success and result.parsed:
        return normalize_decision(result.parsed)

    return RoutingDecision(
        architecture=DEFAULT_ARCHITECTURE,
        confidence=0.4,
        reasoning="LLM routing failed; defaulting to general chat.",
    )
