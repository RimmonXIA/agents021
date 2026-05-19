"""Ensemble: parallel redundant agents + aggregator fusion."""

from __future__ import annotations

import asyncio

from agno.agent import Agent

from arch_chat.models.context import RunContext, RunResult
from arch_chat.runner import run_agent
from arch_chat.workflows.base import ArchitectureRunner

ESSENCE = "Parallel perspectives on the same question, then aggregate"
NAME = "ensemble"


class EnsembleRunner(ArchitectureRunner):
    name = NAME
    essence = ESSENCE
    number = 13

    async def run(self, ctx: RunContext) -> RunResult:
        bullish = Agent(
            name="bullish",
            model=ctx.model,
            instructions="Optimistic analyst. Argue the bullish case with risks noted.",
        )
        value = Agent(
            name="value",
            model=ctx.model,
            instructions="Value investor. Focus on fundamentals and margin of safety.",
        )
        quant = Agent(
            name="quant",
            model=ctx.model,
            instructions="Quant analyst. Focus on data, trends, and statistical signals.",
        )
        cio = Agent(
            name="cio",
            model=ctx.model,
            instructions=(
                "CIO synthesizing three analyst views. Preserve disagreements and risks."
            ),
        )

        ctx.session_state["views"] = {}
        ctx.add_trace("ensemble", "Fan-out: bullish, value, quant in parallel")

        async def run_one(name: str, agent: Agent) -> tuple[str, str]:
            result = await run_agent(agent, ctx.message)
            return name, result.content or ""

        results = await asyncio.gather(
            run_one("bullish", bullish),
            run_one("value", value),
            run_one("quant", quant),
        )
        for name, view in results:
            ctx.session_state["views"][name] = view
            ctx.add_trace(name, view[:300], ctx.snapshot_state())

        views_text = "\n\n".join(
            f"## {n.title()} View\n{v}" for n, v in ctx.session_state["views"].items()
        )
        agg_result = await run_agent(
            cio,
            f"Question: {ctx.message}\n\nAnalyst views:\n{views_text}\n\nSynthesize.",
        )
        content = agg_result.content or ""
        ctx.add_trace("cio_synth", content[:400])
        return RunResult(content=content, trace=ctx.trace, session_state=ctx.session_state)
