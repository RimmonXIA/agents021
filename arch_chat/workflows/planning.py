"""Planning: explicit plan state → execute loop → synthesize."""

from __future__ import annotations

from agno.agent import Agent

from arch_chat.models.context import RunContext, RunResult
from arch_chat.models.state import Plan
from arch_chat.runner import run_agent
from arch_chat.tools.mock_search import flaky_web_search
from arch_chat.workflows.base import ArchitectureRunner

ESSENCE = "Generate explicit step list, then execute in order"
NAME = "planning"


class PlanningRunner(ArchitectureRunner):
    name = NAME
    essence = ESSENCE
    number = 4

    async def run(self, ctx: RunContext) -> RunResult:
        planner = Agent(
            name="planner",
            model=ctx.model,
            output_schema=Plan,
            instructions="Decompose the request into atomic tool-queryable steps.",
        )
        executor = Agent(
            name="executor",
            model=ctx.model,
            tools=[flaky_web_search],
            instructions="Answer exactly one sub-question using tools.",
        )
        synthesizer = Agent(
            name="synthesizer",
            model=ctx.model,
            instructions="Combine intermediate findings into a final answer.",
        )

        plan_result = await run_agent(
            planner, ctx.message, ctx=ctx, step_name="plan", response_model=Plan
        )
        if not plan_result.success:
            return RunResult(content=f"Planner failed: {plan_result.error}", trace=ctx.trace)
        plan: Plan = plan_result.parsed
        ctx.session_state["plan"] = list(plan.steps)
        ctx.session_state["intermediate"] = []
        ctx.add_trace("plan", "\n".join(f"- {s}" for s in plan.steps), ctx.snapshot_state())

        step_idx = 0
        while ctx.session_state["plan"]:
            next_q = ctx.session_state["plan"].pop(0)
            step_idx += 1
            exec_result = await run_agent(
                executor, next_q, ctx=ctx, step_name=f"execute_{step_idx}"
            )
            obs = exec_result.content or ""
            ctx.session_state["intermediate"].append(f"Q: {next_q}\nA: {obs}")
            ctx.add_trace("observation", obs[:400], ctx.snapshot_state())

        notes = "\n\n".join(ctx.session_state["intermediate"])
        synth_prompt = f"Question: {ctx.message}\nNotes:\n{notes}\nFinal answer:"
        synth_result = await run_agent(
            synthesizer, synth_prompt, ctx=ctx, step_name="synthesize"
        )
        content = synth_result.content or ""
        ctx.add_trace("synthesize", content[:400])
        return RunResult(content=content, trace=ctx.trace, session_state=ctx.session_state)
