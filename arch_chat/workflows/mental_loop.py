"""Mental Loop: simulate before executing costly actions."""

from __future__ import annotations

import asyncio

from agno.agent import Agent

from arch_chat.models.context import RunContext, RunResult
from arch_chat.tools.mock_market import execute_action, simulate_action
from arch_chat.workflows.base import ArchitectureRunner

ESSENCE = "Counterfactual simulation before committing real actions"
NAME = "mental_loop"


class MentalLoopRunner(ArchitectureRunner):
    name = NAME
    essence = ESSENCE
    number = 11

    async def run(self, ctx: RunContext) -> RunResult:
        agent = Agent(
            model=ctx.model,
            name="trader",
            tools=[simulate_action, execute_action],
            instructions=[
                "Trading assistant with simulate_action and execute_action tools.",
                "Before any execute_action, always call simulate_action first.",
                "If simulated outcome is worse than holding, do not execute.",
            ],
            show_tool_calls=True,
        )
        ctx.add_trace("mental_loop", "Agent must simulate before execute")
        response = await asyncio.to_thread(agent.run, ctx.message)
        content = str(response.content or "")
        ctx.add_trace("response", content[:500])
        return RunResult(content=content, trace=ctx.trace, session_state=ctx.session_state)
