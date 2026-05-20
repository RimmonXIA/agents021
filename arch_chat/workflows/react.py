"""ReAct: reasoning + tool observation loop."""

from __future__ import annotations

from agno.agent import Agent

from arch_chat.models.context import RunContext, RunResult
from arch_chat.runner import run_agent
from arch_chat.tools.mock_search import flaky_web_search, mock_stock_price
from arch_chat.workflows.base import ArchitectureRunner

ESSENCE = "Thought → Action → Observation loop until done"
NAME = "react"


class ReActRunner(ArchitectureRunner):
    name = NAME
    essence = ESSENCE
    number = 3

    async def run(self, ctx: RunContext) -> RunResult:
        agent = Agent(
            model=ctx.model,
            name="react_agent",
            tools=[flaky_web_search, mock_stock_price],
            instructions=[
                "Research assistant. Think step by step.",
                "For each step decide whether to use a tool or answer.",
                "After each tool observation, re-evaluate before answering.",
            ],
            reasoning=True,
            markdown=True,
        )
        ctx.add_trace("react", "Starting ReAct loop with reasoning=True")
        result = await run_agent(agent, ctx.message, ctx=ctx, step_name="react")
        content = result.content or ""
        ctx.add_trace("answer", content[:500])
        return RunResult(content=content, trace=ctx.trace, session_state=ctx.session_state)
