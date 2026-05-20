"""Tool Use: single agent with structured tool interface."""

from __future__ import annotations

from agno.agent import Agent

from arch_chat.models.context import RunContext, RunResult
from arch_chat.tools.mock_search import mock_stock_price
from arch_chat.workflows.base import ArchitectureRunner

ESSENCE = "Break knowledge boundary via structured tool calls"
NAME = "tool_use"


class ToolUseRunner(ArchitectureRunner):
    name = NAME
    essence = ESSENCE
    number = 2

    async def run(self, ctx: RunContext) -> RunResult:
        agent = Agent(
            model=ctx.model,
            name="tool_agent",
            tools=[mock_stock_price],
            instructions="Use tools to answer questions needing real-time data.",
        )
        ctx.add_trace("tool_use", "Starting agent with mock_stock_price tool")
        import asyncio

        response = await asyncio.to_thread(agent.run, ctx.message)
        content = str(response.content or "")
        ctx.add_trace("response", content[:500])
        return RunResult(content=content, trace=ctx.trace, session_state=ctx.session_state)
