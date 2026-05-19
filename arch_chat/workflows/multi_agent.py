"""Multi-Agent: fixed role pipeline (research → analysis → synthesis)."""

from __future__ import annotations

from agno.agent import Agent

from arch_chat.models.context import RunContext, RunResult
from arch_chat.runner import run_agent
from arch_chat.tools.mock_search import flaky_web_search
from arch_chat.workflows.base import ArchitectureRunner

ESSENCE = "Fixed cognitive role decomposition in a linear pipeline"
NAME = "multi_agent"


class MultiAgentRunner(ArchitectureRunner):
    name = NAME
    essence = ESSENCE
    number = 6

    async def run(self, ctx: RunContext) -> RunResult:
        news_analyst = Agent(
            name="news_analyst",
            model=ctx.model,
            tools=[flaky_web_search],
            instructions="Financial news analyst. Concise markdown on recent news.",
        )
        technical_analyst = Agent(
            name="technical_analyst",
            model=ctx.model,
            instructions="Technical analyst. Concise markdown on trends and indicators.",
        )
        financial_analyst = Agent(
            name="financial_analyst",
            model=ctx.model,
            instructions="Financial analyst. Concise markdown on fundamentals.",
        )
        report_writer = Agent(
            name="report_writer",
            model=ctx.model,
            instructions="Compose final investment memo from three sub-reports.",
        )

        news_result = await run_agent(news_analyst, ctx.message)
        ctx.session_state["news"] = news_result.content or ""
        ctx.add_trace("news_analyst", (news_result.content or "")[:400], ctx.snapshot_state())

        tech_prompt = f"Original request: {ctx.message}\nNews context:\n{ctx.session_state['news']}"
        tech_result = await run_agent(technical_analyst, tech_prompt)
        ctx.session_state["tech"] = tech_result.content or ""
        ctx.add_trace("technical_analyst", (tech_result.content or "")[:400], ctx.snapshot_state())

        fin_prompt = f"Original request: {ctx.message}\nPrior sections available on blackboard."
        fin_result = await run_agent(financial_analyst, fin_prompt)
        ctx.session_state["fin"] = fin_result.content or ""
        ctx.add_trace("financial_analyst", (fin_result.content or "")[:400], ctx.snapshot_state())

        write_prompt = (
            f"User request: {ctx.message}\n\n"
            f"## News\n{ctx.session_state['news']}\n\n"
            f"## Technical\n{ctx.session_state['tech']}\n\n"
            f"## Financial\n{ctx.session_state['fin']}\n\n"
            "Write the final report."
        )
        write_result = await run_agent(report_writer, write_prompt)
        content = write_result.content or ""
        ctx.add_trace("report_writer", content[:400])
        return RunResult(content=content, trace=ctx.trace, session_state=ctx.session_state)
