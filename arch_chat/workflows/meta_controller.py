"""Meta-Controller: one-shot entry routing to specialist agents."""

from __future__ import annotations

import asyncio

from agno.agent import Agent
from agno.team.team import Team, TeamMode

from arch_chat.models.context import RunContext, RunResult
from arch_chat.router.classifier import is_conversational
from arch_chat.runner import run_agent
from arch_chat.workflows.base import ArchitectureRunner

ESSENCE = "Route once to the best specialist at the entry point"
NAME = "meta_controller"


class MetaControllerRunner(ArchitectureRunner):
    name = NAME
    essence = ESSENCE
    number = 12

    async def run(self, ctx: RunContext) -> RunResult:
        generalist = Agent(
            name="generalist",
            model=ctx.model,
            instructions="General Q&A assistant. Answer clearly and concisely.",
        )

        if is_conversational(ctx.message):
            ctx.emit_step_start("fast_path")
            ctx.add_trace("fast_path", "Conversational message → direct generalist")
            result = await run_agent(generalist, ctx.message, ctx=ctx, step_name="generalist")
            content = result.content or ""
            ctx.session_state["selected_agent"] = "generalist"
            ctx.add_trace("generalist", content[:500], ctx.snapshot_state())
            return RunResult(content=content, trace=ctx.trace, session_state=ctx.session_state)

        researcher = Agent(
            name="researcher",
            model=ctx.model,
            instructions="Research assistant for multi-step factual questions.",
        )
        coder = Agent(
            name="coder",
            model=ctx.model,
            instructions="Python coding assistant. Provide working code with explanation.",
        )

        team = Team(
            name="meta_controller",
            mode=TeamMode.route,
            model=ctx.model,
            members=[generalist, researcher, coder],
            instructions=(
                "Choose exactly one member: generalist for general Q&A, "
                "researcher for research-heavy queries, coder for Python tasks. "
                "Do not do the work yourself."
            ),
        )
        ctx.emit_step_start("meta_controller")
        if ctx.stream:
            ctx.stream.spinner("Team routing to specialist...")
        ctx.add_trace("meta_controller", "Team(mode=route) selecting specialist")
        response = await asyncio.to_thread(team.run, ctx.message)
        content = str(response.content or "")
        member = getattr(response, "member_name", None) or "unknown"
        ctx.session_state["selected_agent"] = member
        ctx.add_trace("routed_to", member, ctx.snapshot_state())
        ctx.add_trace("response", content[:500])
        if ctx.stream and content:
            ctx.stream.token(content)
        return RunResult(content=content, trace=ctx.trace, session_state=ctx.session_state)
