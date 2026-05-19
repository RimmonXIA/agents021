"""Blackboard: shared workspace + dynamic controller scheduling."""

from __future__ import annotations

from agno.agent import Agent

from arch_chat.models.context import RunContext, RunResult
from arch_chat.models.state import ControllerDecision
from arch_chat.runner import run_agent
from arch_chat.tools.mock_search import flaky_web_search
from arch_chat.workflows.base import ArchitectureRunner

ESSENCE = "Controller reads shared blackboard and picks next specialist"
NAME = "blackboard"

SPECIALIST_NAMES = ["news", "technical", "financial", "writer", "FINISH"]


class BlackboardRunner(ArchitectureRunner):
    name = NAME
    essence = ESSENCE
    number = 7

    async def run(self, ctx: RunContext) -> RunResult:
        ctx.session_state.setdefault("blackboard", {})
        ctx.session_state["user_request"] = ctx.message
        ctx.session_state["next_agent"] = "news"

        specialists = {
            "news": Agent(
                name="news",
                model=ctx.model,
                tools=[flaky_web_search],
                instructions="Add a news section to the shared report.",
            ),
            "technical": Agent(
                name="technical",
                model=ctx.model,
                instructions="Add a technical analysis section.",
            ),
            "financial": Agent(
                name="financial",
                model=ctx.model,
                instructions="Add a financial fundamentals section.",
            ),
            "writer": Agent(
                name="writer",
                model=ctx.model,
                instructions="Synthesize blackboard sections into final report.",
            ),
        }
        controller = Agent(
            name="controller",
            model=ctx.model,
            output_schema=ControllerDecision,
            instructions=(
                "Controller of a blackboard system. Inspect blackboard and decide "
                f"next specialist from {SPECIALIST_NAMES} or FINISH if report ready."
            ),
        )

        max_iters = 8
        for i in range(max_iters):
            bb = ctx.session_state["blackboard"]
            import json

            snapshot = json.dumps(bb, indent=2, ensure_ascii=False)
            decision_result = await run_agent(
                controller,
                f"Original request: {ctx.message}\n\nBlackboard:\n{snapshot}",
                response_model=ControllerDecision,
            )
            if not decision_result.success:
                break
            decision: ControllerDecision = decision_result.parsed
            ctx.session_state["next_agent"] = decision.next_agent
            ctx.add_trace(
                f"controller_{i}",
                f"→ {decision.next_agent}: {decision.reasoning}",
                ctx.snapshot_state(),
            )

            if decision.next_agent == "FINISH":
                break

            agent = specialists.get(decision.next_agent)
            if not agent:
                ctx.session_state["next_agent"] = "FINISH"
                break

            spec_result = await run_agent(
                agent,
                f"User request: {ctx.message}\nBlackboard so far:\n{snapshot}",
            )
            ctx.session_state["blackboard"][decision.next_agent] = spec_result.content or ""
            ctx.add_trace(
                f"specialist_{decision.next_agent}",
                (spec_result.content or "")[:400],
                ctx.snapshot_state(),
            )

        content = ctx.session_state["blackboard"].get(
            "writer",
            "\n\n".join(
                f"## {k.title()}\n{v}"
                for k, v in ctx.session_state["blackboard"].items()
            ),
        )
        return RunResult(content=str(content), trace=ctx.trace, session_state=ctx.session_state)
