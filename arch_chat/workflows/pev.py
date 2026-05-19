"""PEV: Plan → Execute → Verify with replan on failure."""

from __future__ import annotations

from agno.agent import Agent

from arch_chat.models.context import RunContext, RunResult
from arch_chat.models.state import Plan, VerificationResult
from arch_chat.runner import run_agent
from arch_chat.tools.mock_search import flaky_web_search
from arch_chat.workflows.base import ArchitectureRunner

ESSENCE = "Verify each tool result; replan after repeated failures"
NAME = "pev"


class PEVRunner(ArchitectureRunner):
    name = NAME
    essence = ESSENCE
    number = 5

    async def run(self, ctx: RunContext) -> RunResult:
        planner = Agent(
            name="planner",
            model=ctx.model,
            output_schema=Plan,
            instructions="Decompose into atomic tool-queryable steps.",
        )
        executor = Agent(
            name="pev_executor",
            model=ctx.model,
            tools=[flaky_web_search],
            instructions="Answer exactly one sub-question using tools.",
        )
        verifier = Agent(
            name="verifier",
            model=ctx.model,
            output_schema=VerificationResult,
            instructions=(
                "Given sub-question and raw tool observation, decide if observation "
                "answers it. Treat Error/unavailable/empty as failures."
            ),
        )
        synthesizer = Agent(
            name="synthesizer",
            model=ctx.model,
            instructions="Combine verified findings into final answer.",
        )

        ctx.session_state.setdefault("retries", 0)
        ctx.session_state.setdefault("intermediate", [])

        async def make_plan() -> bool:
            plan_result = await run_agent(
                planner, ctx.message, ctx=ctx, step_name="plan", response_model=Plan
            )
            if not plan_result.success:
                return False
            plan: Plan = plan_result.parsed
            ctx.session_state["plan"] = list(plan.steps)
            ctx.add_trace("plan", "\n".join(f"- {s}" for s in plan.steps), ctx.snapshot_state())
            return True

        if not await make_plan():
            return RunResult(content="Planning failed.", trace=ctx.trace)

        max_replans = 2
        replans = 0
        exec_idx = 0

        while ctx.session_state.get("plan"):
            next_q = ctx.session_state["plan"][0]
            exec_idx += 1
            exec_result = await run_agent(
                executor, next_q, ctx=ctx, step_name=f"pev_execute_{exec_idx}"
            )
            last_obs = exec_result.content or ""
            ctx.session_state["last_obs"] = last_obs
            ctx.session_state["last_q"] = next_q
            ctx.add_trace("pev_execute", f"Sub-question: {next_q}")

            verdict_result = await run_agent(
                verifier,
                f"Sub-question: {next_q}\nObservation:\n{last_obs}",
                ctx=ctx,
                step_name=f"pev_verify_{exec_idx}",
                response_model=VerificationResult,
            )
            if not verdict_result.success:
                return RunResult(content="Verifier failed.", trace=ctx.trace)
            verdict: VerificationResult = verdict_result.parsed
            ctx.session_state["last_verdict"] = verdict.model_dump()
            ctx.add_trace(
                "pev_verify",
                f"success={verdict.is_successful}: {verdict.reasoning}",
                ctx.snapshot_state(),
            )

            if verdict.is_successful:
                ctx.session_state["plan"].pop(0)
                ctx.session_state["intermediate"].append(f"Q: {next_q}\nA: {last_obs}")
                ctx.session_state["retries"] = 0
            else:
                ctx.session_state["retries"] = ctx.session_state.get("retries", 0) + 1
                if ctx.session_state["retries"] >= 2:
                    ctx.session_state["retries"] = 0
                    replans += 1
                    if replans > max_replans:
                        ctx.add_trace("replan", "Max replans exceeded; aborting.")
                        break
                    ctx.add_trace("replan", "Verification failed twice; replanning.")
                    if not await make_plan():
                        break

        notes = "\n\n".join(ctx.session_state.get("intermediate", []))
        synth_result = await run_agent(
            synthesizer,
            f"Question: {ctx.message}\nNotes:\n{notes}\nFinal answer:",
            ctx=ctx,
            step_name="synthesize",
        )
        content = synth_result.content or notes or "No verified results."
        ctx.add_trace("synthesize", content[:400])
        return RunResult(content=content, trace=ctx.trace, session_state=ctx.session_state)
