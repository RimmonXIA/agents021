"""Self-Improvement: iterative generate-critic loop until approved."""

from __future__ import annotations

from agno.agent import Agent

from arch_chat.models.context import RunContext, RunResult
from arch_chat.models.state import EmailCritique, EmailDraft
from arch_chat.runner import run_agent
from arch_chat.workflows.base import ArchitectureRunner

ESSENCE = "Generate → critique → revise loop until quality gate passes"
NAME = "self_improve"


class SelfImproveRunner(ArchitectureRunner):
    name = NAME
    essence = ESSENCE
    number = 16

    async def run(self, ctx: RunContext) -> RunResult:
        generator = Agent(
            name="gen",
            model=ctx.model,
            output_schema=EmailDraft,
            instructions="Write a professional email draft for the user request.",
        )
        critic = Agent(
            name="critic",
            model=ctx.model,
            output_schema=EmailCritique,
            instructions=(
                "Review email for clarity, tone, and completeness. "
                "Approve only if production-ready."
            ),
        )

        ctx.session_state["revision"] = 0
        max_revisions = 3
        last_draft: EmailDraft | None = None
        feedback = ""

        while ctx.session_state["revision"] <= max_revisions:
            gen_prompt = ctx.message
            if feedback:
                gen_prompt += f"\n\nPrevious feedback:\n{feedback}"
            gen_result = await run_agent(generator, gen_prompt, response_model=EmailDraft)
            if not gen_result.success:
                break
            last_draft = gen_result.parsed
            ctx.session_state["last_email"] = last_draft.model_dump()
            ctx.add_trace(
                f"gen_{ctx.session_state['revision']}",
                f"Subject: {last_draft.subject}\n{last_draft.body[:200]}",
                ctx.snapshot_state(),
            )

            crit_result = await run_agent(
                critic,
                f"Subject: {last_draft.subject}\n\n{last_draft.body}",
                response_model=EmailCritique,
            )
            if not crit_result.success:
                break
            critique: EmailCritique = crit_result.parsed
            ctx.session_state["last_critique"] = critique.model_dump()
            ctx.add_trace(
                f"critic_{ctx.session_state['revision']}",
                f"approved={critique.is_approved}: {critique.feedback}",
            )

            if critique.is_approved:
                break
            ctx.session_state["revision"] += 1
            feedback = critique.feedback
            if ctx.session_state["revision"] > max_revisions:
                ctx.add_trace("stop", "Max revisions reached")
                break

        if last_draft:
            content = (
                f"Subject: {last_draft.subject}\n\n{last_draft.body}\n\n"
                f"(Revisions: {ctx.session_state['revision']})"
            )
        else:
            content = "Self-improvement loop failed to produce a draft."
        return RunResult(content=content, trace=ctx.trace, session_state=ctx.session_state)
