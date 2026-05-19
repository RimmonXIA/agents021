"""Reflection: generator → critic → refiner linear workflow."""

from __future__ import annotations

from agno.agent import Agent

from arch_chat.models.context import RunContext, RunResult
from arch_chat.models.state import Critique, DraftCode, RefinedCode
from arch_chat.router.classifier import has_code_intent
from arch_chat.runner import run_agent
from arch_chat.workflows.base import ArchitectureRunner

ESSENCE = "Generate → critique → refine in three explicit passes"
NAME = "reflection"


async def _generalist_fallback(ctx: RunContext) -> RunResult:
    generalist = Agent(
        name="generalist",
        model=ctx.model,
        instructions="General Q&A assistant. Answer clearly and concisely.",
    )
    ctx.add_trace("reflection_skip", "No code intent; delegating to generalist")
    result = await run_agent(generalist, ctx.message, ctx=ctx, step_name="generalist")
    content = result.content or ""
    ctx.add_trace("generalist", content[:500])
    return RunResult(content=content, trace=ctx.trace, session_state=ctx.session_state)


class ReflectionRunner(ArchitectureRunner):
    name = NAME
    essence = ESSENCE
    number = 1

    async def run(self, ctx: RunContext) -> RunResult:
        if not has_code_intent(ctx.message):
            return await _generalist_fallback(ctx)

        model = ctx.model
        generator = Agent(
            name="generator",
            model=model,
            output_schema=DraftCode,
            instructions="Expert Python programmer. Write code and brief explanation.",
        )
        critic = Agent(
            name="critic",
            model=model,
            output_schema=Critique,
            instructions="Senior code reviewer. Analyze bugs, inefficiencies, PEP8.",
        )
        refiner = Agent(
            name="refiner",
            model=model,
            output_schema=RefinedCode,
            instructions="Rewrite code incorporating every critique suggestion.",
        )

        gen_result = await run_agent(
            generator, ctx.message, ctx=ctx, step_name="generator", response_model=DraftCode
        )
        if not gen_result.success:
            return RunResult(content=f"Generator failed: {gen_result.error}", trace=ctx.trace)
        draft: DraftCode = gen_result.parsed
        ctx.session_state["draft"] = draft.model_dump()
        ctx.add_trace("generator", f"{draft.explanation}\n```python\n{draft.code}\n```", ctx.snapshot_state())

        crit_result = await run_agent(
            critic,
            f"Review this code:\n```python\n{draft.code}\n```",
            ctx=ctx,
            step_name="critic",
            response_model=Critique,
        )
        if not crit_result.success:
            return RunResult(content=f"Critic failed: {crit_result.error}", trace=ctx.trace)
        critique: Critique = crit_result.parsed
        ctx.session_state["critique"] = critique.model_dump()
        ctx.add_trace("critic", critique.critique_summary, ctx.snapshot_state())

        ref_prompt = (
            f"Original code:\n```python\n{draft.code}\n```\n"
            f"Critique: {critique.model_dump_json(indent=2)}\nProduce refined code."
        )
        ref_result = await run_agent(
            refiner, ref_prompt, ctx=ctx, step_name="refiner", response_model=RefinedCode
        )
        if not ref_result.success:
            return RunResult(content=f"Refiner failed: {ref_result.error}", trace=ctx.trace)
        refined: RefinedCode = ref_result.parsed
        ctx.session_state["refined"] = refined.model_dump()
        ctx.add_trace("refiner", refined.refinement_summary, ctx.snapshot_state())

        content = f"{refined.refinement_summary}\n\n```python\n{refined.refined_code}\n```"
        if ctx.stream:
            ctx.stream.token(content)
        return RunResult(content=content, trace=ctx.trace, session_state=ctx.session_state)
