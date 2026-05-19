"""Metacognitive: self-boundary reasoning before acting."""

from __future__ import annotations

from agno.agent import Agent

from arch_chat.models.context import RunContext, RunResult
from arch_chat.models.state import MetacognitiveAnalysis
from arch_chat.runner import run_agent
from arch_chat.workflows.base import ArchitectureRunner

ESSENCE = "Assess confidence and strategy before answering"
NAME = "metacognitive"

AGENT_SELF_MODEL = {
    "knowledge_domains": ["general health", "nutrition", "exercise", "general Q&A"],
    "tools_available": ["symptom_checker"],
    "confidence_threshold": 0.7,
    "high_risk_topics": [
        "prescription dosage",
        "emergency medical advice",
        "legal advice",
        "financial investment guarantee",
    ],
}


class MetacognitiveRunner(ArchitectureRunner):
    name = NAME
    essence = ESSENCE
    number = 15

    async def run(self, ctx: RunContext) -> RunResult:
        meta = Agent(
            name="metacognitive",
            model=ctx.model,
            output_schema=MetacognitiveAnalysis,
            instructions=(
                "Analyze whether you can safely answer. "
                f"Self-model: {AGENT_SELF_MODEL}. "
                "strategy: reason_directly | use_tool | escalate"
            ),
        )
        analysis_result = await run_agent(
            meta,
            f"User request: {ctx.message}",
            response_model=MetacognitiveAnalysis,
        )
        if not analysis_result.success:
            return RunResult(content="Metacognitive analysis failed.", trace=ctx.trace)

        analysis: MetacognitiveAnalysis = analysis_result.parsed
        ctx.session_state["analysis"] = analysis.model_dump()
        ctx.add_trace(
            "self_model",
            f"confidence={analysis.confidence:.2f}, strategy={analysis.strategy}\n{analysis.reasoning}",
            ctx.snapshot_state(),
        )

        if analysis.strategy == "escalate" or analysis.confidence < AGENT_SELF_MODEL["confidence_threshold"]:
            content = (
                f"I should not fully answer this request.\n\n"
                f"Confidence: {analysis.confidence:.2f}\n"
                f"Reasoning: {analysis.reasoning}\n\n"
                "Please consult a qualified professional for this topic."
            )
            return RunResult(content=content, trace=ctx.trace, session_state=ctx.session_state, refused=True)

        answer_agent = Agent(
            name="answer",
            model=ctx.model,
            instructions="Answer carefully within your competence boundaries.",
        )
        answer_result = await run_agent(answer_agent, ctx.message)
        content = answer_result.content or ""
        ctx.add_trace("answer", content[:400])
        return RunResult(content=content, trace=ctx.trace, session_state=ctx.session_state)
