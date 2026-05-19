"""Graph Memory: entity-relationship extraction and graph query."""

from __future__ import annotations

from agno.agent import Agent

from arch_chat.models.context import RunContext, RunResult
from arch_chat.models.state import KnowledgeGraphExtract
from arch_chat.runner import run_agent
from arch_chat.workflows.base import ArchitectureRunner

ESSENCE = "Relational reasoning over an extracted knowledge graph"
NAME = "graph_memory"


class GraphMemoryRunner(ArchitectureRunner):
    name = NAME
    essence = ESSENCE
    number = 9

    async def run(self, ctx: RunContext) -> RunResult:
        lower = ctx.message.lower()
        ingest_markers = (" works for ", " works at ", " acquired ", " founded ")

        if any(m in lower for m in ingest_markers):
            graph_maker = Agent(
                name="graph_maker",
                model=ctx.model,
                output_schema=KnowledgeGraphExtract,
                instructions=(
                    "Extract entities and relationships from text. "
                    "Relationship type should be ALL_CAPS verb."
                ),
            )
            extract_result = await run_agent(
                graph_maker, ctx.message, response_model=KnowledgeGraphExtract
            )
            if extract_result.success:
                added = ctx.graph.ingest(extract_result.parsed)
                ctx.add_trace("graph_ingest", f"Added {added} relationships", ctx.snapshot_state())
            else:
                ctx.graph.add_text_facts(ctx.message)
                ctx.add_trace("graph_ingest", "Heuristic ingest fallback")

        query_agent = Agent(
            name="graph_query",
            model=ctx.model,
            instructions=(
                "Answer questions using the knowledge graph snapshot provided. "
                "Be precise about relationships."
            ),
        )
        graph_snapshot = ctx.graph.format_dump()
        ctx.add_trace("graph_query", f"Graph state:\n{graph_snapshot}")

        prompt = f"Graph:\n{graph_snapshot}\n\nQuestion: {ctx.message}"
        if not ctx.graph.relationships:
            answer = ctx.graph.answer_query(ctx.message)
            if answer != "(empty graph)":
                return RunResult(content=answer, trace=ctx.trace, session_state=ctx.session_state)

        result = await run_agent(query_agent, prompt)
        content = result.content or ctx.graph.answer_query(ctx.message)
        ctx.add_trace("response", content[:400])
        return RunResult(content=content, trace=ctx.trace, session_state=ctx.session_state)
