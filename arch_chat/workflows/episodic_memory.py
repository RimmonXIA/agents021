"""Episodic + Semantic Memory: long-term recall across turns."""

from __future__ import annotations

import asyncio
from pathlib import Path

from agno.agent import Agent

from arch_chat.models.context import RunContext, RunResult
from arch_chat.workflows.base import ArchitectureRunner

ESSENCE = "Episodic + semantic memory integrated into the control flow"
NAME = "episodic_memory"


class EpisodicMemoryRunner(ArchitectureRunner):
    name = NAME
    essence = ESSENCE
    number = 8

    async def run(self, ctx: RunContext) -> RunResult:
        lower = ctx.message.lower()
        remember_verbs = ("remember", "note that", "i am", "i'm", "my preference", "allergic")

        if any(v in lower for v in remember_verbs):
            ctx.episodic_facts.append(ctx.message)
            ctx.add_trace("memory_write", f"Stored episodic fact: {ctx.message[:200]}")
            content = f"Noted. I'll remember: {ctx.message}"
            return RunResult(content=content, trace=ctx.trace, session_state=ctx.session_state)

        memory_context = ""
        if ctx.episodic_facts:
            memory_context = "Known facts about user:\n" + "\n".join(
                f"- {f}" for f in ctx.episodic_facts
            )
        if ctx.semantic_facts:
            memory_context += "\nSemantic knowledge:\n" + "\n".join(
                f"- {f}" for f in ctx.semantic_facts
            )

        agent = Agent(
            name="memorized",
            model=ctx.model,
            instructions=(
                "Assistant with long-term memory. Use recalled facts when answering. "
                "If no relevant memory, say so."
            ),
            markdown=True,
        )
        prompt = ctx.message
        if memory_context:
            prompt = f"{memory_context}\n\nUser: {ctx.message}"
            ctx.add_trace("memory_retrieve", memory_context[:400])

        try:
            from agno.memory.v2.memory import Memory
            from agno.memory.v2.db.sqlite import SqliteMemoryDb

            Path(ctx.settings.data_dir).mkdir(parents=True, exist_ok=True)
            memory = Memory(
                db=SqliteMemoryDb(
                    table_name="user_memories",
                    db_file=ctx.settings.memory_db_path,
                ),
                model=ctx.model,
            )
            agent.memory = memory
            agent.enable_agentic_memory = True
            agent.enable_user_memories = True
            agent.add_history_to_messages = True
            agent.num_history_responses = 5
            ctx.add_trace("episodic", "agno Memory + SqliteMemoryDb attached")
        except Exception as e:
            ctx.add_trace("episodic", f"In-memory fallback (agno memory unavailable: {e})")

        response = await asyncio.to_thread(agent.run, prompt, user_id=ctx.user_id)
        content = str(response.content or "")
        ctx.add_trace("response", content[:400])
        return RunResult(content=content, trace=ctx.trace, session_state=ctx.session_state)
