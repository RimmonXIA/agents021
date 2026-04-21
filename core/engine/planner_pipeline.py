"""Streaming planner decomposition into blackboard todos."""
from __future__ import annotations

from typing import Any

from agno.agent import Agent

from core.config import settings
from core.agents.runner import run_agent_stream
from core.engine.ports import StatePort
from core.models import AtomicTask
from core.utils.json_stream import StreamJSONParser
from core.utils.logging import get_logger

logger = get_logger(__name__)


class PlannerPipeline:
    def __init__(self, blackboard: StatePort, planner: Agent) -> None:
        self.bb = blackboard
        self.planner = planner

    async def decompose_intent(self) -> None:
        """
        Uses the planner agent to break down the original intent using a streaming approach.
        Injected tasks are added to the blackboard immediately as they are parsed.
        """
        logger.info(f"Streaming Decomposition for intent: {self.bb.state.original_intent}")

        await self.bb.sync_world_state()
        ws = self.bb.state.world_state
        clock = ""
        if ws:
            clock = (
                f"[CLOCK] Today is {ws.day_of_week}, {ws.current_date} "
                f"(local time {ws.current_time}). Use this as 'today' and 'latest' for decomposition and search.\n\n"
            )

        skills = self.bb.fetch_relevant_skills()
        skills_context = ""
        if skills:
            skills_context = "\n\n### Applicable Skills (SOPs) ###\n"
            for s in skills:
                quality = s.get("quality_score", "")
                quality_text = f" (quality={quality})" if quality else ""
                tags = s.get("tags", "")
                tags_text = f"\n  Tags: {tags}" if tags else ""
                tier = s.get("memory_tier", "")
                tier_text = f"\n  Tier: {tier}" if tier else ""
                skills_context += (
                    f"- **{s['title']}**{quality_text}: {s['description']}\n"
                    f"  Content: {s['content']}{tags_text}{tier_text}\n"
                )

        prompt = clock + self.bb.state.original_intent + skills_context

        parser = StreamJSONParser(target_key="tasks")

        async def _on_chunk(chunk: Any) -> None:
            text = chunk.content if (hasattr(chunk, "content") and chunk.content is not None) else ""
            if not text:
                return
            for task_dict in parser.feed(text):
                try:
                    task = AtomicTask.model_validate(task_dict)
                    await self.bb.add_todo(task)
                    logger.info(f"Incremental Task Injection: {task.id}")
                except Exception as e:
                    logger.error(f"Failed to validate incremental task: {e}")

        result = await run_agent_stream(
            self.planner,
            prompt,
            max_retries=settings.max_retries,
            on_chunk=_on_chunk,
            on_failed_attempt=lambda _agent, attempt, err: logger.error(
                "Streaming decomposition attempt %s failed: %s", attempt, err
            ),
            on_reasoning=lambda thought: logger.debug(f"[IO_Planner Thinking Process]\n{thought}"),
        )
        if not result.success:
            await self.bb.mark_failed()
