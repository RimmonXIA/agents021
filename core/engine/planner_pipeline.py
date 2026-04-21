"""Streaming planner decomposition into blackboard todos."""
from __future__ import annotations

import asyncio
from typing import Any

from agno.agent import Agent

from core.config import settings
from core.memory.blackboard import Blackboard
from core.models import AtomicTask
from core.utils.json_stream import StreamJSONParser
from core.utils.logging import get_logger

logger = get_logger(__name__)


class PlannerPipeline:
    def __init__(self, blackboard: Blackboard, planner: Agent) -> None:
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
                skills_context += (
                    f"- **{s['title']}**: {s['description']}\n  Content: {s['content']}\n"
                )

        prompt = clock + self.bb.state.original_intent + skills_context

        parser = StreamJSONParser(target_key="tasks")

        max_retries = settings.max_retries
        current_attempt = 0

        while current_attempt < max_retries:
            current_attempt += 1
            try:
                stream = self.planner.arun(prompt, stream=True)
                reasoning_buffer: list[str] = []

                async for chunk in stream:
                    reasoning = (
                        chunk.reasoning_content
                        if (hasattr(chunk, "reasoning_content") and chunk.reasoning_content is not None)
                        else ""
                    )
                    if reasoning:
                        reasoning_buffer.append(reasoning)

                    text = chunk.content if (hasattr(chunk, "content") and chunk.content is not None) else ""
                    if text:
                        for task_dict in parser.feed(text):
                            try:
                                task = AtomicTask.model_validate(task_dict)
                                await self.bb.add_todo(task)
                                logger.info(f"Incremental Task Injection: {task.id}")
                            except Exception as e:
                                logger.error(f"Failed to validate incremental task: {e}")

                if reasoning_buffer:
                    logger.debug(f"[IO_Planner Thinking Process]\n{''.join(reasoning_buffer)}")
                return

            except Exception as e:
                logger.error(f"Streaming decomposition attempt {current_attempt} failed: {e}")
                if current_attempt >= max_retries:
                    await self.bb.mark_failed()
                    break
                await asyncio.sleep(1.0 * current_attempt)
