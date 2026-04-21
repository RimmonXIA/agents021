"""
Shared async execution for Agno agents: retries, structured output parsing, optional hooks.
"""
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from agno.agent import Agent
from pydantic import BaseModel

from core.config import settings
from core.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class AgentRunResult:
    """Typed agent execution contract used by executors and EO."""

    success: bool
    content: str | None = None
    parsed: Any = None
    error: Exception | None = None
    attempts: int = 0


def retry_delay_seconds(attempt: int) -> float:
    """Shared linear backoff for planner/agent retries."""
    return 1.0 * attempt


def parse_structured_response(content: Any, response_model: type) -> Any:
    """Parse model output into a Pydantic model (dict, model instance, or JSON string)."""
    if isinstance(content, response_model):
        return content
    if isinstance(content, BaseModel):
        return response_model.model_validate(content.model_dump())
    if isinstance(content, dict):
        return response_model.model_validate(content)
    if isinstance(content, str):
        json_match = re.search(r"```json\s*(.*?)\s*```", content, re.DOTALL) or re.search(
            r"(\{.*\})", content, re.DOTALL
        )
        target_str = json_match.group(1) if json_match else content
        return response_model.model_validate(json.loads(target_str))
    raise ValueError(f"Unsupported content type for structured output: {type(content)}")


async def run_agent(
    agent: Agent,
    prompt: str,
    *,
    response_model: type | None = None,
    max_retries: int | None = None,
    augment_prompt_on_parse_retry: bool = False,
    on_failed_attempt: Callable[[Agent, int, Exception], None] | None = None,
    on_reasoning: Callable[[str], None] | None = None,
    return_none_on_failure: bool = False,
) -> AgentRunResult:
    """
    Run agent.run in a thread pool with retries and optional structured parsing.

    On total failure: returns a typed failure result.
    """
    max_retries = max_retries or settings.max_retries
    current_attempt = 0
    last_error: Exception | None = None
    current_prompt = prompt

    while current_attempt < max_retries:
        current_attempt += 1
        try:
            if current_attempt > 1:
                logger.warning(f"Retry attempt {current_attempt}/{max_retries} for {agent.name}...")
                if (
                    augment_prompt_on_parse_retry
                    and last_error
                    and "parse" in str(last_error).lower()
                ):
                    current_prompt = (
                        prompt
                        + "\n\n### ERROR FROM PREVIOUS ATTEMPT ###\n"
                        "Your response was not valid JSON or did not match the schema.\n"
                        f"Error: {last_error}\nPlease try again."
                    )

            response = await asyncio.to_thread(agent.run, current_prompt, output_schema=response_model)
            content = response.content

            if hasattr(response, "reasoning_content") and response.reasoning_content:
                if on_reasoning:
                    on_reasoning(response.reasoning_content)
                else:
                    logger.debug(f"[{agent.name} Thinking Process]\n{response.reasoning_content}")

            if not content:
                raise ValueError("Received empty response from model.")

            if response_model:
                parsed = parse_structured_response(content, response_model)
                return AgentRunResult(
                    success=True,
                    content=str(content),
                    parsed=parsed,
                    attempts=current_attempt,
                )
            return AgentRunResult(success=True, content=str(content), attempts=current_attempt)
        except Exception as e:
            last_error = e
            if on_failed_attempt:
                on_failed_attempt(agent, current_attempt, e)
            else:
                logger.error(f"{agent.name} attempt {current_attempt} failed: {e}")
            if current_attempt >= max_retries:
                break
    if return_none_on_failure:
        return AgentRunResult(success=False, error=None, attempts=current_attempt)
    return AgentRunResult(success=False, error=last_error, attempts=current_attempt)


async def run_agent_stream(
    agent: Agent,
    prompt: str,
    *,
    max_retries: int | None = None,
    on_chunk: Callable[[Any], Awaitable[None]] | None = None,
    on_failed_attempt: Callable[[Agent, int, Exception], None] | None = None,
    on_reasoning: Callable[[str], None] | None = None,
) -> AgentRunResult:
    """
    Run agent.arun(stream=True) with shared retry semantics.

    Returns a typed success/failure result while delegating stream handling via callback.
    """
    max_retries = max_retries or settings.max_retries
    current_attempt = 0
    last_error: Exception | None = None

    while current_attempt < max_retries:
        current_attempt += 1
        try:
            if current_attempt > 1:
                logger.warning(f"Retry attempt {current_attempt}/{max_retries} for {agent.name} (stream)...")
            stream = agent.arun(prompt, stream=True)
            reasoning_buffer: list[str] = []
            content_parts: list[str] = []
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
                    content_parts.append(text)
                if on_chunk:
                    await on_chunk(chunk)
            if reasoning_buffer:
                joined = "".join(reasoning_buffer)
                if on_reasoning:
                    on_reasoning(joined)
                else:
                    logger.debug(f"[{agent.name} Thinking Process]\n{joined}")
            return AgentRunResult(success=True, content="".join(content_parts), attempts=current_attempt)
        except Exception as e:
            last_error = e
            if on_failed_attempt:
                on_failed_attempt(agent, current_attempt, e)
            else:
                logger.error(f"{agent.name} stream attempt {current_attempt} failed: {e}")
            if current_attempt >= max_retries:
                break
            await asyncio.sleep(retry_delay_seconds(current_attempt))
    return AgentRunResult(success=False, error=last_error, attempts=current_attempt)
