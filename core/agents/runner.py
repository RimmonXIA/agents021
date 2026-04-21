"""
Shared async execution for Agno agents: retries, structured output parsing, optional hooks.
"""
from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Callable
from typing import Any

from agno.agent import Agent
from pydantic import BaseModel

from core.config import settings
from core.utils.logging import get_logger

logger = get_logger(__name__)


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
) -> Any:
    """
    Run agent.run in a thread pool with retries and optional structured parsing.

    On total failure: returns the last exception (default) or None if return_none_on_failure.
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
                return parse_structured_response(content, response_model)
            return content
        except Exception as e:
            last_error = e
            if on_failed_attempt:
                on_failed_attempt(agent, current_attempt, e)
            else:
                logger.error(f"{agent.name} attempt {current_attempt} failed: {e}")
            if current_attempt >= max_retries:
                break
    if return_none_on_failure:
        return None
    return last_error
