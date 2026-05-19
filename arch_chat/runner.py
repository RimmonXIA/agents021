from __future__ import annotations

import asyncio
import json
import re
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from agno.agent import Agent
from pydantic import BaseModel

if TYPE_CHECKING:
    from arch_chat.models.context import RunContext


@dataclass(frozen=True)
class AgentRunResult:
    success: bool
    content: str | None = None
    parsed: Any = None
    error: Exception | None = None
    attempts: int = 0


def parse_structured_response(content: Any, response_model: type) -> Any:
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


async def _run_agent_blocking(
    agent: Agent,
    prompt: str,
    *,
    response_model: type | None = None,
) -> AgentRunResult:
    response = await asyncio.to_thread(agent.run, prompt, output_schema=response_model)
    content = response.content
    if not content:
        raise ValueError("Empty response from model.")
    if response_model:
        parsed = parse_structured_response(content, response_model)
        return AgentRunResult(success=True, content=str(content), parsed=parsed)
    return AgentRunResult(success=True, content=str(content))


async def run_agent_stream(
    agent: Agent,
    prompt: str,
    *,
    max_retries: int = 3,
    on_chunk: Callable[[Any], Awaitable[None]] | None = None,
    on_reasoning: Callable[[str], None] | None = None,
) -> AgentRunResult:
    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            stream = agent.arun(prompt, stream=True)
            reasoning_buffer: list[str] = []
            content_parts: list[str] = []
            async for chunk in stream:
                reasoning = (
                    chunk.reasoning_content
                    if hasattr(chunk, "reasoning_content") and chunk.reasoning_content is not None
                    else ""
                )
                if reasoning:
                    reasoning_buffer.append(reasoning)
                    if on_reasoning:
                        on_reasoning(reasoning)
                text = chunk.content if hasattr(chunk, "content") and chunk.content is not None else ""
                if text:
                    content_parts.append(text)
                if on_chunk:
                    await on_chunk(chunk)
            return AgentRunResult(success=True, content="".join(content_parts), attempts=attempt)
        except Exception as e:
            last_error = e
            if attempt >= max_retries:
                break
            await asyncio.sleep(float(attempt))
    return AgentRunResult(success=False, error=last_error, attempts=max_retries)


async def run_agent(
    agent: Agent,
    prompt: str,
    *,
    ctx: RunContext | None = None,
    stream: Any | None = None,
    step_name: str | None = None,
    response_model: type | None = None,
    max_retries: int = 3,
) -> AgentRunResult:
    name = step_name or getattr(agent, "name", None) or "agent"
    sink = (ctx.stream if ctx else None) or stream

    if sink:
        sink.step_start(name, getattr(agent, "name", "") or "")

    last_error: Exception | None = None
    for attempt in range(1, max_retries + 1):
        try:
            if response_model is not None:
                if sink:
                    sink.spinner(f"{name} thinking...")
                result = await _run_agent_blocking(agent, prompt, response_model=response_model)
            elif sink:
                async def on_chunk(chunk: Any) -> None:
                    text = chunk.content if hasattr(chunk, "content") and chunk.content else ""
                    if text:
                        sink.token(text)

                def on_reasoning(text: str) -> None:
                    sink.reasoning(text)

                result = await run_agent_stream(
                    agent,
                    prompt,
                    max_retries=1,
                    on_chunk=on_chunk,
                    on_reasoning=on_reasoning,
                )
            else:
                result = await _run_agent_blocking(agent, prompt, response_model=response_model)

            if not result.success:
                raise result.error or ValueError("Agent run failed")

            return result
        except Exception as e:
            last_error = e
            if attempt >= max_retries:
                break
            await asyncio.sleep(float(attempt))

    if sink:
        sink.step_end(name, f"failed: {last_error}")
    return AgentRunResult(success=False, error=last_error, attempts=max_retries)
