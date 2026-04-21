"""Tests for core.agents.runner structured parsing."""
import json
from types import SimpleNamespace

import pytest
from pydantic import BaseModel

from core.agents.runner import parse_structured_response, retry_delay_seconds, run_agent, run_agent_stream


class _Sample(BaseModel):
    x: int


def test_parse_structured_response_from_dict() -> None:
    out = parse_structured_response({"x": 1}, _Sample)
    assert out.x == 1


def test_parse_structured_response_from_json_block() -> None:
    text = '```json\n{"x": 42}\n```'
    out = parse_structured_response(text, _Sample)
    assert out.x == 42


def test_parse_structured_response_raw_braces() -> None:
    text = '{"x": 7}'
    out = parse_structured_response(text, _Sample)
    assert out.x == 7


def test_parse_structured_response_invalid_json() -> None:
    with pytest.raises(json.JSONDecodeError):
        parse_structured_response("not json", _Sample)


@pytest.mark.asyncio
async def test_run_agent_returns_typed_failure_result() -> None:
    class _FailingAgent:
        name = "FailingAgent"

        def run(self, prompt: str, output_schema: type | None = None) -> SimpleNamespace:
            raise RuntimeError("boom")

    result = await run_agent(_FailingAgent(), "hi", max_retries=2)  # type: ignore[arg-type]
    assert result.success is False
    assert isinstance(result.error, RuntimeError)
    assert result.attempts == 2


def test_retry_delay_seconds() -> None:
    assert retry_delay_seconds(1) == 1.0
    assert retry_delay_seconds(3) == 3.0


@pytest.mark.asyncio
async def test_run_agent_stream_retries_and_succeeds() -> None:
    class _Chunk:
        def __init__(self, content: str) -> None:
            self.content = content
            self.reasoning_content = ""

    class _FlakyStreamAgent:
        name = "FlakyStream"

        def __init__(self) -> None:
            self.calls = 0

        async def arun(self, prompt: str, stream: bool = True):  # noqa: ANN001
            del prompt, stream
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("first fail")
            yield _Chunk("ok")

    pieces: list[str] = []

    async def on_chunk(chunk):  # noqa: ANN001
        pieces.append(chunk.content)

    result = await run_agent_stream(_FlakyStreamAgent(), "hello", on_chunk=on_chunk, max_retries=2)  # type: ignore[arg-type]
    assert result.success is True
    assert pieces == ["ok"]
