"""Tests for core.agents.runner structured parsing."""
import json

import pytest
from pydantic import BaseModel

from core.agents.runner import parse_structured_response


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
