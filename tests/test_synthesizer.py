import pytest
from agno.agent import Agent

from core.agents.synthesizer import AgentSynthesizer


def test_synthesizer_load_search_agent() -> None:
    """Test that the synthesizer can load a valid agent template."""
    asynth = AgentSynthesizer()
    agent = asynth.synthesize("search", {"query": "test"})
    
    assert isinstance(agent, Agent)
    assert agent.name == "SearchAgent"

def test_synthesizer_invalid_capability() -> None:
    """Test that the synthesizer raises a ValueError for unknown capabilities."""
    asynth = AgentSynthesizer()
    with pytest.raises(ValueError, match="is not supported"):
        asynth.synthesize("non_existent_capability", {})

def test_synthesizer_context_injection() -> None:
    """Test that context is correctly passed to the agent instructions."""
    asynth = AgentSynthesizer()
    context = {"special_data": "secret_code"}
    agent = asynth.synthesize("search", context)
    
    # Check if context keys exist in instructions
    instr_str = str(agent.instructions)
    assert "special_data" in instr_str
    assert "secret_code" in instr_str
