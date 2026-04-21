from typing import Any

from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.tools.duckduckgo import DuckDuckGoTools

from core.config import settings


def get_agent(context: dict[str, Any]) -> Agent:
    """
    Standard Search Agent instantiated by AS.
    """
    instructions = [
        "You are the Web Search Sub-agent.",
        "Your goal is to gather objective, factual information from public sources.",
        "Focus on academic, governmental, and official news reports.",
        "Summarize the findings neutrally and concisely, including specific version numbers and release dates."
    ]
    
    if context:
        instructions.append("\n### Context Injected from Blackboard ###")
        for key, value in context.items():
            if value:
                instructions.append(f"{key}: {value}")

    return Agent(
        model=DeepSeek(id=settings.subagent_model),
        name="SearchAgent",
        description="A sub-agent capable of performing web searches.",
        instructions=instructions,
        tools=[DuckDuckGoTools()],
    )
