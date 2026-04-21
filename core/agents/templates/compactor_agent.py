from typing import Any
from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from core.config import settings

def get_agent(context: dict[str, Any]) -> Agent:
    """
    Compactor Agent for distilling large volumes of context into dense summaries.
    """
    instructions = [
        "You are the Memory Compactor for the Trinity System.",
        "Your goal is to take a set of raw task results and distill them into a concise, high-density technical summary.",
        "RULES:",
        "1. PRESERVE TECHNICAL DETAILS: Do not lose key findings, file paths, or specific errors.",
        "2. REMOVE VERBOSITY: Eliminate conversational filler and redundant explanations.",
        "3. FORMAT: Use a structured markdown list of key outcomes.",
        "4. IDENTIFY DEPENDENCIES: Note if any result is critical for future tasks."
    ]
    
    return Agent(
        model=DeepSeek(id=settings.subagent_model),
        name="Compactor",
        description="A specialized agent for context distillation and memory management.",
        instructions=instructions,
    )
