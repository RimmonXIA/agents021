from typing import Any

from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.tools.file import FileTools

from core.config import settings


def get_agent(context: dict[str, Any]) -> Agent:
    """
    File Analyzer Agent for reading files and codebases.
    """
    instructions = [
        "You are the File Analyzer Sub-agent.",
        "You navigate and read local file systems to answer questions about codebases or documents.",
        "Use the tools to read files accurately."
    ]
    
    if context:
        instructions.append("\n### Context Injected from Blackboard ###")
        for key, value in context.items():
            if value:
                instructions.append(f"{key}: {value}")

    return Agent(
        model=DeepSeek(id=settings.subagent_model),
        name="FileAnalyzer",
        description="A sub-agent capable of reading local files.",
        instructions=instructions,
        tools=[FileTools()],
    )
