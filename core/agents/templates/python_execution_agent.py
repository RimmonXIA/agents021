from typing import Any

from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.tools.python import PythonTools

from core.config import settings


def get_agent(context: dict[str, Any]) -> Agent:
    """
    Python Executor Agent for running local analysis or code.
    """
    instructions = [
        "You are the Python Executor Sub-agent.",
        "You write and execute Python code to solve analytical or mathematical problems.",
        "Always verify your outputs by printing or returning them from your script.",
        "Be extremely careful not to execute destructive commands."
    ]
    
    if context:
        instructions.append("\n### Context Injected from Blackboard ###")
        for key, value in context.items():
            if value:
                instructions.append(f"{key}: {value}")

    return Agent(
        model=DeepSeek(id=settings.subagent_model),
        name="PythonExecutor",
        description="A sub-agent capable of writing and executing Python scripts locally.",
        instructions=instructions,
        tools=[PythonTools()],
    )
