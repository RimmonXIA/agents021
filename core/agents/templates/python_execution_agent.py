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
        "When using Python tools, arguments MUST be valid JSON. Escape newlines and quotes properly.",
        "Keep tool calls compact: avoid passing massive string literals into function arguments.",
        "Always verify your outputs by printing or returning them from your script.",
        "Be extremely careful not to execute destructive commands."
    ]
    
    if context:
        instructions.append("\n### Context Injected from Blackboard ###")
        for key, value in context.items():
            if value:
                val_str = str(value)
                if len(val_str) > 2000:
                    val_str = val_str[:2000] + "... [TRUNCATED]"
                instructions.append(f"{key}: {val_str}")

    return Agent(
        model=DeepSeek(id=settings.subagent_model),
        name="PythonExecutor",
        description="A sub-agent capable of writing and executing Python scripts locally.",
        instructions=instructions,
        tools=[PythonTools()],
    )
