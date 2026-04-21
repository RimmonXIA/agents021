from typing import Any
from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.tools.file import FileTools
from agno.tools.shell import ShellTools

from core.config import settings

def get_agent(context: dict[str, Any]) -> Agent:
    """
    Developer Agent for code manipulation and system execution.
    Similar to Claude Code, it can read/write files and run shell commands.
    """
    instructions = [
        f"TODAY'S DATE: {context.get('current_date', 'Unknown')}",
        "You are the Developer Sub-agent, a world-class software engineer.",
        "Your goal is to help the user build and maintain their codebase.",
        "You have access to the file system and a shell. Use them wisely.",
        "CONCISENESS: Be extremely concise. Avoid filler words.",
        "REASONING: Think step-by-step. Before making major changes, explain your plan briefly.",
        "DANGEROUS COMMANDS: Always be careful with commands that delete or move files.",
        "STYLE: Follow the project's existing coding style."
    ]
    
    if context:
        instructions.append("\n### Context Injected from Blackboard ###")
        for key, value in context.items():
            if value:
                # Truncate large context values to avoid blowing up the prompt
                val_str = str(value)
                if len(val_str) > 1000:
                    val_str = val_str[:1000] + "... [TRUNCATED]"
                instructions.append(f"{key}: {val_str}")

    return Agent(
        model=DeepSeek(id=settings.subagent_model),
        name="Developer",
        description="A sub-agent capable of code manipulation and shell execution.",
        instructions=instructions,
        tools=[FileTools(), ShellTools()],
    )
