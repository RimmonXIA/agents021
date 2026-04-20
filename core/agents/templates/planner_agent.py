import json
from typing import Any

from agno.agent import Agent
from agno.models.deepseek import DeepSeek

from core.config import settings
from core.models import IOPlan


def get_agent(context: dict[str, Any]) -> Agent:
    """
    The Planner Agent used internally by IO to decompose tasks.
    """
    schema = json.dumps(IOPlan.model_json_schema(), indent=2)
    return Agent(
        model=DeepSeek(id=settings.planner_model),
        name="IO_Planner",
        description="Decomposes user objectives into strict execution plans based on First Principles.",
        instructions=[
            "You are the Cognitive Engine for the Trinity Multi-Agent System.",
            "Your goal is to decompose the user's high-level intent into a set of atomic, executable tasks.",
            "CORE RULES:",
            "1. DECOMPOSITION: Use First Principles to break down the goal into independent steps.",
            "2. CAPABILITIES: You can only use 'search', 'python_execution', or 'file_analyzer'.",
            "3. DEPENDENCIES: 'depends_on' MUST be a list of task ID strings (e.g., ['task_1']). Use [] if none.",
            "4. IDs: All task 'id' values MUST be strings (e.g., 'task_1', 'task_2'), never integers.",
            "5. OUTPUT: You MUST return a valid JSON object matching the JSON Schema below EXACTLY.",
            "6. NO MARKDOWN: Do not wrap your response in markdown code blocks. Start with '{' and end with '}'.",
            "7. NO CHAT: Provide ONLY the JSON. No conversational filler.",
            f"REQUIRED JSON SCHEMA:\n{schema}",
            f"CONTEXT FROM BLACKBOARD: {context.get('additional_instructions', 'None')}"
        ],
    )
