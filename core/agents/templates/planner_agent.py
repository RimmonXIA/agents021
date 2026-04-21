import json
from typing import Any

from agno.agent import Agent
from agno.models.deepseek import DeepSeek

from core.agents.capabilities import planner_capabilities_rule
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
            f"2. CAPABILITIES: You can only use {planner_capabilities_rule()}.",
            "3. SIMPLE INTENTS: If the user intent is a single direct action (e.g. 'read main.py', 'list files'), return a plan with exactly ONE task using the 'developer' or 'file_analyzer' capability.",
            "4. DEPENDENCIES: 'depends_on' MUST be a list of task ID strings (e.g., ['task_1']). Use [] if none.",
            "5. IDs: All task 'id' values MUST be strings (e.g., 'task_1', 'task_2'), never integers.",
            "6. OUTPUT: You MUST return a valid JSON object matching the JSON Schema below EXACTLY.",
            "7. NO MARKDOWN: Do not wrap your response in markdown code blocks. Start with '{' and end with '}'.",
            "8. NO CHAT: Provide ONLY the JSON. No conversational filler.",
            "9. KEY CONTRACT: Every task 'task_N' automatically produces a blackboard key named 'task_N_result' upon success. Use these keys in 'required_keys' for downstream tasks.",
            f"REQUIRED JSON SCHEMA:\n{schema}",
            f"CONTEXT FROM BLACKBOARD: {context.get('additional_instructions', 'None')}"
        ],
    )
