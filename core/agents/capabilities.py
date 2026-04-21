"""
Registry of capability names ↔ `core.agents.templates.<name>_agent` modules.

Keep planner-facing roles in sync with `planner_agent` instructions via
`planner_capabilities_rule()`.
"""
from __future__ import annotations

from typing import Final

# Sub-agents the planner may assign to tasks (excludes internal roles).
PLANNER_CAPABILITIES: Final[tuple[str, ...]] = (
    "search",
    "python_execution",
    "file_analyzer",
    "developer",
)

# All template modules probed by AgentSynthesizer (includes orchestration/internal roles).
SYNTHESIS_CAPABILITIES: Final[tuple[str, ...]] = PLANNER_CAPABILITIES + (
    "planner",
    "compactor",
)


def planner_capabilities_rule() -> str:
    """Human-readable list for planner system prompts."""
    return ", ".join(f"'{c}'" for c in PLANNER_CAPABILITIES)
