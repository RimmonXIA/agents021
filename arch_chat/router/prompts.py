"""LLM router prompts — architecture selection via structured LLM call."""

from __future__ import annotations

from arch_chat.registry import ARCH_REGISTRY, list_architectures

# Per-architecture routing hints for the classifier (from docs/agent_arch.md)
ARCH_ROUTING_HINTS: dict[str, str] = {
    "reflection": "Explicit code write/review/refine; generator→critic→refiner pipeline",
    "tool_use": "Simple factual lookup needing one tool call (price, lookup)",
    "react": "Multi-step research; repeated tool use driven by observations",
    "planning": "Task needs ordered sub-steps decomposed upfront",
    "pev": "Tool results may fail; need verify-and-replan loop",
    "multi_agent": "Fixed multi-role report (news + technical + financial + write)",
    "blackboard": "Open-ended task; dynamic specialist order via shared workspace",
    "episodic_memory": "Remember/recall user preferences or past facts across turns",
    "graph_memory": "Relationship queries (who works for, entity links)",
    "tot": "Puzzles, constrained search, branching solution paths",
    "mental_loop": "Costly actions (trade, deploy); simulate before execute",
    "meta_controller": "Greetings, identity, general Q&A, ambiguous or short chat",
    "ensemble": "High-stakes judgment needing multiple perspectives on same question",
    "dry_run": "Side-effect actions: publish, post, send, delete, deploy",
    "metacognitive": "Medical/legal/financial risk; may need refusal or escalation",
    "self_improve": "Quality-critical writing with iterative critique loop",
    "cellular_automata": "Grid pathfinding, cellular automata, distance-field demos",
}


def build_architecture_catalog() -> str:
    lines = []
    for entry in list_architectures():
        hint = ARCH_ROUTING_HINTS.get(entry.name, entry.essence)
        lines.append(f"  {entry.number:02d}. {entry.name}: {hint}")
    return "\n".join(lines)


def build_router_system_prompt() -> str:
    valid_names = ", ".join(sorted(ARCH_REGISTRY.keys()))
    catalog = build_architecture_catalog()
    return f"""You are the architecture router for an agent control-flow demo system.

Your job: read the user message and pick exactly ONE architecture name from the allowed list.
You must output JSON matching the RoutingDecision schema: architecture, confidence (0.0-1.0), reasoning.

## Allowed architecture names (use exactly one)
{valid_names}

## Architecture catalog
{catalog}

## Routing rules
1. meta_controller — default for greetings, "who are you", help, general chat, ambiguous messages
2. reflection — ONLY when user explicitly asks to write, implement, refactor, or debug code
3. tool_use — single-step tool lookup (stock price, simple data fetch)
4. react — multi-step research requiring several tool rounds
5. planning — compare/analyze/report tasks needing explicit step decomposition
6. pev — when verification of unreliable tool output matters
7. dry_run — any publish/post/send/delete/deploy side effect
8. episodic_memory — "remember", "recall", user preferences, allergies
9. graph_memory — "who works for", entity relationships
10. metacognitive — medical dosage, legal advice, emergency, high-risk topics
11. tot — wolf/goat/cabbage puzzle, explicit puzzle or search-tree problems
12. cellular_automata — grid, pathfinding, cellular automata demo
13. mental_loop — buy/sell/trade/portfolio with simulation before action
14. ensemble — investment advice, high-stakes recommendations needing multiple views
15. blackboard — comprehensive multi-domain analysis with dynamic ordering
16. multi_agent — structured investment memo with fixed analyst pipeline
17. self_improve — polish email/document with iterative quality loop

## Examples
- "who are you?" → meta_controller (general chat)
- "Write a python function to sort a list" → reflection (code generation)
- "What is Apple's stock price?" → tool_use (simple lookup)
- "Compare AAPL and MSFT revenue and explain the gap" → planning (multi-step)
- "Remember I'm allergic to peanuts" → episodic_memory
- "Publish this tweet about AI" → dry_run (side effect)
- "Solve the wolf goat cabbage puzzle" → tot

When uncertain, choose meta_controller with moderate confidence and explain why.
Never invent architecture names outside the allowed list."""


def build_router_user_prompt(message: str) -> str:
    return f"""User message:
{message}

Select the best architecture. Return architecture (exact name from allowed list), confidence, and brief reasoning."""
