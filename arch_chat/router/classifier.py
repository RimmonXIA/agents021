"""Heuristic pre-filter for architecture routing."""

from __future__ import annotations

SIDE_EFFECT_VERBS = ("publish", "post", "send", "delete", "deploy", "commit", "tweet")
REMEMBER_VERBS = ("remember", "recall", "my preference", "note that", "i am allergic")
PUZZLE_KEYWORDS = ("wolf", "goat", "cabbage", "river crossing", "puzzle")
GRID_KEYWORDS = ("grid", "pathfinding", "cellular automata", "maze", "distance field")
RISK_KEYWORDS = (
    "prescription",
    "dosage",
    "legal advice",
    "invest all",
    "emergency",
    "diagnose",
)
CODE_KEYWORDS = ("python", "function", "code", "implement", "refactor", "debug")
CODING_VERBS = ("write", "create", "fix", "sort", "build", "generate", "develop", "program")
RESEARCH_KEYWORDS = ("compare", "analyze", "research", "investigate", "report")
HIGH_STAKES = ("should i invest", "recommend", "advise", "judgment", "decision")
TRADE_KEYWORDS = ("buy", "sell", "trade", "portfolio", "stock")
MULTI_DOMAIN = ("investment memo", "full report", "comprehensive analysis")
GENERAL_CHAT_PATTERNS = (
    "who are you",
    "what are you",
    "hello",
    "hi",
    "hey",
    "help",
    "what can you do",
    "introduce yourself",
)
SPECIALIZED_ARCHS = (
    "tot",
    "cellular_automata",
    "dry_run",
    "pev",
    "mental_loop",
    "graph_memory",
    "ensemble",
    "blackboard",
)


def is_conversational(message: str) -> bool:
    lower = message.lower().strip()
    if any(p in lower for p in GENERAL_CHAT_PATTERNS):
        return True
    if len(lower.split()) <= 6 and not _has_specialized_intent(message):
        return True
    return False


def _has_specialized_intent(message: str) -> bool:
    lower = message.lower()
    if any(v in lower for v in SIDE_EFFECT_VERBS + REMEMBER_VERBS + PUZZLE_KEYWORDS):
        return True
    if any(v in lower for v in GRID_KEYWORDS + RISK_KEYWORDS + TRADE_KEYWORDS):
        return True
    if any(v in lower for v in HIGH_STAKES + MULTI_DOMAIN + RESEARCH_KEYWORDS):
        return True
    if "who works" in lower or " works for " in lower:
        return True
    if has_code_intent(message):
        return True
    return False


def has_code_intent(message: str) -> bool:
    lower = message.lower()
    if "```" in message:
        return True
    has_code_kw = any(k in lower for k in CODE_KEYWORDS)
    has_coding_verb = any(v in lower for v in CODING_VERBS)
    return has_code_kw and has_coding_verb


def _matches_specialized(message: str, arch: str) -> bool:
    lower = message.lower()
    checks = {
        "tot": lambda: any(k in lower for k in PUZZLE_KEYWORDS),
        "cellular_automata": lambda: any(k in lower for k in GRID_KEYWORDS),
        "dry_run": lambda: any(v in lower for v in SIDE_EFFECT_VERBS),
        "pev": lambda: "employee count" in lower or "unreliable" in lower or "verify" in lower,
        "mental_loop": lambda: any(k in lower for k in TRADE_KEYWORDS),
        "graph_memory": lambda: "who works" in lower or "relationship" in lower or " works for " in lower,
        "ensemble": lambda: any(k in lower for k in HIGH_STAKES),
        "blackboard": lambda: any(k in lower for k in MULTI_DOMAIN),
    }
    checker = checks.get(arch)
    return checker() if checker else False


def heuristic_route(message: str) -> str | None:
    lower = message.lower()
    if any(v in lower for v in SIDE_EFFECT_VERBS):
        return "dry_run"
    if any(v in lower for v in REMEMBER_VERBS):
        return "episodic_memory"
    if any(v in lower for v in PUZZLE_KEYWORDS):
        return "tot"
    if any(v in lower for v in GRID_KEYWORDS):
        return "cellular_automata"
    if any(v in lower for v in RISK_KEYWORDS):
        return "metacognitive"
    if any(v in lower for v in TRADE_KEYWORDS):
        return "mental_loop"
    if "who works" in lower or "relationship" in lower or " works for " in lower:
        return "graph_memory"
    if any(v in lower for v in HIGH_STAKES):
        return "ensemble"
    if any(v in lower for v in MULTI_DOMAIN):
        return "blackboard"
    if is_conversational(message):
        return "meta_controller"
    if any(v in lower for v in CODE_KEYWORDS) and any(
        w in lower for w in ("review", "refine", "improve", "email")
    ):
        return "self_improve"
    if has_code_intent(message):
        return "reflection"
    if "employee count" in lower or "unreliable" in lower or "verify" in lower:
        return "pev"
    if any(v in lower for v in RESEARCH_KEYWORDS):
        return "planning"
    if "?" in message and len(message.split()) > 8:
        return "react"
    return None
