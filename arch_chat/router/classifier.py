"""Shared intent helpers used by workflows (not for routing)."""

from __future__ import annotations

CODE_KEYWORDS = ("python", "function", "code", "implement", "refactor", "debug")
CODING_VERBS = ("write", "create", "fix", "sort", "build", "generate", "develop", "program")
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
RISK_KEYWORDS = (
    "prescription",
    "dosage",
    "legal advice",
    "invest all",
    "emergency",
    "diagnose",
)


def is_conversational(message: str) -> bool:
    lower = message.lower().strip()
    if any(p in lower for p in GENERAL_CHAT_PATTERNS):
        return True
    return len(lower.split()) <= 6 and not _has_specialized_intent(message)


def _has_specialized_intent(message: str) -> bool:
    lower = message.lower()
    specialized_markers = (
        "publish",
        "post",
        "remember",
        "wolf",
        "goat",
        "grid",
        "pathfinding",
        "trade",
        "portfolio",
        "works for",
        "python",
        "function",
        "compare",
        "analyze",
        "prescription",
        "puzzle",
    )
    return any(m in lower for m in specialized_markers) or has_code_intent(message)


def has_code_intent(message: str) -> bool:
    lower = message.lower()
    if "```" in message:
        return True
    has_code_kw = any(k in lower for k in CODE_KEYWORDS)
    has_coding_verb = any(v in lower for v in CODING_VERBS)
    return has_code_kw and has_coding_verb
