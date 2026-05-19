"""Session memory facade for /memory command."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class SessionMemory:
    episodic_facts: list[str] = field(default_factory=list)
    semantic_facts: list[str] = field(default_factory=list)

    def format_display(self) -> str:
        lines = ["Episodic facts:"]
        if self.episodic_facts:
            lines.extend(f"  - {f}" for f in self.episodic_facts)
        else:
            lines.append("  (none)")
        lines.append("Semantic facts:")
        if self.semantic_facts:
            lines.extend(f"  - {f}" for f in self.semantic_facts)
        else:
            lines.append("  (none)")
        return "\n".join(lines)

    def clear(self) -> None:
        self.episodic_facts.clear()
        self.semantic_facts.clear()
