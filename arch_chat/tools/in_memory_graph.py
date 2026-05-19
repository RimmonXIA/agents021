"""In-memory knowledge graph — Neo4j substitute for Graph Memory architecture."""

from __future__ import annotations

from dataclasses import dataclass, field

from arch_chat.models.state import KnowledgeGraphExtract, Node, Relationship


@dataclass
class InMemoryGraph:
    relationships: list[Relationship] = field(default_factory=list)

    def ingest(self, extract: KnowledgeGraphExtract) -> int:
        added = 0
        existing = {(r.source.id, r.type, r.target.id) for r in self.relationships}
        for rel in extract.relationships:
            key = (rel.source.id, rel.type, rel.target.id)
            if key not in existing:
                self.relationships.append(rel)
                existing.add(key)
                added += 1
        return added

    def add_text_facts(self, text: str) -> None:
        """Simple heuristic ingest without LLM."""
        lower = text.lower()
        if "works for" in lower or "works at" in lower:
            parts = text.replace(" works for ", "|").replace(" works at ", "|").split("|")
            if len(parts) == 2:
                self.relationships.append(
                    Relationship(
                        source=Node(id=parts[0].strip(), type="Person"),
                        target=Node(id=parts[1].strip(), type="Company"),
                        type="WORKS_FOR",
                    )
                )

    def query(self, subject: str | None = None, rel_type: str | None = None) -> list[Relationship]:
        results = []
        for r in self.relationships:
            if subject and subject.lower() not in r.source.id.lower() and subject.lower() not in r.target.id.lower():
                continue
            if rel_type and r.type != rel_type:
                continue
            results.append(r)
        return results

    def format_dump(self) -> str:
        if not self.relationships:
            return "(empty graph)"
        lines = []
        for r in self.relationships:
            lines.append(f"({r.source.id}:{r.source.type})-[:{r.type}]->({r.target.id}:{r.target.type})")
        return "\n".join(lines)

    def answer_query(self, question: str) -> str:
        q = question.lower()
        if "who works" in q or "works for" in q:
            matches = self.query(rel_type="WORKS_FOR")
            if not matches:
                return "No WORKS_FOR relationships found."
            return "\n".join(f"{r.source.id} WORKS_FOR {r.target.id}" for r in matches)
        matches = self.query()
        if not matches:
            return "Graph is empty."
        return self.format_dump()
