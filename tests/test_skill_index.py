from core.config import settings
from core.memory.skill_index import SkillIndex
from core.models import Skill


def test_skill_index_dedup_prefers_higher_quality(temp_lancedb: str) -> None:
    index = SkillIndex(temp_lancedb)
    low = Skill(
        title="Data Validation SOP",
        description="Use before writing reports.",
        content_markdown="1. Check source",
        quality_score=0.3,
    )
    high = Skill(
        title="Data Validation SOP",
        description="Use before writing reports.",
        content_markdown="1. Check source\n2. Cross-verify",
        quality_score=0.9,
    )

    old_flag = settings.skill_dedup_v2
    settings.skill_dedup_v2 = True
    try:
        index.persist_skill(low)
        index.persist_skill(high)
    finally:
        settings.skill_dedup_v2 = old_flag

    skills = index.fetch_for_intent("validate report sources")
    assert skills
    assert skills[0]["title"] == "Data Validation SOP"


def test_skill_index_retrieval_includes_quality_and_tags(temp_lancedb: str) -> None:
    index = SkillIndex(temp_lancedb)
    index.persist_skill(
        Skill(
            title="API Fallback SOP",
            description="Use when primary provider is unstable.",
            content_markdown="Fallback to secondary provider and log incident.",
            tags=["resilience", "fallback"],
            quality_score=0.8,
        )
    )

    results = index.fetch_for_intent("fallback when api provider fails")
    assert results
    top = results[0]
    assert "quality_score" in top
    assert "tags" in top


def test_skill_index_hard_stage_filters_non_active_records(temp_lancedb: str) -> None:
    index = SkillIndex(temp_lancedb)
    index.persist_skill(
        Skill(
            title="Archived SOP",
            description="old",
            content_markdown="legacy",
            quality_score=0.9,
            status="deprecated",
        )
    )
    index.persist_skill(
        Skill(
            title="Active SOP",
            description="new",
            content_markdown="active steps",
            quality_score=0.7,
            status="active",
        )
    )
    results = index.retrieve_reflect_answer("active steps", stage="hard")
    assert results
    assert results[0]["title"] == "Active SOP"


def test_skill_index_hard_write_gate_rejects_missing_evidence(temp_lancedb: str) -> None:
    index = SkillIndex(temp_lancedb)
    old_stage = settings.skill_write_gate_stage
    settings.skill_write_gate_stage = "hard"
    try:
        index.persist_skill(
            Skill(
                title="No Evidence SOP",
                description="missing grounding",
                content_markdown="steps",
                quality_score=0.9,
                evidence_step_ids=[],
            )
        )
    finally:
        settings.skill_write_gate_stage = old_stage
    results = index.fetch_for_intent("missing grounding")
    assert not any(r.get("title") == "No Evidence SOP" for r in results)


def test_skill_index_returns_gate_rationale_and_provenance(temp_lancedb: str) -> None:
    index = SkillIndex(temp_lancedb)
    index.persist_skill(
        Skill(
            title="Audited SOP",
            description="auditable",
            content_markdown="audited content",
            quality_score=0.8,
            evidence_step_ids=[1],
            gate_decision="accept",
            gate_rationale="evidence-backed",
            provenance={"source": "unit-test"},
        )
    )
    results = index.fetch_for_intent("auditable")
    assert results
    top = results[0]
    assert top.get("gate_rationale")
    assert top.get("provenance")
