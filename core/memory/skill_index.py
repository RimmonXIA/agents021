"""LanceDB-backed skill retrieval for planner context."""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import lancedb

from core.config import settings
from core.models import Skill
from core.utils.logging import get_logger

logger = get_logger(__name__)


class SkillIndex:
    """Semantic search over distilled skills (SOPs) for injection into planner prompts."""

    def __init__(self, lancedb_dir: str) -> None:
        Path(lancedb_dir).mkdir(parents=True, exist_ok=True)
        self._lancedb_dir = lancedb_dir
        self.vector_db: Any = lancedb.connect(lancedb_dir)
        self._last_query_debug: dict[str, float] = {}
        self._valid_stages = {"observe", "soft", "hard"}

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {tok for tok in re.split(r"[^a-zA-Z0-9_]+", text.lower()) if tok}

    @staticmethod
    def _jaccard(a: set[str], b: set[str]) -> float:
        if not a and not b:
            return 0.0
        union = a | b
        if not union:
            return 0.0
        return len(a & b) / len(union)

    @staticmethod
    def _normalize_title(title: str) -> str:
        return re.sub(r"\s+", " ", title.strip().lower())

    def _embedding_payload(self, skill: Skill) -> str:
        tags = " ".join(skill.tags)
        return (
            f"title: {skill.title}\n"
            f"description: {skill.description}\n"
            f"tags: {tags}\n"
            f"content:\n{skill.content_markdown}"
        )

    def _content_fingerprint(self, skill: Skill) -> str:
        data = f"{self._normalize_title(skill.title)}::{skill.description.strip()}::{skill.content_markdown.strip()}"
        return hashlib.sha256(data.encode("utf-8")).hexdigest()[:16]

    @staticmethod
    def _all_rows(table: Any) -> list[dict[str, Any]]:
        try:
            df = table.to_pandas()
            return df.to_dict("records")
        except Exception:
            return []

    @staticmethod
    def _table_columns(table: Any) -> set[str]:
        try:
            df = table.to_pandas()
            return set(str(col) for col in df.columns)
        except Exception:
            return set()

    def _adapt_row_to_table_schema(self, table: Any, row: dict[str, Any]) -> dict[str, Any]:
        columns = self._table_columns(table)
        if not columns:
            return row
        adapted = {k: v for k, v in row.items() if k in columns}
        # Ensure minimal required fields remain if old schema is tiny.
        if not adapted:
            adapted = {
                "id": row.get("id", ""),
                "title": row.get("title", ""),
                "description": row.get("description", ""),
                "content_markdown": row.get("content_markdown", ""),
                "text": row.get("text", ""),
            }
        return adapted

    def _append_audit_event(
        self,
        *,
        operation: str,
        skill_id: str,
        title: str,
        decision: str,
        rationale: str,
        metadata: dict[str, Any] | None = None,
    ) -> None:
        event = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "operation": operation,
            "skill_id": skill_id,
            "title": title,
            "decision": decision,
            "rationale": rationale,
            "metadata_json": json.dumps(metadata or {}, ensure_ascii=True),
        }
        try:
            table = self.vector_db.open_table("skill_audit")
            table.add([event])
        except Exception:
            self.vector_db.create_table("skill_audit", data=[event])

    def _is_evidence_backed(self, skill: Skill) -> bool:
        return bool(skill.evidence_step_ids) and skill.quality_score >= settings.eo_min_quality_score

    def _normalized_stage(self, stage: str) -> str:
        if stage in self._valid_stages:
            return stage
        return "observe"

    def _route_memory_tier(self, row: dict[str, Any]) -> str:
        tier = str(row.get("memory_tier", "")).strip().lower()
        if tier in {"hot", "warm", "cold"}:
            return tier
        quality = float(row.get("quality_score", 0.0) or 0.0)
        status = str(row.get("status", "active") or "active")
        if status != "active":
            return "cold"
        if quality >= 0.85:
            return "hot"
        if quality >= 0.50:
            return "warm"
        return "cold"

    def _retrieve_candidates(self, original_intent: str) -> list[dict[str, Any]]:
        table = self.vector_db.open_table("skills")
        try:
            results = table.search(original_intent).limit(30).to_list()
        except Exception:
            results = self._all_rows(table)[:200]
        return results

    def _score_and_rank(self, original_intent: str, rows: list[dict[str, Any]]) -> list[tuple[float, dict[str, str]]]:
        query_terms = self._tokenize(original_intent)
        scored_rows: list[tuple[float, dict[str, str]]] = []
        for rank_idx, r in enumerate(rows):
            title = str(r.get("title", ""))
            description = str(r.get("description", ""))
            content = str(r.get("content_markdown", ""))
            tags = r.get("tags", [])
            if not isinstance(tags, list):
                tags = []
            tier = self._route_memory_tier(r)
            tier_bonus = {"hot": 0.10, "warm": 0.05, "cold": 0.0}.get(tier, 0.0)
            skill_tokens = self._tokenize(f"{title} {description} {content} {' '.join(str(x) for x in tags)}")
            lexical = self._jaccard(query_terms, skill_tokens)
            quality = float(r.get("quality_score", 0.0) or 0.0)
            sem_rank_bonus = 1.0 / (1.0 + rank_idx)
            score = (0.45 * lexical) + (0.30 * quality) + (0.15 * sem_rank_bonus) + tier_bonus
            scored_rows.append(
                (
                    score,
                    {
                        "id": str(r.get("id", "")),
                        "title": title,
                        "description": description,
                        "content": content,
                        "quality_score": str(quality),
                        "tags": ", ".join(str(x) for x in tags),
                        "status": str(r.get("status", "active")),
                        "memory_tier": tier,
                        "gate_decision": str(r.get("gate_decision", "")),
                        "gate_rationale": str(r.get("gate_rationale", "")),
                        "provenance": str(r.get("provenance", {})),
                    },
                )
            )
        return scored_rows

    def _select_diverse(self, scored_rows: list[tuple[float, dict[str, str]]], stage: str) -> list[dict[str, str]]:
        selected: list[dict[str, str]] = []
        selected_token_sets: list[set[str]] = []
        seen_titles: set[str] = set()
        for _, skill in sorted(scored_rows, key=lambda item: item[0], reverse=True):
            normalized_title = self._normalize_title(skill["title"])
            if normalized_title in seen_titles:
                continue
            if stage == "hard" and skill.get("status", "active") != "active":
                continue
            candidate_tokens = self._tokenize(
                f"{skill['title']} {skill['description']} {skill['content']}"
            )
            too_similar = any(self._jaccard(candidate_tokens, existing) > 0.75 for existing in selected_token_sets)
            if too_similar:
                continue
            selected.append(skill)
            selected_token_sets.append(candidate_tokens)
            seen_titles.add(normalized_title)
            if len(selected) >= 3:
                break
        return selected

    def retrieve_reflect_answer(self, original_intent: str, stage: str = "observe") -> list[dict[str, str]]:
        """Planner-facing retrieval router: retrieve -> reflect -> answer."""
        norm_stage = self._normalized_stage(stage)
        logger.info(f"Searching for relevant skills for intent: '{original_intent}'")
        try:
            # retrieve
            retrieved_rows = self._retrieve_candidates(original_intent)
            # reflect
            scored_rows = self._score_and_rank(original_intent, retrieved_rows)
            # answer
            selected = self._select_diverse(scored_rows, stage=norm_stage)

            if selected:
                logger.info(f"Found {len(selected)} relevant skills in memory.")
            else:
                logger.info("No relevant skills found.")
            return selected
        except Exception as e:
            logger.debug(f"Skills table not found or error during retrieval: {e}")
            return []

    def fetch_for_intent(self, original_intent: str) -> list[dict[str, str]]:
        stage = settings.normalized_memory_router_stage if settings.memory_tiering_enabled else "observe"
        return self.retrieve_reflect_answer(original_intent, stage=stage)

    def persist_skill(self, skill: Skill) -> None:
        """Persist a distilled skill for future semantic retrieval."""
        stage = self._normalized_stage(settings.skill_write_gate_stage)
        eligible = self._is_evidence_backed(skill)
        if not eligible and stage == "hard":
            self._append_audit_event(
                operation="write_rejected",
                skill_id=skill.id,
                title=skill.title,
                decision="reject",
                rationale="Rejected by hard write gate: missing evidence or insufficient quality.",
                metadata={"quality_score": skill.quality_score, "evidence_step_ids": skill.evidence_step_ids},
            )
            logger.info("Hard write gate rejected skill '%s'.", skill.title)
            return
        if not eligible and stage == "soft":
            skill.status = "deprecated"
            skill.gate_decision = "revise"
            if not skill.gate_rationale:
                skill.gate_rationale = "Soft gate admitted low-confidence skill as deprecated."
        if eligible and not skill.gate_decision:
            skill.gate_decision = "accept"

        payload = self._embedding_payload(skill)
        fingerprint = self._content_fingerprint(skill)
        data = [
            {
                "id": skill.id,
                "title": skill.title,
                "description": skill.description,
                "content_markdown": skill.content_markdown,
                "tags": skill.tags,
                "source_session_ids": skill.source_session_ids,
                "evidence_step_ids": skill.evidence_step_ids,
                "quality_score": skill.quality_score,
                "version": skill.version,
                "supersedes": skill.supersedes or "",
                "status": skill.status,
                "memory_tier": skill.memory_tier,
                "gate_decision": skill.gate_decision,
                "gate_rationale": skill.gate_rationale,
                "provenance": skill.provenance,
                "fingerprint": fingerprint,
                "text": payload,
            }
        ]
        try:
            table = self.vector_db.open_table("skills")
            if settings.skill_dedup_v2:
                existing = self._all_rows(table)
                duplicate_fingerprint = next(
                    (
                        row
                        for row in existing
                        if str(row.get("fingerprint", "")) == fingerprint
                    ),
                    None,
                )
                if duplicate_fingerprint:
                    self._append_audit_event(
                        operation="dedup_skip",
                        skill_id=skill.id,
                        title=skill.title,
                        decision="skip",
                        rationale="Fingerprint duplicate detected.",
                        metadata={"duplicate_of": str(duplicate_fingerprint.get("id", ""))},
                    )
                    logger.info("Skip persisting fingerprint-duplicate skill '%s'.", skill.title)
                    return

                duplicate_title = next(
                    (
                        row
                        for row in existing
                        if self._normalize_title(str(row.get("title", ""))) == self._normalize_title(skill.title)
                    ),
                    None,
                )
                if duplicate_title:
                    quality_old = float(duplicate_title.get("quality_score", 0.0) or 0.0)
                    if skill.quality_score <= quality_old:
                        self._append_audit_event(
                            operation="merge_skip",
                            skill_id=skill.id,
                            title=skill.title,
                            decision="skip",
                            rationale="Lower-quality duplicate title retained existing active skill.",
                            metadata={"duplicate_of": str(duplicate_title.get("id", ""))},
                        )
                        logger.info("Skip persisting duplicate/lower-quality skill '%s'.", skill.title)
                        return
                    merged_tags = sorted(
                        {
                            *(skill.tags or []),
                            *(duplicate_title.get("tags", []) if isinstance(duplicate_title.get("tags", []), list) else []),
                        }
                    )
                    merged_evidence = sorted(
                        {
                            *(skill.evidence_step_ids or []),
                            *(
                                duplicate_title.get("evidence_step_ids", [])
                                if isinstance(duplicate_title.get("evidence_step_ids", []), list)
                                else []
                            ),
                        }
                    )
                    skill.tags = merged_tags
                    skill.evidence_step_ids = [int(x) for x in merged_evidence]
                    skill.version = int(duplicate_title.get("version", 1) or 1) + 1
                    skill.supersedes = str(duplicate_title.get("id", ""))
                    data[0]["version"] = skill.version
                    data[0]["supersedes"] = skill.supersedes or ""
                    data[0]["tags"] = skill.tags
                    data[0]["evidence_step_ids"] = skill.evidence_step_ids
                    self._append_audit_event(
                        operation="supersede_merge",
                        skill_id=skill.id,
                        title=skill.title,
                        decision="accept",
                        rationale="Higher-quality duplicate title superseded prior version with merged evidence.",
                        metadata={"supersedes": skill.supersedes, "new_version": skill.version},
                    )
            if skill.quality_score < settings.skill_deprecate_threshold and skill.status == "active":
                skill.status = "deprecated"
                data[0]["status"] = "deprecated"
                self._append_audit_event(
                    operation="deprecate",
                    skill_id=skill.id,
                    title=skill.title,
                    decision="deprecate",
                    rationale="Quality score below deprecation threshold.",
                    metadata={"quality_score": skill.quality_score},
                )
            data_to_add = [self._adapt_row_to_table_schema(table, data[0])]
            table.add(data_to_add)
            self._append_audit_event(
                operation="persist",
                skill_id=skill.id,
                title=skill.title,
                decision=skill.gate_decision or "observe",
                rationale=skill.gate_rationale or "Persisted skill entry.",
                metadata={
                    "status": skill.status,
                    "quality_score": skill.quality_score,
                    "evidence_step_ids": skill.evidence_step_ids,
                },
            )
            logger.info("Skill '%s' persisted to existing table.", skill.title)
        except Exception as e:
            try:
                self.vector_db.create_table("skills", data=data)
                self._append_audit_event(
                    operation="persist",
                    skill_id=skill.id,
                    title=skill.title,
                    decision=skill.gate_decision or "observe",
                    rationale=skill.gate_rationale or "Persisted skill entry.",
                    metadata={
                        "status": skill.status,
                        "quality_score": skill.quality_score,
                        "evidence_step_ids": skill.evidence_step_ids,
                    },
                )
                logger.info("Skill '%s' persisted to new table.", skill.title)
            except Exception:
                logger.error("Failed to persist skill '%s': %s", skill.title, e)
