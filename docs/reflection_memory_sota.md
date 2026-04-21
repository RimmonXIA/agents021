# Reflection-Memory: Why It Exists and What SOTA Looks Like

## Context
Trinity already has an `EvolutionaryOptimizer (EO)` and persistent stores for trajectory and skills. The remaining question is not "whether to store more logs", but "how to turn execution evidence into reliable future decisions."

This note consolidates a meta-level decision framework before implementation.

## Why Reflection-Memory Should Exist in Trinity
- Prevent repeated failure patterns across sessions.
- Convert one-off success traces into reusable procedural knowledge (SOP/skills).
- Improve planning quality under longer and more parallel task graphs.
- Create measurable learning loops instead of static orchestration behavior.

Without reflection-memory, Trinity remains an execution engine. With reflection-memory, Trinity can become an adaptive system.

## What Reflection-Memory Should Be
Reflection-memory should be a closed loop, not a storage bucket:
1. Capture structured evidence from execution.
2. Distill candidate skills or SOPs.
3. Validate faithfulness, novelty, and utility.
4. Archive with provenance and lifecycle metadata.
5. Retrieve with task-aware control.
6. Apply in planning and execution.
7. Measure impact and evolve the memory base.

Design principle: memory is for better future decisions, not for preserving all past text.

## Domain SOTA Snapshot

### Reflexion (verbal RL)
- Core idea: learn by linguistic feedback instead of model weight updates.
- Contribution to Trinity design: introduce episodic reflection buffers tied to explicit feedback signals.
- Reference: [arXiv:2303.11366](https://arxiv.org/abs/2303.11366)

### Generative Agents
- Core idea: retrieval based on recency, relevance, and importance; periodic reflection synthesis.
- Contribution to Trinity design: score-based memory retrieval and trigger-based reflection rather than fixed timing.
- Reference: [arXiv:2304.03442](https://arxiv.org/abs/2304.03442)

### Voyager
- Core idea: growing reusable skill library + iterative self-verification.
- Contribution to Trinity design: skill composability and verifier loops are key to lifelong capability growth.
- Reference: [arXiv:2305.16291](https://arxiv.org/abs/2305.16291)

### MemGPT
- Core idea: hierarchical memory and virtual context management.
- Contribution to Trinity design: separate hot context from archival memory with explicit movement policies.
- Reference: [arXiv:2310.08560](https://arxiv.org/abs/2310.08560)

### MemR3
- Core idea: retrieval is a closed-loop controller (retrieve/reflect/answer router + evidence-gap tracker).
- Contribution to Trinity design: upgrade from open-loop retrieve-then-answer to evidence-driven retrieval control.
- Reference: [arXiv:2512.20237](https://arxiv.org/abs/2512.20237)

### Hermes Agent Memory (industry implementation pattern)
- Core idea: layered memory with bounded always-on memory, session search, and procedural skills/extensions.
- Contribution to Trinity design: practical memory layering and operational constraints (capacity, curation, compatibility).
- Reference: [Hermes memory docs](https://hermes-agent.nousresearch.com/docs/user-guide/features/memory/)

## Implications for Trinity (Quality-First)
- Evidence-linked skills: every accepted skill must map to trajectory evidence.
- Multi-pass EO: candidate -> critic -> quality gate, instead of single-pass extraction.
- Dynamic retrieval control: route between retrieve, reflect, and answer based on evidence gaps.
- Versioned knowledge lifecycle: merge, supersede, and deprecate skills over time.
- Evaluation-first operation: retrieval precision and planner uplift are release gates, not nice-to-have metrics.

## Initial Decision Contract
- North star: increase task success quality with auditable cross-session learning.
- Non-goal: maximizing memory volume without utility proof.
- Release strategy: enable by flags, observe behavior, then tighten gates.

## Architecture Contract for Implementation
- EO input must be a structured `ReflectionPack` with evidence-rich step records.
- Skill persistence must include provenance (`source_session_ids`, `evidence_step_ids`) and lifecycle metadata (`version`, `supersedes`, `status`).
- Multi-pass EO must support candidate, critic, and gate stages with strict schema outputs.
- Retrieval must combine semantic candidate recall with lexical/quality reranking and diversity filtering.

## KPI Contract
- Retrieval quality: `precision@3 >= 0.60`, `recall@3 >= 0.70` on curated intents.
- Skill quality gate: accepted skills must satisfy `quality_score >= 0.65` and non-empty `evidence_step_ids`.
- Planner uplift: fewer dead-end tasks versus baseline (`dead_end_task_rate` decreases by >= 20%).
- Reliability: EO and trajectory persistence tests must be deterministic and passing in CI.

