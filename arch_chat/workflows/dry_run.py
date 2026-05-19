"""Dry-Run: side-effect gating with human approval."""

from __future__ import annotations

import re

from agno.agent import Agent

from arch_chat.models.context import RunContext, RunResult
from arch_chat.runner import run_agent
from arch_chat.tools.mock_publish import publish_post
from arch_chat.workflows.base import ArchitectureRunner

ESSENCE = "Preview side effects; commit only after explicit approval"
NAME = "dry_run"


class DryRunRunner(ArchitectureRunner):
    name = NAME
    essence = ESSENCE
    number = 14

    async def run(self, ctx: RunContext) -> RunResult:
        proposer = Agent(
            name="proposer",
            model=ctx.model,
            instructions=(
                "Draft a social media post for the user request. "
                "Return JSON-like content and hashtags list in text: "
                "CONTENT: ... HASHTAGS: tag1, tag2"
            ),
        )
        propose_result = await run_agent(proposer, ctx.message)
        draft = propose_result.content or ctx.message
        ctx.session_state["draft"] = draft
        ctx.add_trace("propose", draft[:400], ctx.snapshot_state())

        content_match = re.search(r"CONTENT:\s*(.+?)(?:HASHTAGS:|$)", draft, re.DOTALL | re.I)
        tags_match = re.search(r"HASHTAGS:\s*(.+)", draft, re.I)
        post_content = (content_match.group(1).strip() if content_match else draft)[:500]
        tags = [t.strip() for t in (tags_match.group(1).split(",") if tags_match else ["ai", "agent"])]

        preview = publish_post(post_content, tags, dry_run=True)
        ctx.session_state["preview"] = preview
        ctx.add_trace("preview", preview)

        approved = False
        if ctx.approval_callback:
            approved = await ctx.approval_callback(preview)
        ctx.session_state["approved"] = approved
        ctx.add_trace("approve", f"approved={approved}")

        if approved:
            live = publish_post(post_content, tags, dry_run=False)
            ctx.add_trace("commit", live)
            content = f"Approved and published.\n\n{live}"
        else:
            content = f"Not approved. Preview only:\n\n{preview}"
        return RunResult(content=content, trace=ctx.trace, session_state=ctx.session_state)
