"""Dry-run approval prompt."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel

console = Console()


async def request_approval(preview: str, session) -> bool:
    console.print(Panel(preview, title="Dry-Run Preview — Approval Required", border_style="yellow"))
    try:
        answer = await session.prompt_async("Approve commit? [y/N]: ")
    except (KeyboardInterrupt, EOFError):
        return False
    return answer.strip().lower() in ("y", "yes")
