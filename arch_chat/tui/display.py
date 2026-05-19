"""TUI display helpers."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from arch_chat.models.context import RunResult, TraceStep
from arch_chat.models.state import RoutingDecision
from arch_chat.registry import list_architectures

console = Console()


def print_banner() -> None:
    table = Table(title="17 Agent Architectures", show_header=True, header_style="bold cyan")
    table.add_column("#", style="dim", width=4)
    table.add_column("Name", style="green")
    table.add_column("Essence", style="white")
    for entry in list_architectures():
        table.add_row(str(entry.number), entry.name, entry.essence)
    console.print(table)
    console.print(
        "[dim]Type a message or /help. Auto-routing enabled; use /arch <name> to override.[/dim]\n"
    )


def print_routing(decision: RoutingDecision, overridden: bool = False) -> None:
    label = "Override" if overridden else "Routed"
    console.print(
        Panel(
            f"[bold]{label}:[/bold] [cyan]{decision.architecture}[/cyan] "
            f"(confidence {decision.confidence:.0%})\n{decision.reasoning}",
            title="Architecture Selection",
            border_style="blue",
        )
    )


def print_arch_header(name: str, essence: str) -> None:
    console.print(f"\n[bold magenta]Active:[/bold magenta] [cyan]{name}[/cyan] — {essence}\n")


def print_trace(steps: list[TraceStep], verbose: bool = True) -> None:
    if not verbose or not steps:
        return
    table = Table(title="Control Flow Trace", show_header=True, header_style="bold yellow")
    table.add_column("Step", style="cyan", width=18)
    table.add_column("Output", style="white")
    for step in steps:
        content = step.content[:300] + ("..." if len(step.content) > 300 else "")
        table.add_row(step.name, content)
    console.print(table)


def print_response(result: RunResult) -> None:
    style = "red" if result.refused else "green"
    title = "Refused" if result.refused else "Assistant"
    console.print(Panel(result.content, title=title, border_style=style))


def print_error(msg: str) -> None:
    console.print(Panel(Text(msg, style="bold red"), title="Error", border_style="red"))
