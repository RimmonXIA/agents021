"""Live streaming UI for architecture chat turns."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Protocol

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.text import Text

from arch_chat.models.context import RunResult
from arch_chat.models.state import RoutingDecision
from arch_chat.registry import ARCH_REGISTRY


class StreamSink(Protocol):
    def start_turn(self, message: str) -> None: ...
    def phase(self, name: str, detail: str = "") -> None: ...
    def routing(self, decision: RoutingDecision, overridden: bool = False) -> None: ...
    def step_start(self, name: str, agent: str = "") -> None: ...
    def step_end(self, name: str, preview: str = "") -> None: ...
    def token(self, text: str) -> None: ...
    def reasoning(self, text: str) -> None: ...
    def tool_call(self, name: str, args: str = "") -> None: ...
    def spinner(self, message: str) -> None: ...
    def error(self, msg: str) -> None: ...
    def finalize(self, result: RunResult) -> None: ...


@dataclass
class NullStreamUI:
    """Records events for tests; no-op rendering."""

    events: list[tuple[str, Any]] = field(default_factory=list)
    trace_verbose: bool = True

    def start_turn(self, message: str) -> None:
        self.events.append(("start_turn", message))

    def phase(self, name: str, detail: str = "") -> None:
        self.events.append(("phase", name, detail))

    def routing(self, decision: RoutingDecision, overridden: bool = False) -> None:
        self.events.append(("routing", decision, overridden))

    def step_start(self, name: str, agent: str = "") -> None:
        self.events.append(("step_start", name, agent))

    def step_end(self, name: str, preview: str = "") -> None:
        self.events.append(("step_end", name, preview))

    def token(self, text: str) -> None:
        self.events.append(("token", text))

    def reasoning(self, text: str) -> None:
        self.events.append(("reasoning", text))

    def tool_call(self, name: str, args: str = "") -> None:
        self.events.append(("tool_call", name, args))

    def spinner(self, message: str) -> None:
        self.events.append(("spinner", message))

    def error(self, msg: str) -> None:
        self.events.append(("error", msg))

    def finalize(self, result: RunResult) -> None:
        self.events.append(("finalize", result))

    @contextmanager
    def live_context(self):
        yield self


@dataclass
class LiveStreamUI:
    """Rich Live dashboard for a single chat turn."""

    console: Console
    session_id: str = ""
    trace_verbose: bool = True
    _phase: str = "idle"
    _phase_detail: str = ""
    _architecture: str = ""
    _essence: str = ""
    _response_buffer: str = ""
    _reasoning_buffer: str = ""
    _activity: list[str] = field(default_factory=list)
    _current_step: str = ""
    _spinner_msg: str = ""
    _live: Live | None = field(default=None, repr=False)
    _streamed: bool = field(default=False, repr=False)

    def start_turn(self, message: str) -> None:
        self._phase = "starting"
        self._phase_detail = message[:80]
        self._response_buffer = ""
        self._reasoning_buffer = ""
        self._activity.clear()
        self._current_step = ""
        self._spinner_msg = ""
        self._streamed = True
        self._refresh()

    def phase(self, name: str, detail: str = "") -> None:
        self._phase = name
        self._phase_detail = detail
        self._spinner_msg = ""
        self._refresh()

    def routing(self, decision: RoutingDecision, overridden: bool = False) -> None:
        self._architecture = decision.architecture
        entry = ARCH_REGISTRY.get(decision.architecture)
        self._essence = entry.essence if entry else ""
        label = "Override" if overridden else "Routed"
        self._activity.append(
            f"[blue]{label}[/blue] [cyan]{decision.architecture}[/cyan] "
            f"({decision.confidence:.0%}) — {decision.reasoning[:120]}"
        )
        if len(self._activity) > 8:
            self._activity.pop(0)
        self._refresh()

    def step_start(self, name: str, agent: str = "") -> None:
        self._current_step = name
        label = f"{name} ({agent})" if agent else name
        self._activity.append(f"[cyan]▶[/cyan] [bold]{label}[/bold]")
        if len(self._activity) > 8:
            self._activity.pop(0)
        self._refresh()

    def step_end(self, name: str, preview: str = "") -> None:
        self._current_step = ""
        if self.trace_verbose and preview:
            short = preview[:120] + ("..." if len(preview) > 120 else "")
            self._activity.append(f"[green]✓[/green] {name}: {short}")
            if len(self._activity) > 8:
                self._activity.pop(0)
        self._refresh()

    def token(self, text: str) -> None:
        if text:
            self._response_buffer += text
            self._refresh()

    def reasoning(self, text: str) -> None:
        if text:
            self._reasoning_buffer += text
            self._refresh()

    def tool_call(self, name: str, args: str = "") -> None:
        line = f"[yellow]⚙ {name}[/yellow]"
        if args:
            line += f" {args[:80]}"
        self._activity.append(line)
        if len(self._activity) > 8:
            self._activity.pop(0)
        self._refresh()

    def spinner(self, message: str) -> None:
        self._spinner_msg = message
        self._refresh()

    def error(self, msg: str) -> None:
        self._stop_live()
        self.console.print(Panel(Text(msg, style="bold red"), title="Error", border_style="red"))

    def finalize(self, result: RunResult) -> None:
        self._stop_live()
        if not self._streamed:
            return
        content = self._response_buffer or result.content
        style = "red" if result.refused else "green"
        title = "Refused" if result.refused else "Assistant"
        self.console.print(Panel(content, title=title, border_style=style))

    @property
    def streamed(self) -> bool:
        return self._streamed

    def _make_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body", ratio=2),
            Layout(name="footer", size=8 if self.trace_verbose else 3),
        )

        phase_label = self._phase.replace("_", " ").title()
        header_parts = [f"[bold cyan]arch_chat[/bold cyan] [dim]| session {self.session_id}[/dim]"]
        header_parts.append(f"[magenta]{phase_label}[/magenta]")
        if self._architecture:
            header_parts.append(f"[cyan]{self._architecture}[/cyan]")
        if self._essence:
            header_parts.append(f"[dim]{self._essence}[/dim]")
        layout["header"].update(Panel(Text(" ".join(header_parts)), border_style="blue"))

        body_parts: list[Any] = []
        if self._reasoning_buffer and self.trace_verbose:
            body_parts.append(Text(self._reasoning_buffer[-600:], style="dim italic"))
        if self._response_buffer:
            body_parts.append(Text(self._response_buffer))
        elif self._spinner_msg:
            body_parts.append(Text(f"⏳ {self._spinner_msg}", style="yellow"))
        elif self._current_step:
            body_parts.append(Text(f"⏳ Running {self._current_step}...", style="yellow"))
        else:
            body_parts.append(Text("Waiting...", style="dim"))
        layout["body"].update(Panel(Group(*body_parts), title="Response", border_style="green"))

        if self.trace_verbose and self._activity:
            footer_lines = [Text.from_markup(line) for line in self._activity]
            layout["footer"].update(Panel(Group(*footer_lines), title="Activity", border_style="yellow"))
        elif self._spinner_msg:
            layout["footer"].update(Panel(Text(self._spinner_msg, style="dim"), border_style="dim"))
        else:
            layout["footer"].update(Panel(Text("", style="dim"), border_style="dim"))

        return layout

    def _refresh(self) -> None:
        if self._live:
            self._live.update(self._make_layout())

    def _stop_live(self) -> None:
        if self._live:
            self._live.stop()
            self._live = None

    @contextmanager
    def live_context(self):
        self._live = Live(
            self._make_layout(),
            console=self.console,
            refresh_per_second=10,
            transient=True,
        )
        self._live.start()
        try:
            yield self
        finally:
            self._stop_live()
