import sys
from typing import Any, Dict, List, Optional
from rich.console import Console, Group
from rich.panel import Panel
from rich.table import Table
from rich.live import Live
from rich.layout import Layout
from rich.status import Status
from rich.text import Text
from rich.theme import Theme
from rich.syntax import Syntax
from rich.progress import Progress, SpinnerColumn, TextColumn

from core.models import IOPlan, AtomicTask

# Custom Theme — prefixed with 't.' to avoid collisions with agno's internal Rich console
TRINITY_THEME = Theme({
    "t.info": "cyan",
    "t.warning": "yellow",
    "t.error": "red bold",
    "t.success": "green bold",
    "t.step": "blue bold",
    "t.agent": "magenta bold",
    "t.intent": "white bold italic",
})

console = Console(theme=TRINITY_THEME)

def print_banner():
    banner = """
    [bold cyan]████████╗██████╗ ██╗███╗   ██╗██╗████████╗██╗   ██╗[/bold cyan]
    [bold cyan]╚══██╔══╝██╔══██╗██║████╗  ██║██║╚══██╔══╝╚██╗ ██╔╝[/bold cyan]
    [bold cyan]   ██║   ██████╔╝██║██╔██╗ ██║██║   ██║    ╚████╔╝ [/bold cyan]
    [bold cyan]   ██║   ██╔══██╗██║██║╚██╗██║██║   ██║     ╚██╔╝  [/bold cyan]
    [bold cyan]   ██║   ██║  ██║██║██║ ╚████║██║   ██║      ██║   [/bold cyan]
    [bold cyan]   ╚═╝   ╚═╝  ╚═╝╚═╝╚═╝  ╚═══╝╚═╝   ╚═╝      ╚═╝   [/bold cyan]
    [dim cyan]         Next-Gen Multi-Agent Orchestration[/dim cyan]
    """
    console.print(banner)

def confirm_plan(plan: IOPlan) -> bool:
    console.print("\n[bold cyan]Proposed Execution Plan:[/bold cyan]")
    table = Table(box=None, expand=True)
    table.add_column("ID", style="dim", width=4)
    table.add_column("Task Description", style="white")
    table.add_column("Capabilities", style="t.info")
    table.add_column("Depends On", style="t.warning")

    for task in plan.tasks:
        table.add_row(
            task.id,
            task.description,
            ", ".join(task.required_capabilities),
            ", ".join(task.depends_on) if task.depends_on else "-"
        )
    
    console.print(Panel(table, border_style="cyan", title="IOPlan Decomposition"))
    
    from rich.prompt import Confirm
    return Confirm.ask("[bold yellow]Do you want to proceed with this plan?[/bold yellow]")

class LiveOrchestrator:
    def __init__(self, session_id: str, intent: str):
        self.session_id = session_id
        self.intent = intent
        self.tasks: List[Dict[str, Any]] = []
        self.shared_memory: Dict[str, Any] = {}
        self.thinking_log: List[str] = []
        self._stats: Dict[str, int] = {"running": 0, "pending": 0, "completed": 0}
        self._last_stats_key: tuple[int, int, int] | None = None
        self._task_activity: Dict[str, str] = {}
        self.layout = self._make_layout()
        self.live = Live(self.layout, console=console, refresh_per_second=4, screen=False)

    def _make_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="main"),
            Layout(name="footer", size=10)
        )
        layout["main"].split_row(
            Layout(name="tasks", ratio=2),
            Layout(name="memory", ratio=1)
        )
        return layout

    def update_tasks(self, todo: List[AtomicTask], completed: List[AtomicTask], running: List[tuple[AtomicTask, Any]]):
        table = Table(title="Execution Queue", expand=True)
        table.add_column("ID", justify="right", style="dim", width=4)
        table.add_column("Status", width=10)
        table.add_column("Task", style="white")

        # Completed
        for t in completed:
            table.add_row(t.id, "[t.success]DONE[/t.success]", t.description)
        
        # Running
        for task, _ in running:
            table.add_row(task.id, "[t.info]RUNNING[/t.info]", task.description)
        
        # Pending
        running_ids = {t.id for t, _ in running}
        for t in todo:
            if t.id not in running_ids:
                table.add_row(t.id, "[dim]PENDING[/dim]", t.description)

        self.layout["tasks"].update(Panel(table, border_style="blue"))

    def update_memory(self, memory: Dict[str, Any]):
        table = Table(title="Blackboard State", expand=True)
        table.add_column("Key", style="t.warning")
        table.add_column("Value", style="white", overflow="ellipsis")

        if not memory:
            table.add_row("[dim]—[/dim]", "[dim]No entries yet[/dim]")
        else:
            for k, v in memory.items():
                val_str = str(v)[:50] + "..." if len(str(v)) > 50 else str(v)
                table.add_row(k, val_str)

        self.layout["memory"].update(Panel(table, border_style="yellow"))

    def handle_event(self, event: str, data: Dict[str, Any]) -> None:
        """Orchestrator ui_callback (same events as REPL → ChatUI)."""
        if event == "stats":
            key = (int(data["running"]), int(data["pending"]), int(data["completed"]))
            if key == self._last_stats_key:
                return
            self._last_stats_key = key
            self._stats = {"running": key[0], "pending": key[1], "completed": key[2]}
            self._refresh_footer()
        elif event == "task_status":
            tid = str(data["task_id"])
            status = str(data["status"])
            if status == "finished":
                self._task_activity.pop(tid, None)
            else:
                self._task_activity[tid] = status
            self._refresh_footer()

    def _refresh_footer(self) -> None:
        s = self._stats
        summary = Text.assemble(
            ("RUNNING ", "bold"),
            (f"{s['running']}  ", "cyan"),
            ("PENDING ", "bold"),
            (f"{s['pending']}  ", "dim"),
            ("DONE ", "bold"),
            (f"{s['completed']}", "t.success"),
        )

        body_parts: List[Any] = [summary]
        for tid in sorted(self._task_activity.keys()):
            st = self._task_activity[tid]
            line = Text()
            line.append(f"{tid} ", style="cyan")
            line.append(st, overflow="ellipsis")
            body_parts.append(line)

        if self.thinking_log:
            body_parts.append(Text("— Thinking —", style="dim italic"))
            for entry in self.thinking_log[-6:]:
                body_parts.append(Text.from_markup(entry, overflow="ellipsis"))

        idle = (
            not self._task_activity
            and not self.thinking_log
            and s["running"] == 0
            and s["pending"] == 0
            and s["completed"] == 0
        )
        if idle:
            body_parts = [Text("Waiting for task activity…", style="dim")]

        self.layout["footer"].update(
            Panel(Group(*body_parts), title="Activity", border_style="magenta")
        )

    def add_thinking(self, agent_name: str, thought: str):
        self.thinking_log.append(f"[t.agent]{agent_name}[/t.agent]: {thought}")
        if len(self.thinking_log) > 20:
            self.thinking_log.pop(0)
        self._refresh_footer()

    def refresh_header(self):
        intent_display = self.intent if len(self.intent) <= 50 else self.intent[:50] + "..."
        header_text = Text.assemble(
            ("TRINITY ", "bold cyan"),
            ("| ", "dim"),
            (f"Session: {self.session_id} ", "t.info"),
            ("| ", "dim"),
            ("Intent: ", "dim"),
            (intent_display, "t.intent"),
        )
        self.layout["header"].update(Panel(header_text, border_style="cyan"))

    def start(self):
        self.refresh_header()
        self._refresh_footer()
        self.live.start()

    def stop(self):
        self.live.stop()

class ChatUI:
    """UI for the interactive REPL mode with a live Task HUD."""
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.console = console
        self._live: Optional[Live] = None
        self._status_data: Dict[str, str] = {} # task_id -> status_text
        self._stats = {"running": 0, "pending": 0, "completed": 0}
        self._last_stats: tuple[int, int, int] | None = None

    def _make_hud(self) -> Panel:
        """Creates a compact HUD panel for the live display."""
        table = Table.grid(expand=True)
        table.add_column(style="bold cyan", width=20)
        table.add_column(style="dim italic")

        # Summary line
        summary = Text.assemble(
            (" ◈ TRINITY SWARM ", "bold white on blue"),
            (f"  RUNNING: {self._stats['running']} ", "bold cyan"),
            (f"  PENDING: {self._stats['pending']} ", "dim"),
            (f"  DONE: {self._stats['completed']} ", "bold green"),
        )
        
        # Detail lines for running tasks
        details = Group(*[
            Text.from_markup(f"  [cyan]•[/cyan] [bold]{tid}[/bold]: {status}") 
            for tid, status in self._status_data.items()
        ])

        return Panel(
            Group(summary, details),
            border_style="blue",
            padding=(0, 1)
        )

    def _hud_is_meaningful(self) -> bool:
        s = self._stats
        return bool(s["running"] or s["pending"] or s["completed"] or self._status_data)

    def _ensure_live(self) -> None:
        """Start Rich Live only when there is something to show (avoids fighting streamed logs)."""
        if self._live or not self._hud_is_meaningful():
            return
        self._live = Live(self._make_hud(), console=self.console, refresh_per_second=4, transient=True)
        self._live.start()

    def start_live(self):
        """Reserved for compatibility; HUD starts lazily via _ensure_live."""
        self._ensure_live()

    def stop_live(self):
        """Stops the live HUD."""
        if self._live:
            self._live.stop()
            self._live = None
        self._last_stats = None

    def update_stats(self, running: int, pending: int, completed: int):
        self._stats = {"running": running, "pending": pending, "completed": completed}
        key = (running, pending, completed)
        if key == self._last_stats:
            return
        self._last_stats = key
        self._ensure_live()
        if self._live:
            self._live.update(self._make_hud())

    def update_task_status(self, task_id: str, status: str):
        if status == "finished":
            if task_id in self._status_data:
                del self._status_data[task_id]
        else:
            self._status_data[task_id] = status

        self._ensure_live()
        if self._live:
            self._live.update(self._make_hud())

    def print_task_start(self, task_id: str, description: str):
        self.console.print(f"[t.step]▶ Task {task_id}:[/t.step] {description}")

    def print_task_success(self, task_id: str, output: str):
        body = output if len(output) <= 16_000 else output[:16_000] + "\n\n… [truncated]"
        self.console.print(Panel(body, title=f"✓ Task {task_id}", border_style="green"))

    def print_error(self, message: str):
        self.console.print(f"[t.error]✘ ERROR:[/t.error] {message}")


def print_task_results_from_shared_memory(shared_memory: Dict[str, Any]) -> None:
    """
    Print each `*_result` value as a panel. Call this after Rich Live is stopped;
    printing during Live often does not show up (output is overwritten).
    """
    keys = sorted(k for k in shared_memory if k.endswith("_result"))
    if not keys:
        return
    console.print()
    for key in keys:
        val = shared_memory[key]
        text = val if isinstance(val, str) else str(val)
        tid = key.removesuffix("_result")
        body = text if len(text) <= 16_000 else text[:16_000] + "\n\n… [truncated]"
        console.print(Panel(body, title=f"✓ Task {tid}", border_style="green"))


def print_summary(session_id: str, status: str, result_keys: List[str]):
    console.print("\n" + "=" * 60)
    if status == "completed":
        console.print(f"[t.success]TRINITY LOOP FINISHED SUCCESSFULLY[/t.success]")
    else:
        console.print(f"[t.error]TRINITY LOOP FAILED[/t.error]")
    
    console.print(f"Session ID: [dim]{session_id}[/dim]")
    console.print(f"Produced Keys: [t.info]{', '.join(result_keys)}[/t.info]")
    console.print("=" * 60 + "\n")

def get_feedback() -> int:
    from rich.prompt import IntPrompt
    return IntPrompt.ask(
        "[bold yellow]Rate the result (1-10)[/bold yellow]",
        choices=[str(i) for i in range(1, 11)],
        default=10
    )
