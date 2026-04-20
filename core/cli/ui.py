import sys
from typing import Any, List, Dict
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

    def update_tasks(self, todo: List[AtomicTask], completed: List[AtomicTask], running_ids: List[str]):
        table = Table(title="Execution Queue", expand=True)
        table.add_column("ID", justify="right", style="dim", width=4)
        table.add_column("Status", width=10)
        table.add_column("Task", style="white")

        # Completed
        for t in completed:
            table.add_row(t.id, "[t.success]DONE[/t.success]", t.description)
        
        # Running
        for t_id in running_ids:
            # We need to find the task in todo
            task = next((t for t in todo if t.id == t_id), None)
            if task:
                table.add_row(task.id, "[t.info]RUNNING[/t.info]", task.description)
        
        # Pending
        for t in todo:
            if t.id not in running_ids:
                table.add_row(t.id, "[dim]PENDING[/dim]", t.description)

        self.layout["tasks"].update(Panel(table, border_style="blue"))

    def update_memory(self, memory: Dict[str, Any]):
        table = Table(title="Blackboard State", expand=True)
        table.add_column("Key", style="t.warning")
        table.add_column("Value", style="white", overflow="ellipsis")

        for k, v in memory.items():
            val_str = str(v)[:50] + "..." if len(str(v)) > 50 else str(v)
            table.add_row(k, val_str)

        self.layout["memory"].update(Panel(table, border_style="yellow"))

    def add_thinking(self, agent_name: str, thought: str):
        self.thinking_log.append(f"[agent]{agent_name}[/agent]: {thought}")
        if len(self.thinking_log) > 20:
            self.thinking_log.pop(0)
        
        log_text = Text.from_markup("\n".join(self.thinking_log))
        self.layout["footer"].update(Panel(log_text, title="Thinking Stream", border_style="magenta"))

    def refresh_header(self):
        header_text = Text.assemble(
            ("TRINITY ", "bold cyan"),
            ("| ", "dim"),
            (f"Session: {self.session_id} ", "info"),
            ("| ", "dim"),
            (f"Intent: {self.intent[:50]}...", "intent")
        )
        self.layout["header"].update(Panel(header_text, border_style="cyan"))

    def start(self):
        self.refresh_header()
        self.live.start()

    def stop(self):
        self.live.stop()

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
