import sys
import asyncio
from typing import Optional
from prompt_toolkit import PromptSession
from prompt_toolkit.history import FileHistory
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from core.utils.logging import get_logger
from core.cli import ui

logger = get_logger("trinity.repl")
console = Console()

class CommandProcessor:
    def __init__(self, orchestrator, blackboard):
        self.orchestrator = orchestrator
        self.bb = blackboard
        self.ui = ui.ChatUI(blackboard.state.session_id)
        self.commands = {
            "/help": self._cmd_help,
            "/exit": self._cmd_exit,
            "/clear": self._cmd_clear,
            "/reset": self._cmd_reset,
            "/status": self._cmd_status,
            "/plan": self._cmd_plan,
        }

    async def process(self, text: str) -> bool:
        """Process a line of input. Returns True to continue, False to exit."""
        text = text.strip()
        if not text:
            return True

        if text.startswith("/"):
            parts = text.split()
            cmd = parts[0].lower()
            args = parts[1:]
            
            if cmd in self.commands:
                return await self.commands[cmd](args)
            else:
                self.ui.print_error(f"Unknown command: {cmd}")
                return True
        
        # If not a command, it's an intent for the orchestrator
        await self._run_intent(text)
        return True

    async def _run_intent(self, intent: str):
        console.print(f"\n[bold cyan]>[/bold cyan] [italic]{intent}[/italic]\n")
        
        self.bb.state.original_intent = intent
        
        async with self.bb.lock:
            self.bb.state.todo_list = []
            self.bb.state.status = "running"

        # Define UI Callback for the Orchestrator
        def ui_callback(event: str, data: dict):
            if event == "stats":
                self.ui.update_stats(data["running"], data["pending"], data["completed"])
            elif event == "task_status":
                self.ui.update_task_status(data["task_id"], data["status"])

        # Inject callback into orchestrator
        self.orchestrator.ui_callback = ui_callback

        # Thinking is logged once via Rich (DEBUG); no duplicate ChatThinkingHandler print.
        try:
            await self.orchestrator.run_loop()
            self.ui.stop_live()
            ui.print_task_results_from_shared_memory(self.bb.state.shared_memory)
            ui.print_summary(self.bb.state.session_id, self.bb.state.status, list(self.bb.state.shared_memory.keys()))
        except Exception as e:
            self.ui.stop_live()
            self.ui.print_error(str(e))
        finally:
            self.orchestrator.ui_callback = None

    async def _cmd_help(self, args) -> bool:
        help_text = """
[bold cyan]Available Commands:[/bold cyan]
  [bold]/help[/bold]      - Show this help message
  [bold]/exit[/bold]      - Exit the session
  [bold]/clear[/bold]     - Clear the terminal screen
  [bold]/reset[/bold]     - Reset the current blackboard state
  [bold]/status[/bold]    - Show the current session status and memory
  [bold]/plan[/bold]      - Show the last execution plan
"""
        console.print(help_text)
        return True

    async def _cmd_exit(self, args) -> bool:
        console.print("[yellow]Goodbye![/yellow]")
        return False

    async def _cmd_clear(self, args) -> bool:
        console.clear()
        ui.print_banner()
        return True

    async def _cmd_reset(self, args) -> bool:
        # Re-initialize blackboard with the same session ID but empty state
        sid = self.bb.state.session_id
        async with self.bb.lock:
            self.bb.state.shared_memory = {}
            self.bb.state.todo_list = []
            self.bb.state.completed_tasks = []
            self.bb.state.status = "idle"
        console.print(f"[green]Session {sid} reset.[/green]")
        return True

    async def _cmd_status(self, args) -> bool:
        ui.print_summary(self.bb.state.session_id, self.bb.state.status, list(self.bb.state.shared_memory.keys()))
        return True

    async def _cmd_plan(self, args) -> bool:
        plan = await self.bb.get_full_plan()
        if plan and plan.tasks:
            ui.confirm_plan(plan) # Just use the same print logic
        else:
            console.print("[yellow]No plan available for this session.[/yellow]")
        return True

async def run_repl(orchestrator, blackboard):
    """The main REPL loop."""
    ui.print_banner()
    console.print(f"[dim]Session ID: {blackboard.state.session_id}[/dim]")
    console.print("[bold green]Welcome to Trinity Interactive Shell. Type /help for commands.[/bold green]\n")

    session = PromptSession(
        history=FileHistory(".trinity_history"),
        auto_suggest=AutoSuggestFromHistory(),
    )
    
    processor = CommandProcessor(orchestrator, blackboard)

    while True:
        try:
            user_input = await session.prompt_async("trinity> ")
            should_continue = await processor.process(user_input)
            if not should_continue:
                break
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        except Exception as e:
            console.print(f"[bold red]Error:[/bold red] {e}")
            logger.exception("REPL Error")

    console.print("[yellow]Exiting Trinity...[/yellow]")
