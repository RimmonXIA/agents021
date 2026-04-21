import asyncio
import json
import uuid
import logging
from pathlib import Path
from typing import Optional
import typer
from rich.logging import RichHandler
from rich.table import Table

from core.config import settings
from core.agents.synthesizer import AgentSynthesizer
from core.engine.orchestrator import IntentOrchestrator
from core.memory.blackboard import Blackboard
from core.utils.logging import setup_logging, get_logger
from core.cli import ui
from core.cli.repl import run_repl

app = typer.Typer(help="Trinity: Modern Multi-Agent Orchestration CLI")
logger = get_logger("trinity")

class CLIThinkingHandler(logging.Handler):
    def __init__(self, live_ui: ui.LiveOrchestrator):
        super().__init__()
        self.live_ui = live_ui

    def emit(self, record):
        msg = self.format(record)
        if "Thinking Process" in msg:
            # Extract agent name and thought
            try:
                # Format: [AgentName Thinking Process]\nThought
                header, thought = msg.split("]\n", 1)
                agent_name = header.strip("[").replace(" Thinking Process", "")
                self.live_ui.add_thinking(agent_name, thought)
            except Exception:
                pass

@app.command()
def run(
    intent: str = typer.Argument(..., help="The user's original intent or task."),
    review: bool = typer.Option(True, help="Review the execution plan before starting."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging.")
):
    """Run the Trinity Agentic Loop for a given intent."""
    asyncio.run(_run_async(intent, review, verbose))

async def _run_async(intent: str, review: bool, verbose: bool):
    ui.print_banner()
    
    setup_logging(level="INFO" if verbose else "WARNING")
    session_id = str(uuid.uuid4())
    
    # 1. Initialize Blackboard
    bb = Blackboard(session_id=session_id, original_intent=intent)
    orchestrator = IntentOrchestrator(blackboard=bb)

    # 2. HITL Review
    if review:
        with ui.console.status("[bold cyan]Decomposing intent...[/bold cyan]"):
            await orchestrator.decompose_intent()
        
        plan = await bb.get_full_plan()
        if not ui.confirm_plan(plan):
            ui.console.print("[yellow]Execution aborted by user.[/yellow]")
            return

    # 4. Live Orchestration
    live_ui = ui.LiveOrchestrator(session_id, intent)
    orchestrator.ui_callback = live_ui.handle_event

    # Add CLI Thinking Handler
    thinking_handler = CLIThinkingHandler(live_ui)
    logging.getLogger().addHandler(thinking_handler)

    try:
        live_ui.start()
        
        # We need a way to pulse the UI. 
        # For now, we'll just run the loop and hope the updates happen via hooks.
        # Actually, we can wrap the orchestrator's loop steps.
        
        # Note: orchestrator.run_loop() is a monolithic call.
        # To make it truly live, we might need to modify run_loop or use polling.
        # Let's add a polling task.
        
        async def ui_poller():
            while bb.state.status == "running":
                live_ui.update_tasks(bb.state.todo_list, bb.state.completed_tasks, list(orchestrator.running_tasks.values()))
                live_ui.update_memory(bb.state.shared_memory)
                await asyncio.sleep(0.5)

        poller = asyncio.create_task(ui_poller())
        await orchestrator.run_loop()
        await poller
        
    finally:
        orchestrator.ui_callback = None
        live_ui.stop()
        logging.getLogger().removeHandler(thinking_handler)

    # 5. Assistant output (after Live stops — Rich Live hides prints during run)
    ui.print_task_results_from_shared_memory(bb.state.shared_memory)

    # 6. Summary & Feedback
    ui.print_summary(session_id, bb.state.status, list(bb.state.shared_memory.keys()))
    
    if bb.state.status == "completed":
        rating = ui.get_feedback()
        # Save rating to session metadata if needed
        logger.info(f"User Rating: {rating}/10")

@app.command()
def doctor():
    """Check environment health and configuration."""
    ui.console.print("[bold cyan]Trinity Doctor Diagnostic[/bold cyan]\n")

    checks = {
        "DEEPSEEK_API_KEY": settings.deepseek_api_key if hasattr(settings, "deepseek_api_key") else None,
        "SQLITE_DB_PATH": settings.sqlite_db_path,
        "LANCE_DB_DIR": settings.lancedb_dir,
    }

    table = Table(box=None)
    table.add_column("Check", style="white")
    table.add_column("Status", justify="center")
    table.add_column("Details", style="dim")

    for name, val in checks.items():
        status = "[t.success]PASS[/t.success]" if val else "[t.error]FAIL[/t.error]"
        detail = str(val) if val else "Not configured"
        table.add_row(name, status, detail)

    ui.console.print(table)

    # Capability probe
    ui.console.print("\n[bold cyan]Capability Probe[/bold cyan]")
    synth = AgentSynthesizer()
    cap_table = Table(box=None)
    cap_table.add_column("Capability", style="white")
    cap_table.add_column("Status", justify="center")
    cap_table.add_column("Details", style="dim")

    for cap in sorted(synth.available_capabilities):
        cap_table.add_row(cap, "[t.success]OK[/t.success]", "Template loaded successfully")
    for cap, reason in sorted(synth.broken_capabilities.items()):
        cap_table.add_row(cap, "[t.error]BROKEN[/t.error]", reason)

    ui.console.print(cap_table)


@app.command()
def rollout_status(
    report_path: str = typer.Option("evals/kpi_report.json", help="Path to generated KPI report."),
):
    """Validate observe/soft/hard rollout gates from KPI report."""
    path = Path(report_path)
    if not path.exists():
        ui.console.print(f"[t.error]KPI report not found: {path}[/t.error]")
        raise typer.Exit(code=1)

    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        ui.console.print(f"[t.error]Failed to parse KPI report: {exc}[/t.error]")
        raise typer.Exit(code=1)

    stage_gates = report.get("stage_gates", {})
    rollback_triggers = report.get("rollback_triggers", {})
    if not isinstance(stage_gates, dict) or not isinstance(rollback_triggers, dict):
        ui.console.print("[t.error]KPI report missing stage_gates or rollback_triggers.[/t.error]")
        raise typer.Exit(code=1)

    table = Table(box=None)
    table.add_column("Stage", style="white")
    table.add_column("Pass", justify="center")
    table.add_column("Checks", style="dim")
    for stage in ("observe", "soft", "hard"):
        gate = stage_gates.get(stage, {})
        checks = gate.get("checks", {}) if isinstance(gate, dict) else {}
        check_summary = ", ".join(f"{k}={bool(v)}" for k, v in checks.items()) if checks else "n/a"
        stage_pass = bool(gate.get("pass", False)) if isinstance(gate, dict) else False
        stage_status = "[t.success]PASS[/t.success]" if stage_pass else "[t.error]FAIL[/t.error]"
        table.add_row(stage, stage_status, check_summary)
    ui.console.print(table)

    active_triggers = [name for name, active in rollback_triggers.items() if bool(active)]
    if active_triggers:
        ui.console.print(f"[t.error]Rollback required:[/t.error] {', '.join(active_triggers)}")
        raise typer.Exit(code=1)
    ui.console.print("[t.success]No rollback triggers active. Promotion allowed.[/t.success]")

@app.command()
def list_sessions(limit: int = typer.Option(20, min=1, max=500, help="Maximum sessions to list.")):
    """List previous Trinity sessions."""
    bb = Blackboard(session_id="session-index")
    sessions = bb.list_sessions(limit=limit)
    if not sessions:
        ui.console.print("[t.warning]No persisted sessions found.[/t.warning]")
        return
    table = Table(box=None)
    table.add_column("Session ID", style="white")
    for sid in sessions:
        table.add_row(sid)
    ui.console.print(table)

@app.command()
def chat(
    session_id: Optional[str] = typer.Option(None, help="Continue an existing session ID."),
    verbose: bool = typer.Option(False, "--verbose", "-v", help="Enable verbose logging.")
):
    """Start an interactive Claude-like chat session."""
    setup_logging(level="DEBUG" if verbose else "INFO")
    
    sid = session_id or str(uuid.uuid4())
    bb = Blackboard(session_id=sid)
    orchestrator = IntentOrchestrator(blackboard=bb)
    
    asyncio.run(run_repl(orchestrator, bb))

if __name__ == "__main__":
    app()
