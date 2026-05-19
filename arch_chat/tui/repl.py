"""Interactive REPL for architecture chatbot."""

from __future__ import annotations

from prompt_toolkit import PromptSession
from prompt_toolkit.auto_suggest import AutoSuggestFromHistory
from prompt_toolkit.history import FileHistory
from rich.console import Console

from arch_chat.engine import ChatEngine
from arch_chat.registry import ARCH_REGISTRY, list_architectures
from arch_chat.tui import display
from arch_chat.tui.approval import request_approval
from arch_chat.tui.stream_ui import LiveStreamUI

console = Console()


class CommandProcessor:
    def __init__(self, engine: ChatEngine, session: PromptSession):
        self.engine = engine
        self.session = session
        self.commands = {
            "/help": self._cmd_help,
            "/exit": self._cmd_exit,
            "/arch": self._cmd_arch,
            "/auto": self._cmd_auto,
            "/trace": self._cmd_trace,
            "/stream": self._cmd_stream,
            "/memory": self._cmd_memory,
            "/graph": self._cmd_graph,
            "/reset": self._cmd_reset,
            "/clear": self._cmd_clear,
        }

    async def process(self, text: str) -> bool:
        text = text.strip()
        if not text:
            return True
        if text.startswith("/"):
            parts = text.split()
            cmd = parts[0].lower()
            args = parts[1:]
            handler = self.commands.get(cmd)
            if handler:
                return await handler(args)
            display.print_error(f"Unknown command: {cmd}")
            return True
        await self._run_chat(text)
        return True

    async def _run_chat(self, message: str) -> None:
        console.print(f"\n[bold cyan]>[/bold cyan] [italic]{message}[/italic]")
        stream_ui = None
        try:
            if self.engine.stream_enabled:
                stream_ui = LiveStreamUI(
                    console=console,
                    session_id=self.engine.session_id,
                    trace_verbose=self.engine.trace_verbose,
                )
                self.engine.stream_sink = stream_ui
                with stream_ui.live_context():
                    result = await self.engine.handle_message(message)
                    stream_ui.finalize(result)
            else:
                result = await self.engine.handle_message(message)
                if self.engine.last_routing:
                    overridden = bool(self.engine.arch_override)
                    display.print_routing(self.engine.last_routing, overridden=overridden)
                entry = ARCH_REGISTRY.get(
                    self.engine.last_routing.architecture if self.engine.last_routing else ""
                )
                if entry:
                    display.print_arch_header(entry.name, entry.essence)
                display.print_trace(result.trace, verbose=self.engine.trace_verbose)
                display.print_response(result)

            self.engine.session_state.update(result.session_state)
        except Exception as e:
            if stream_ui:
                stream_ui.error(str(e))
            else:
                display.print_error(str(e))
        finally:
            self.engine.stream_sink = None

    async def _cmd_help(self, args: list[str]) -> bool:
        console.print(
            "[bold cyan]Commands:[/bold cyan]\n"
            "  /help           — this message\n"
            "  /exit           — quit\n"
            "  /arch [name]    — show/set architecture override\n"
            "  /auto           — re-enable auto routing\n"
            "  /trace on|off   — toggle control-flow trace\n"
            "  /stream on|off  — toggle live streaming UI\n"
            "  /memory [clear] — show or clear session memory\n"
            "  /graph          — dump knowledge graph\n"
            "  /reset          — reset workflow session state\n"
            "  /clear          — clear screen\n"
        )
        console.print("[bold]Architectures:[/bold]")
        for entry in list_architectures():
            console.print(f"  [green]{entry.name}[/green] — {entry.essence}")
        return True

    async def _cmd_exit(self, _args: list[str]) -> bool:
        console.print("[yellow]Goodbye![/yellow]")
        return False

    async def _cmd_arch(self, args: list[str]) -> bool:
        if not args:
            current = self.engine.arch_override or ("auto" if self.engine.auto_route else "none")
            console.print(f"Current: [cyan]{current}[/cyan]")
            return True
        name = args[0].lower().replace("-", "_")
        if name == "auto":
            return await self._cmd_auto([])
        if name not in ARCH_REGISTRY:
            display.print_error(f"Unknown architecture: {name}")
            return True
        self.engine.arch_override = name
        self.engine.auto_route = False
        console.print(f"[green]Override set to {name} for next messages.[/green]")
        return True

    async def _cmd_auto(self, _args: list[str]) -> bool:
        self.engine.arch_override = None
        self.engine.auto_route = True
        console.print("[green]Auto-routing enabled.[/green]")
        return True

    async def _cmd_trace(self, args: list[str]) -> bool:
        if args and args[0].lower() == "off":
            self.engine.trace_verbose = False
        elif args and args[0].lower() == "on":
            self.engine.trace_verbose = True
        else:
            self.engine.trace_verbose = not self.engine.trace_verbose
        state = "on" if self.engine.trace_verbose else "off"
        console.print(f"Trace verbose: [cyan]{state}[/cyan]")
        return True

    async def _cmd_stream(self, args: list[str]) -> bool:
        if args and args[0].lower() == "off":
            self.engine.stream_enabled = False
        elif args and args[0].lower() == "on":
            self.engine.stream_enabled = True
        else:
            self.engine.stream_enabled = not self.engine.stream_enabled
        state = "on" if self.engine.stream_enabled else "off"
        console.print(f"Stream UI: [cyan]{state}[/cyan]")
        return True

    async def _cmd_memory(self, args: list[str]) -> bool:
        if args and args[0].lower() == "clear":
            self.engine.clear_memory()
            console.print("[green]Memory cleared.[/green]")
            return True
        console.print(self.engine.memory.format_display())
        return True

    async def _cmd_graph(self, _args: list[str]) -> bool:
        console.print(self.engine.graph.format_dump())
        return True

    async def _cmd_reset(self, _args: list[str]) -> bool:
        self.engine.reset_session()
        console.print(f"[green]Session {self.engine.session_id} workflow state reset.[/green]")
        return True

    async def _cmd_clear(self, _args: list[str]) -> bool:
        console.clear()
        display.print_banner()
        return True


async def run_repl(engine: ChatEngine) -> None:
    display.print_banner()
    console.print(f"[dim]Session: {engine.session_id}[/dim]")
    console.print(
        "[bold green]Architecture Chat — type /help for commands. "
        "Live streaming enabled.[/bold green]\n"
    )

    session = PromptSession(
        history=FileHistory(".arch_chat_history"),
        auto_suggest=AutoSuggestFromHistory(),
    )
    engine.approval_callback = lambda preview: request_approval(preview, session)
    processor = CommandProcessor(engine, session)

    while True:
        try:
            user_input = await session.prompt_async("arch> ")
            should_continue = await processor.process(user_input)
            if not should_continue:
                break
        except KeyboardInterrupt:
            continue
        except EOFError:
            break
        except Exception as e:
            display.print_error(str(e))

    console.print("[yellow]Exiting Architecture Chat...[/yellow]")
