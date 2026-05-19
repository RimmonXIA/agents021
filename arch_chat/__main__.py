"""Module entry: python -m arch_chat"""

from __future__ import annotations

import argparse
import asyncio
import sys


def main() -> None:
    parser = argparse.ArgumentParser(description="Architecture Chatbot TUI")
    parser.add_argument("--session-id", help="Session identifier")
    parser.add_argument("--arch", help="Force architecture for first message")
    parser.add_argument("--trace", action="store_true", help="Enable verbose trace")
    parser.add_argument("--no-trace", action="store_true", help="Disable verbose trace")
    parser.add_argument("--no-stream", action="store_true", help="Disable live streaming UI")
    args = parser.parse_args()

    try:
        from arch_chat.config import get_settings
        from arch_chat.engine import ChatEngine
        from arch_chat.tui.repl import run_repl
    except Exception as e:
        print(f"Failed to load arch_chat: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        settings = get_settings()
    except Exception as e:
        print(
            f"Configuration error (need DEEPSEEK_API_KEY in .env): {e}",
            file=sys.stderr,
        )
        sys.exit(1)

    engine = ChatEngine(settings, session_id=args.session_id)
    if args.arch:
        engine.arch_override = args.arch.lower().replace("-", "_")
        engine.auto_route = False
    if args.trace:
        engine.trace_verbose = True
    if args.no_trace:
        engine.trace_verbose = False
    if args.no_stream:
        engine.stream_enabled = False

    asyncio.run(run_repl(engine))


if __name__ == "__main__":
    main()
