import logging

from rich.console import Console
from rich.logging import RichHandler
from core.cli.ui import TRINITY_THEME


def setup_logging(level: str = "INFO") -> None:
    """
    Sets up a standardized logging configuration using Rich for beautiful console output.
    """
    console = Console(theme=TRINITY_THEME)
    
    logging.basicConfig(
        level=level,
        format="%(message)s",
        datefmt="[%X]",
        handlers=[RichHandler(rich_tracebacks=True, console=console, show_path=False)]
    )

def get_logger(name: str) -> logging.Logger:
    """
    Returns a logger instance with the specified name.
    """
    return logging.getLogger(name)
