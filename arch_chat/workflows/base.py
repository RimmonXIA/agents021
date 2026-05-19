from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from arch_chat.models.context import RunContext, RunResult


class ArchitectureRunner(ABC):
    name: str
    essence: str
    number: int

    @abstractmethod
    async def run(self, ctx: RunContext) -> RunResult:
        ...
