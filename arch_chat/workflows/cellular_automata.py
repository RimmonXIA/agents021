"""Cellular Automata: decentralized pathfinding via local rules."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field

from arch_chat.models.context import RunContext, RunResult
from arch_chat.workflows.base import ArchitectureRunner

ESSENCE = "Local rules produce global pathfinding behavior without central planner"
NAME = "cellular_automata"


@dataclass
class CellAgent:
    type: str  # EMPTY | OBSTACLE | GOAL
    pathfinding_value: float = float("inf")

    def update(self, neighbors: list[CellAgent]) -> None:
        if self.type == "OBSTACLE":
            return
        if self.type == "GOAL":
            self.pathfinding_value = 0
            return
        m = min((n.pathfinding_value for n in neighbors), default=float("inf"))
        self.pathfinding_value = min(self.pathfinding_value, m + 1)


def neighbors_of(grid: list[list[CellAgent]], r: int, c: int) -> list[CellAgent]:
    rows, cols = len(grid), len(grid[0])
    result = []
    for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols:
            result.append(grid[nr][nc])
    return result


def run_ca(grid: list[list[CellAgent]], steps: int = 50) -> list[list[CellAgent]]:
    for _ in range(steps):
        snapshot = [[copy.deepcopy(cell) for cell in row] for row in grid]
        for r in range(len(grid)):
            for c in range(len(grid[0])):
                grid[r][c].update(neighbors_of(snapshot, r, c))
    return grid


def default_grid() -> list[list[CellAgent]]:
    layout = [
        "SEEEG",
        "EOEOE",
        "EEEEE",
    ]
    grid = []
    for row in layout:
        cells = []
        for ch in row:
            if ch == "S":
                cells.append(CellAgent(type="EMPTY"))
            elif ch == "G":
                cells.append(CellAgent(type="GOAL"))
            elif ch == "O":
                cells.append(CellAgent(type="OBSTACLE"))
            else:
                cells.append(CellAgent(type="EMPTY"))
        grid.append(cells)
    return grid


def grid_to_str(grid: list[list[CellAgent]]) -> str:
    lines = []
    for row in grid:
        parts = []
        for cell in row:
            if cell.type == "OBSTACLE":
                parts.append(" # ")
            elif cell.type == "GOAL":
                val = " G " if cell.pathfinding_value == float("inf") else f"{int(cell.pathfinding_value):2d}"
                parts.append(val)
            elif cell.pathfinding_value == float("inf"):
                parts.append(" . ")
            else:
                parts.append(f"{int(cell.pathfinding_value):2d}")
        lines.append("".join(parts))
    return "\n".join(lines)


class CellularAutomataRunner(ArchitectureRunner):
    name = NAME
    essence = ESSENCE
    number = 17

    async def run(self, ctx: RunContext) -> RunResult:
        ctx.add_trace("ca_init", "Building grid: S=start area, G=goal, #=obstacle")
        grid = default_grid()
        ctx.add_trace("ca_before", grid_to_str(grid))

        grid = run_ca(grid, steps=50)
        ctx.add_trace("ca_after", grid_to_str(grid), ctx.snapshot_state())

        goal_val = next(
            (c.pathfinding_value for row in grid for c in row if c.type == "GOAL"),
            float("inf"),
        )
        content = (
            f"Cellular automata pathfinding complete ({50} iterations).\n\n"
            f"Distance field at goal: {goal_val}\n\n"
            f"{grid_to_str(grid)}\n\n"
            f"Lower values = closer to goal. No central planner — only local neighbor rules."
        )
        if ctx.message.lower() not in ("", "run", "demo"):
            content = f"Query noted: {ctx.message}\n\n{content}"
        return RunResult(content=content, trace=ctx.trace, session_state=ctx.session_state)
