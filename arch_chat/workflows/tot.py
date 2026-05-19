"""Tree-of-Thoughts: programmatic BFS search with LLM proposer."""

from __future__ import annotations

from dataclasses import dataclass, field

from agno.agent import Agent

from arch_chat.models.context import RunContext, RunResult
from arch_chat.models.state import PuzzleMoves
from arch_chat.runner import run_agent
from arch_chat.workflows.base import ArchitectureRunner

ESSENCE = "Search tree of candidate paths with programmatic pruning"
NAME = "tot"

DANGEROUS = [("wolf", "goat"), ("goat", "cabbage")]


@dataclass(frozen=True)
class PuzzleState:
    left_bank: frozenset[str] = field(default_factory=lambda: frozenset({"wolf", "goat", "cabbage", "farmer"}))
    right_bank: frozenset[str] = field(default_factory=frozenset)
    boat_location: str = "left"
    move_description: str = "Initial state."

    def is_valid(self) -> bool:
        unguarded = self.left_bank if self.boat_location == "right" else self.right_bank
        if "farmer" in unguarded:
            return True
        for a, b in DANGEROUS:
            if {a, b}.issubset(unguarded):
                return False
        return True

    def is_goal(self) -> bool:
        return self.right_bank == frozenset({"wolf", "goat", "cabbage", "farmer"})


def apply_move(state: PuzzleState, item: str) -> PuzzleState | None:
    if state.boat_location == "left":
        if item not in state.left_bank and item != "farmer":
            return None
        new_left = set(state.left_bank)
        new_right = set(state.right_bank)
        if item in new_left:
            new_left.remove(item)
            new_right.add(item)
        if "farmer" in new_left:
            new_left.remove("farmer")
            new_right.add("farmer")
        return PuzzleState(
            left_bank=frozenset(new_left),
            right_bank=frozenset(new_right),
            boat_location="right",
            move_description=f"Moved {item} and farmer to right",
        )
    if item not in state.right_bank and item != "farmer":
        return None
    new_left = set(state.left_bank)
    new_right = set(state.right_bank)
    if item in new_right:
        new_right.remove(item)
        new_left.add(item)
    if "farmer" in new_right:
        new_right.remove("farmer")
        new_left.add("farmer")
    return PuzzleState(
        left_bank=frozenset(new_left),
        right_bank=frozenset(new_right),
        boat_location="left",
        move_description=f"Moved {item} and farmer to left",
    )


def tot_solve_bfs(max_depth: int = 12) -> list[PuzzleState] | None:
    start = PuzzleState()
    queue: list[list[PuzzleState]] = [[start]]
    visited = {start.left_bank, start.right_bank, start.boat_location}

    while queue:
        path = queue.pop(0)
        current = path[-1]
        if current.is_goal():
            return path
        if len(path) >= max_depth:
            continue
        for item in ["wolf", "goat", "cabbage", "farmer"]:
            nxt = apply_move(current, item)
            if nxt and nxt.is_valid():
                key = (nxt.left_bank, nxt.right_bank, nxt.boat_location)
                if key not in visited:
                    visited.add(key)
                    queue.append(path + [nxt])
    return None


class ToTRunner(ArchitectureRunner):
    name = NAME
    essence = ESSENCE
    number = 10

    async def run(self, ctx: RunContext) -> RunResult:
        puzzle_keywords = ("wolf", "goat", "cabbage", "river", "cross")
        is_puzzle = any(k in ctx.message.lower() for k in puzzle_keywords)

        if is_puzzle:
            ctx.add_trace("tot_bfs", "Programmatic BFS on wolf-goat-cabbage state space")
            path = tot_solve_bfs()
            if path:
                steps = "\n".join(f"{i + 1}. {s.move_description}" for i, s in enumerate(path[1:], 1))
                content = f"Solved in {len(path) - 1} moves:\n{steps}"
                ctx.session_state["solution"] = steps
                ctx.add_trace("solution", steps[:400], ctx.snapshot_state())
                return RunResult(content=content, trace=ctx.trace, session_state=ctx.session_state)
            return RunResult(content="No solution found within search depth.", trace=ctx.trace)

        proposer = Agent(
            name="tot_proposer",
            model=ctx.model,
            output_schema=PuzzleMoves,
            instructions="Propose 3 distinct next-step approaches for the problem.",
        )
        result = await run_agent(proposer, ctx.message, response_model=PuzzleMoves)
        if result.success:
            moves: PuzzleMoves = result.parsed
            lines = "\n".join(f"- {m.move_description}" for m in moves.moves)
            ctx.add_trace("tot_propose", lines)
            content = f"Candidate paths to explore:\n{lines}\n\n(BFS search applies to puzzle domains.)"
        else:
            content = "ToT proposer failed. Try a puzzle-style question (wolf/goat/cabbage)."
        return RunResult(content=content, trace=ctx.trace, session_state=ctx.session_state)
