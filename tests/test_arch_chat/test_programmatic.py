from arch_chat.tools.in_memory_graph import InMemoryGraph
from arch_chat.workflows.cellular_automata import (
    CellAgent,
    default_grid,
    grid_to_str,
    run_ca,
)
from arch_chat.workflows.tot import PuzzleState, apply_move, tot_solve_bfs


def test_puzzle_state_valid_initial():
    s = PuzzleState()
    assert s.is_valid()
    assert not s.is_goal()


def test_puzzle_state_goal():
    s = PuzzleState(
        left_bank=frozenset(),
        right_bank=frozenset({"wolf", "goat", "cabbage", "farmer"}),
        boat_location="right",
    )
    assert s.is_goal()


def test_tot_bfs_finds_solution():
    path = tot_solve_bfs(max_depth=15)
    assert path is not None
    assert path[-1].is_goal()


def test_apply_move_invalid_item():
    s = PuzzleState()
    result = apply_move(s, "dragon")
    assert result is None


def test_cellular_automata_run():
    grid = default_grid()
    run_ca(grid, steps=10)
    goal_val = next(c.pathfinding_value for row in grid for c in row if c.type == "GOAL")
    assert goal_val == 0


def test_in_memory_graph_query():
    g = InMemoryGraph()
    g.add_text_facts("Alice works for Acme")
    answer = g.answer_query("who works for")
    assert "Alice" in answer
    assert "Acme" in answer
