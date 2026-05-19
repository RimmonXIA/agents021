"""In-memory market simulator for Mental Loop architecture."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field


@dataclass
class Portfolio:
    cash: float = 10000.0
    shares: float = 0.0

    def value(self, price: float) -> float:
        return self.cash + self.shares * price


@dataclass
class Market:
    price: float = 100.0
    portfolio: Portfolio = field(default_factory=Portfolio)

    def step(self, action: str, amount: float = 0.0) -> None:
        action = action.lower().strip()
        if action == "buy" and amount > 0:
            cost = amount * self.price
            if self.portfolio.cash >= cost:
                self.portfolio.cash -= cost
                self.portfolio.shares += amount
        elif action == "sell" and amount > 0:
            sell = min(amount, self.portfolio.shares)
            self.portfolio.shares -= sell
            self.portfolio.cash += sell * self.price
        elif action == "hold":
            self.price *= 1.01 if self.portfolio.shares > 0 else 0.99


REAL = Market()


def simulate_action(action: str, amount: float, horizon: int = 5) -> str:
    """Roll out an action on a forked copy of the market."""
    sim = copy.deepcopy(REAL)
    sim.step(action, amount)
    for _ in range(horizon - 1):
        sim.step("hold")
    val = sim.portfolio.value(sim.price)
    return f"Simulated value after {horizon} days: ${val:.2f} (price=${sim.price:.2f})"


def execute_action(action: str, amount: float) -> str:
    """Commit the action to the REAL market."""
    REAL.step(action, amount)
    val = REAL.portfolio.value(REAL.price)
    return f"Executed: {action} {amount}. Portfolio now ${val:.2f} (price=${REAL.price:.2f})."


def reset_market() -> None:
    global REAL
    REAL = Market()
