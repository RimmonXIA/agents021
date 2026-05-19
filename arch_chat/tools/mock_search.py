"""Flaky web search tool from PEV architecture doc."""


def flaky_web_search(query: str) -> str:
    """Search the web. Intentionally unreliable for PEV demos."""
    q = query.lower()
    if "employee count" in q or "unavailable" in q:
        return "Error: Could not retrieve data. The API endpoint is currently unavailable."
    if "apple" in q or "aapl" in q:
        return "Apple Inc. (AAPL) reported revenue of $94.9B in Q1 2024."
    if "microsoft" in q or "msft" in q:
        return "Microsoft (MSFT) reported revenue of $62.0B in Q2 FY2024."
    return f"Mock search result for: {query}"


def mock_stock_price(symbol: str) -> str:
    """Return a mock stock price."""
    prices = {"AAPL": 172.35, "MSFT": 415.20, "GOOG": 175.50}
    sym = symbol.upper().strip()
    price = prices.get(sym, 100.0)
    return f"The current price of {sym} is ${price:.2f}."
