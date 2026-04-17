# tools/web_search.py
import json

def duckduckgo_search(query: str) -> str:
    """Mock search capability."""
    return json.dumps([
        {"url": "https://example.com/mock1", "snippet": f"Mock result for {query} 1"},
        {"url": "https://example.com/mock2", "snippet": f"Mock result for {query} 2"},
    ])
