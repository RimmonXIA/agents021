# core/micro_plane/capability_registry.py

from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from tools.web_search import duckduckgo_search
from tools.data_processor import extract_json_entities
from tools.code_analyzer import ast_parse_complexity

def get_search_capability() -> Agent:
    return Agent(
        model=DeepSeek(),
        name="SearchCapability",
        description="Executes web searches and returns structured results.",
        instructions=[
            "Accept search query string as input.",
            "Execute web search using the provided tool.",
            "Format output as a JSON array of search result objects containing 'url' and 'snippet'.",
            "Do not output conversational text."
        ],
        tools=[duckduckgo_search],
    )

def get_data_extraction_capability() -> Agent:
    return Agent(
        model=DeepSeek(),
        name="DataExtractionCapability",
        description="Extracts structured entities from raw text.",
        instructions=[
            "Accept raw text as input.",
            "Extract entities (names, dates, organizations) using the provided tool.",
            "Format output as a strictly typed JSON object matching the extracted entity schema.",
            "Do not output conversational text."
        ],
        tools=[extract_json_entities],
    )

def get_code_analysis_capability() -> Agent:
    return Agent(
        model=DeepSeek(),
        name="CodeAnalysisCapability",
        description="Analyzes Python source code for AST complexity metrics.",
        instructions=[
            "Accept Python source code string as input.",
            "Analyze code using the provided AST parsing tool.",
            "Format output as a JSON object containing 'cyclomatic_complexity' and 'function_count'.",
            "Do not output conversational text."
        ],
        tools=[ast_parse_complexity],
    )
