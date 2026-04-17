# tools/code_analyzer.py
import ast
import json

def ast_parse_complexity(code: str) -> str:
    """Mock AST parser returning complexity."""
    try:
        tree = ast.parse(code)
        func_count = sum(isinstance(node, ast.FunctionDef) for node in ast.walk(tree))
    except SyntaxError:
        func_count = -1
    return json.dumps({"cyclomatic_complexity": 5, "function_count": func_count})
