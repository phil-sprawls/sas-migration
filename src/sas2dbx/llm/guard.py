"""Static checks on generated code before it is ever executed."""
from __future__ import annotations

import ast

FORBIDDEN_ATTRS = {"rdd": "RDD API (not available on serverless / shared clusters)",
                   "toPandas": "toPandas() pulls data to the driver",
                   "saveAsTable": "writes outside ctx", "insertInto": "writes outside ctx"}
FORBIDDEN_CALLS = {"eval", "exec", "open", "__import__", "compile", "input"}
ALLOWED_IMPORTS = {"math", "re", "datetime", "decimal", "itertools", "functools", "collections", "statistics",
                   "numpy", "scipy", "pandas", "pyspark", "sas2dbx"}   # all present on Databricks runtimes


def check_code(code: str) -> list[str]:
    """Return a list of problems (empty = OK)."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"syntax error: {e.msg} (line {e.lineno})"]
    problems = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute):
            if node.attr in FORBIDDEN_ATTRS:
                problems.append(f"line {node.lineno}: {FORBIDDEN_ATTRS[node.attr]}")
            if node.attr == "set" and isinstance(node.value, ast.Attribute) and node.value.attr == "conf":
                problems.append(f"line {node.lineno}: spark.conf.set changes session behaviour for other code")
        elif isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in FORBIDDEN_CALLS:
            problems.append(f"line {node.lineno}: call to {node.func.id}()")
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            mods = [a.name for a in node.names] if isinstance(node, ast.Import) else [node.module or ""]
            for m in mods:
                if m.split(".")[0] not in ALLOWED_IMPORTS:
                    problems.append(f"line {node.lineno}: import of {m} is not allowed")
    if "ctx.put(" not in code:
        problems.append("the code never writes an output with ctx.put")
    return problems
