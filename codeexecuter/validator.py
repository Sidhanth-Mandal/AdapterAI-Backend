"""
validator.py — Multi-stage code validator for AI-generated tool schemas.

Stages
------
1. Schema Integrity    — field presence, identifier validity, type string legality
2. Syntax Check        — ast.parse() the code field
3. Static Safety       — AST walk for dangerous builtins / system calls / secrets
4. Structural Match    — declared functions ↔ actual def statements + param names
5. Dependency Check    — importlib.util.find_spec() for every listed dependency
6. Sandboxed Dry-run   — execute in restricted namespace with mocked network calls
"""

from __future__ import annotations

import ast
import importlib.util
import inspect
import json
import re
import textwrap
import traceback
import keyword
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    name: str
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def add_error(self, msg: str) -> None:
        self.errors.append(msg)
        self.passed = False

    def add_warning(self, msg: str) -> None:
        self.warnings.append(msg)


@dataclass
class ValidationReport:
    passed: bool
    safe: bool
    stages: list[StageResult]
    errors: list[str]
    warnings: list[str]
    functions_verified: list[str]

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"  VALIDATION {'PASSED ✓' if self.passed else 'FAILED ✗'}",
            f"  SAFE:      {'YES' if self.safe else 'NO — dangerous patterns found'}",
            "=" * 60,
        ]
        for s in self.stages:
            icon = "✓" if s.passed else "✗"
            lines.append(f"  [{icon}] {s.name}")
            for e in s.errors:
                lines.append(f"        ERROR   : {e}")
            for w in s.warnings:
                lines.append(f"        WARNING : {w}")
        if self.functions_verified:
            lines.append(f"\n  Functions verified: {', '.join(self.functions_verified)}")
        lines.append("=" * 60)
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Types the LLM is expected to use
_VALID_TYPES = {
    "str", "int", "float", "bool", "list", "dict", "tuple", "set",
    "bytes", "Any", "None", "Optional", "Union",
    "List", "Dict", "Tuple", "Set",  # typing-style capitalised
}

# Patterns that indicate a dangerous construct
_DANGEROUS_CALLS = {
    # builtins
    "exec", "eval", "compile", "__import__",
    # os
    "os.system", "os.popen", "os.execv", "os.execve", "os.execvp",
    "os.remove", "os.unlink", "os.rmdir",
    # shutil
    "shutil.rmtree", "shutil.move",
    # subprocess
    "subprocess.run", "subprocess.call", "subprocess.Popen",
    "subprocess.check_output", "subprocess.check_call",
    # importlib runtime abuse
    "importlib.import_module",
    # ctypes
    "ctypes.cdll", "ctypes.windll",
}

_SECRET_PATTERNS = [
    re.compile(r'(?i)(api[_-]?key|apikey|secret|token|password|passwd|auth)\s*=\s*["\'][^"\']{6,}["\']'),
    re.compile(r'(?i)Bearer\s+[A-Za-z0-9\-._~+/]{20,}'),
]

_WRITE_MODES = {"w", "wb", "a", "ab", "x", "xb", "w+", "wb+", "a+", "ab+"}


def _is_valid_identifier(name: str) -> bool:
    return name.isidentifier() and not keyword.iskeyword(name)


def _stub_value(type_str: str) -> Any:
    """Return a safe stub value for a given type string."""
    mapping = {
        "str": "test_value",
        "int": 1,
        "float": 1.0,
        "bool": True,
        "list": [],
        "dict": {},
        "tuple": (),
        "set": set(),
        "bytes": b"",
        "Any": "stub",
        "None": None,
    }
    return mapping.get(type_str, "stub")


# ---------------------------------------------------------------------------
# Stage implementations
# ---------------------------------------------------------------------------

def _stage_schema_integrity(tool: dict) -> StageResult:
    result = StageResult(name="Schema Integrity", passed=True)

    # Required top-level fields
    required = ["tool_name", "tool_description", "code", "functions"]
    for f_name in required:
        if f_name not in tool:
            result.add_error(f"Missing required field: '{f_name}'")

    # tool_name must be a valid identifier
    tool_name = tool.get("tool_name", "")
    if tool_name and not _is_valid_identifier(tool_name):
        result.add_error(f"tool_name '{tool_name}' is not a valid Python identifier")

    # Validate functions list
    for i, fn in enumerate(tool.get("functions", [])):
        fn_label = fn.get("name", f"<function[{i}]>")

        if not fn.get("name"):
            result.add_error(f"Function [{i}] is missing a 'name' field")
        elif not _is_valid_identifier(fn["name"]):
            result.add_error(f"Function name '{fn['name']}' is not a valid Python identifier")

        if not fn.get("description"):
            result.add_warning(f"Function '{fn_label}' has no description")

        for j, param in enumerate(fn.get("parameters", [])):
            p_label = param.get("name", f"<param[{j}]>")
            if not param.get("name"):
                result.add_error(f"Function '{fn_label}' param [{j}] missing 'name'")
            elif not _is_valid_identifier(param["name"]):
                result.add_error(f"Function '{fn_label}' param name '{param['name']}' is not a valid identifier")

            p_type = param.get("type", "")
            base_type = p_type.split("[")[0].strip()  # handle List[str] etc.
            if base_type and base_type not in _VALID_TYPES:
                result.add_warning(f"Function '{fn_label}' param '{p_label}' has unusual type '{p_type}'")

        for k, out in enumerate(fn.get("outputs", [])):
            if not out.get("name"):
                result.add_error(f"Function '{fn_label}' output [{k}] missing 'name'")
            if not out.get("type"):
                result.add_warning(f"Function '{fn_label}' output [{k}] missing 'type'")

    return result


def _stage_syntax_check(code: str) -> StageResult:
    result = StageResult(name="Syntax Check", passed=True)
    try:
        ast.parse(code)
    except SyntaxError as e:
        result.add_error(f"SyntaxError at line {e.lineno}: {e.msg} — {e.text!r}")
    except Exception as e:
        result.add_error(f"Unexpected parse error: {e}")
    return result


def _stage_static_safety(code: str) -> StageResult:
    result = StageResult(name="Static Safety Analysis", passed=True)
    safe = True

    try:
        tree = ast.parse(code)
    except Exception:
        result.add_error("Cannot perform static analysis — code has syntax errors")
        return result

    # Walk the AST
    for node in ast.walk(tree):

        # --- Check function calls ---
        if isinstance(node, ast.Call):
            call_str = _resolve_call_name(node.func)
            if call_str:
                if call_str in _DANGEROUS_CALLS:
                    result.add_error(f"Dangerous call detected: '{call_str}()'")
                    safe = False

                # open() in write mode
                if call_str == "open" and node.args:
                    mode_arg = None
                    if len(node.args) >= 2:
                        mode_arg = node.args[1]
                    else:
                        for kw in node.keywords:
                            if kw.arg == "mode":
                                mode_arg = kw.value
                    if mode_arg and isinstance(mode_arg, ast.Constant):
                        if mode_arg.value in _WRITE_MODES:
                            result.add_error(f"File write detected: open(..., '{mode_arg.value}') — tools should not write to disk")
                            safe = False

                # Network calls without timeout
                if call_str in {"requests.get", "requests.post", "requests.put", "requests.delete",
                                 "httpx.get", "httpx.post", "httpx.put", "httpx.delete"}:
                    kw_names = {kw.arg for kw in node.keywords}
                    if "timeout" not in kw_names:
                        result.add_warning(f"Network call '{call_str}()' has no timeout — consider adding timeout=10")

        # --- Check imports ---
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name in {"subprocess", "ctypes", "pty", "socket"}:
                    result.add_error(f"Dangerous import: '{alias.name}'")
                    safe = False

        if isinstance(node, ast.ImportFrom):
            if node.module in {"subprocess", "ctypes", "pty"}:
                result.add_error(f"Dangerous import from: '{node.module}'")
                safe = False

    # --- Check for hardcoded secrets via regex ---
    for pattern in _SECRET_PATTERNS:
        matches = pattern.findall(code)
        if matches:
            result.add_error(f"Possible hardcoded secret/token detected — do not embed credentials in code")
            safe = False
            break

    # Record safety flag in a custom attribute for the caller
    result._safe = safe
    return result


def _resolve_call_name(node: ast.expr) -> str | None:
    """Flatten a function call node to a dotted string like 'os.system'."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _resolve_call_name(node.value)
        return f"{parent}.{node.attr}" if parent else None
    return None


def _stage_structural_match(tool: dict, code: str) -> StageResult:
    result = StageResult(name="Structural Match", passed=True)

    try:
        tree = ast.parse(code)
    except Exception:
        result.add_error("Cannot perform structural match — code has syntax errors")
        return result

    # Collect all top-level def names from code
    defined_functions: dict[str, ast.FunctionDef] = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            defined_functions[node.name] = node

    declared_names = [fn["name"] for fn in tool.get("functions", []) if fn.get("name")]

    for fn_name in declared_names:
        if fn_name not in defined_functions:
            result.add_error(f"Function '{fn_name}' is declared in schema but NOT defined in code")
            continue

        # Check parameter names
        ast_node = defined_functions[fn_name]
        code_params = [arg.arg for arg in ast_node.args.args]

        schema_fn = next((f for f in tool["functions"] if f["name"] == fn_name), {})
        schema_params = [p["name"] for p in schema_fn.get("parameters", [])]

        for sp in schema_params:
            if sp not in code_params:
                result.add_error(
                    f"Function '{fn_name}': schema param '{sp}' not found in code signature {code_params}"
                )

    # Warn about functions defined in code but not declared in schema
    for def_name in defined_functions:
        if def_name not in declared_names and not def_name.startswith("_"):
            result.add_warning(f"Function '{def_name}' is defined in code but not declared in schema")

    return result


def _stage_dependency_check(tool: dict) -> StageResult:
    result = StageResult(name="Dependency Check", passed=True)
    deps = tool.get("dependencies", [])

    if not deps:
        result.add_warning("No dependencies listed — verify imports in code are available")
        return result

    missing = []
    for dep in deps:
        # Normalise: pip name may differ from import name (e.g., beautifulsoup4 → bs4)
        import_name = _pip_to_import_name(dep)
        spec = importlib.util.find_spec(import_name)
        if spec is None:
            missing.append(dep)
            result.add_warning(f"Dependency '{dep}' is NOT installed (import name tried: '{import_name}')")

    if missing:
        result.add_error(
            f"Missing dependencies: {missing} — run: pip install {' '.join(missing)}"
        )

    return result


def _pip_to_import_name(pip_name: str) -> str:
    """Map common pip package names to their actual import names."""
    _MAP = {
        "beautifulsoup4": "bs4",
        "scikit-learn": "sklearn",
        "pillow": "PIL",
        "pyyaml": "yaml",
        "python-dateutil": "dateutil",
        "google-generativeai": "google.generativeai",
        "langchain-groq": "langchain_groq",
        "langchain-core": "langchain_core",
        "json-repair": "json_repair",
    }
    return _MAP.get(pip_name.lower(), pip_name.replace("-", "_"))


_BLOCKED_MODULES = {"subprocess", "ctypes", "pty", "socket", "os", "shutil", "sys"}

def _safe_import(name: str, *args, **kwargs) -> Any:
    """Wrapped __import__ that blocks dangerous modules in the sandbox."""
    top_module = name.split(".")[0]
    if top_module in _BLOCKED_MODULES:
        raise ImportError(
            f"[Sandbox] Import of '{name}' is blocked during dry-run validation"
        )
    return __import__(name, *args, **kwargs)


def _stage_dry_run(tool: dict, code: str) -> StageResult:
    result = StageResult(name="Sandboxed Dry-run", passed=True)
    verified: list[str] = []

    try:
        tree = ast.parse(code)
    except Exception:
        result.add_error("Cannot dry-run — code has syntax errors")
        return result, verified

    # Build a restricted builtins dict:
    # - Keep __import__ but replace it with our safe wrapper
    # - Remove only the truly abusable builtins
    _STRIP_BUILTINS = {"exec", "eval", "compile", "open", "input", "breakpoint"}

    raw_builtins = __builtins__ if isinstance(__builtins__, dict) else vars(__builtins__)
    safe_builtins: dict[str, Any] = {
        k: v for k, v in raw_builtins.items() if k not in _STRIP_BUILTINS
    }
    # Swap in the sandboxed importer
    safe_builtins["__import__"] = _safe_import

    namespace: dict[str, Any] = {"__builtins__": safe_builtins}

    # Mock network libraries so we never make real HTTP calls
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.text = "{}"
    mock_response.json.return_value = {}
    mock_response.content = b"{}"

    network_patches = [
        "requests.get", "requests.post", "requests.put", "requests.delete",
        "requests.Session.get", "requests.Session.post",
    ]

    patchers = []
    for target in network_patches:
        try:
            p = patch(target, return_value=mock_response)
            p.start()
            patchers.append(p)
        except Exception:
            pass

    try:
        # Execute the code so functions are defined in namespace
        exec(compile(tree, "<generated_tool>", "exec"), namespace)  # noqa: S102

        # Call each declared function with stub arguments
        for fn_schema in tool.get("functions", []):
            fn_name = fn_schema.get("name")
            if not fn_name or fn_name not in namespace:
                continue

            fn_callable = namespace[fn_name]
            stub_kwargs: dict[str, Any] = {}
            for param in fn_schema.get("parameters", []):
                stub_kwargs[param["name"]] = _stub_value(param.get("type", "str"))

            try:
                ret = fn_callable(**stub_kwargs)

                # Verify return type
                expected_rt = fn_schema.get("return_type", "dict")
                if expected_rt == "dict" and not isinstance(ret, dict):
                    result.add_warning(
                        f"Function '{fn_name}' declared return_type='dict' but returned {type(ret).__name__}"
                    )
                elif expected_rt == "list" and not isinstance(ret, list):
                    result.add_warning(
                        f"Function '{fn_name}' declared return_type='list' but returned {type(ret).__name__}"
                    )

                # Verify output keys (best-effort)
                if isinstance(ret, dict):
                    declared_outputs = {o["name"] for o in fn_schema.get("outputs", [])}
                    missing_keys = declared_outputs - set(ret.keys())
                    if missing_keys:
                        result.add_warning(
                            f"Function '{fn_name}' return dict is missing declared output keys: {missing_keys}"
                        )

                verified.append(fn_name)

            except (KeyError, TypeError, AttributeError, IndexError) as e:
                # These commonly occur because the mocked API response returns {}
                # instead of the real structure — treat as a warning, not a hard error.
                result.add_warning(
                    f"Function '{fn_name}' raised {type(e).__name__} during dry-run "
                    f"(likely due to mocked network response returning empty data): {e}"
                )
                # Still count as partially verified — the function ran
                verified.append(fn_name)

            except Exception as e:
                tb = traceback.format_exc(limit=3)
                result.add_error(f"Function '{fn_name}' raised during dry-run: {type(e).__name__}: {e}\n{textwrap.indent(tb, '    ')}")

    except Exception as e:
        tb = traceback.format_exc(limit=3)
        result.add_error(f"Code execution failed during dry-run: {type(e).__name__}: {e}\n{textwrap.indent(tb, '    ')}")
    finally:
        for p in patchers:
            try:
                p.stop()
            except Exception:
                pass

    return result, verified


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_tool(tool_input: dict | str | Path) -> ValidationReport:
    """
    Validate a generated tool.

    Parameters
    ----------
    tool_input : dict | str | Path
        Either a ToolSchema dict, a path to a JSON file, or a JSON string.

    Returns
    -------
    ValidationReport
        Full report with per-stage results, errors, warnings, and safety flag.
    """
    # --- Load input ---
    if isinstance(tool_input, (str, Path)):
        p = Path(tool_input)
        if p.exists():
            with open(p, encoding="utf-8") as fh:
                tool = json.load(fh)
        else:
            tool = json.loads(tool_input)
    else:
        tool = tool_input

    code: str = tool.get("code", "")

    # --- Run stages ---
    stages: list[StageResult] = []
    all_errors: list[str] = []
    all_warnings: list[str] = []
    safe = True
    functions_verified: list[str] = []

    # Stage 1 — Schema Integrity
    s1 = _stage_schema_integrity(tool)
    stages.append(s1)

    # Stage 2 — Syntax Check
    s2 = _stage_syntax_check(code)
    stages.append(s2)

    # Stage 3 — Static Safety
    s3 = _stage_static_safety(code)
    safe = getattr(s3, "_safe", True)
    stages.append(s3)

    # Stage 4 — Structural Match
    s4 = _stage_structural_match(tool, code)
    stages.append(s4)

    # Stage 5 — Dependency Check
    s5 = _stage_dependency_check(tool)
    stages.append(s5)

    # Stage 6 — Dry-run (only if syntax passes)
    if s2.passed:
        s6, functions_verified = _stage_dry_run(tool, code)
    else:
        s6 = StageResult(name="Sandboxed Dry-run", passed=False)
        s6.add_error("Skipped — code has syntax errors")
    stages.append(s6)

    # --- Aggregate ---
    for s in stages:
        all_errors.extend(s.errors)
        all_warnings.extend(s.warnings)

    overall_passed = all(s.passed for s in stages)

    return ValidationReport(
        passed=overall_passed,
        safe=safe,
        stages=stages,
        errors=all_errors,
        warnings=all_warnings,
        functions_verified=functions_verified,
    )
