"""
validator.py — Multi-stage code validator for AI-generated tool schemas.

Stages
------
1. Schema Integrity   — field presence, identifier validity, type string legality
2. Syntax Check       — ast.parse() the code field
3. Static Safety      — AST walk for dangerous builtins / system calls / secrets
4. Structural Match   — declared functions <-> actual def statements + param names
5. Dependency Check   — importlib.util.find_spec() for every listed dependency

Note: runtime execution / sandboxed dry-run is handled by the separate executor.
"""

from __future__ import annotations

import ast
import json
import keyword
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class StageResult:
    name: str
    passed: bool
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    is_safe: bool = True  # FIX #4: proper declared field instead of dynamic _safe attribute

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

    def summary(self) -> str:
        lines = [
            "=" * 60,
            f"  VALIDATION {'PASSED [OK]' if self.passed else 'FAILED [!!]'}",
            f"  SAFE:      {'YES' if self.safe else 'NO -- dangerous patterns found'}",
            "=" * 60,
        ]
        for s in self.stages:
            icon = "+" if s.passed else "-"
            lines.append(f"  [{icon}] {s.name}")
            for e in s.errors:
                lines.append(f"        ERROR   : {e}")
            for w in s.warnings:
                lines.append(f"        WARNING : {w}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.summary()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

# Types the LLM is expected to use
_VALID_TYPES = {
    "str", "int", "float", "bool", "list", "dict", "tuple", "set",
    "bytes", "Any", "None", "Optional", "Union",
    "List", "Dict", "Tuple", "Set",  # typing-style capitalised
}

# S2: valid return types for schema validation
_VALID_RETURN_TYPES = {
    "str", "int", "float", "bool", "list", "dict", "tuple", "set",
    "bytes", "Any", "None",
    "List", "Dict", "Tuple", "Set",
}

# Patterns that indicate a dangerous construct
_DANGEROUS_CALLS = {
    # builtins
    "exec", "eval", "compile", "__import__",
    # os
    "os.system", "os.popen", "os.execv", "os.execve", "os.execvp",
    "os.remove", "os.unlink", "os.rmdir",
    # shutil — also caught by blocking in sandbox but listed here for static check
    "shutil.rmtree", "shutil.move",
    # subprocess
    "subprocess.run", "subprocess.call", "subprocess.Popen",
    "subprocess.check_output", "subprocess.check_call",
    # importlib runtime abuse
    "importlib.import_module",
    # ctypes
    "ctypes.cdll", "ctypes.windll",
    # sys — FIX #6: added to match sandbox blocking for consistency
    "sys.exit", "sys.modules",
}

_SECRET_PATTERNS = [
    re.compile(r'(?i)(api[_-]?key|apikey|secret|token|password|passwd|auth)\s*=\s*["\'][^"\'"]{6,}["\']'),
    re.compile(r'(?i)Bearer\s+[A-Za-z0-9\-._~+/]{20,}'),
]

_WRITE_MODES = {"w", "wb", "a", "ab", "x", "xb", "w+", "wb+", "a+", "ab+"}


def _is_valid_identifier(name: str) -> bool:
    return name.isidentifier() and not keyword.iskeyword(name)


def _stub_value(type_str: str) -> Any:
    """Return a safe stub value for a given type string.

    S6: handles complex generic types like List[str], Optional[int], Dict[str, Any]
    by stripping the bracket suffix and resolving the base type.
    """
    base = type_str.split("[")[0].strip()   # "List[str]" → "List", "Optional[int]" → "Optional"
    mapping = {
        "str": "test_value",
        "int": 1,
        "float": 1.0,
        "bool": True,
        "list": [],
        "List": [],
        "dict": {},
        "Dict": {},
        "tuple": (),
        "Tuple": (),
        "set": set(),
        "Set": set(),
        "bytes": b"",
        "Any": "stub",
        "None": None,
        "Optional": None,   # safest default for Optional[X] — avoids type errors
        "Union": "stub",
    }
    return mapping.get(base, mapping.get(type_str, "stub"))


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

        # S2: validate return_type is a known type
        rt = fn.get("return_type", "")
        if rt and rt.split("[")[0].strip() not in _VALID_RETURN_TYPES:
            result.add_warning(f"Function '{fn_label}' has unusual return_type: '{rt}'")

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
            # S3: wildcard import detection
            for alias in node.names:
                if alias.name == "*":
                    result.add_warning(
                        f"Wildcard import 'from {node.module} import *' detected — "
                        "avoid namespace pollution"
                    )

    # --- Check for hardcoded secrets via regex ---
    for pattern in _SECRET_PATTERNS:
        matches = pattern.findall(code)
        if matches:
            result.add_error("Possible hardcoded secret/token detected — do not embed credentials in code")
            safe = False
            break

    # FIX #4: use proper declared field instead of dynamic attribute
    result.is_safe = safe
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

    # FIX #3 / S1: use iter_child_nodes instead of ast.walk to only collect
    # TOP-LEVEL module-scope function definitions. ast.walk previously descended
    # into all nested scopes (closures, decorators, inner functions) causing
    # false-positive warnings for helper closures like 'decorator' and 'wrapper'.
    defined_functions: dict[str, ast.FunctionDef] = {
        node.name: node
        for node in ast.iter_child_nodes(tree)
        if isinstance(node, ast.FunctionDef)
    }

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

    # Warn about public functions defined at module scope but not declared in schema
    for def_name in defined_functions:
        if def_name not in declared_names and not def_name.startswith("_"):
            result.add_warning(f"Function '{def_name}' is defined in code but not declared in schema")

    return result


def _stage_dependency_check(tool: dict, code: str) -> StageResult:
    """
    Scan the code for third-party imports and ensure every one is listed
    in the tool schema's 'dependencies' field.

    Behaviour
    ---------
    - Stdlib modules are silently ignored.
    - Non-stdlib modules already listed in 'dependencies' pass fine.
    - Non-stdlib modules NOT listed but considered safe are auto-added to
      ``tool['dependencies']`` and reported as a warning.
    - Modules in _HARMFUL_MODULES (should already be caught by Stage 3)
      are flagged as errors and are NOT auto-added.

    Note: this stage does NOT check whether packages are installed on the
    current system — that is the executor's responsibility.
    """
    result = StageResult(name="Dependency Check", passed=True)

    try:
        tree = ast.parse(code)
    except Exception:
        result.add_error("Cannot check dependencies — code has syntax errors")
        return result

    # Python 3.10+ provides a complete stdlib set; fall back to a curated list
    stdlib_mods: frozenset[str] = getattr(sys, "stdlib_module_names", _STDLIB_FALLBACK)

    # Collect every top-level module name that appears in an import statement
    imported_tops: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                top = alias.name.split(".")[0]
                if top:
                    imported_tops.add(top)
        elif isinstance(node, ast.ImportFrom) and node.module:
            top = node.module.split(".")[0]
            if top:
                imported_tops.add(top)

    # Only care about third-party modules
    third_party = {m for m in imported_tops if m not in stdlib_mods}

    # Build the set of import-names already covered by declared dependencies
    declared_deps: list[str] = tool.setdefault("dependencies", [])
    declared_import_names: set[str] = {_pip_to_import_name(d) for d in declared_deps}

    auto_added: list[str] = []

    for mod in sorted(third_party):
        if mod in declared_import_names:
            continue  # already covered

        if mod in _HARMFUL_MODULES:
            result.add_error(
                f"Module '{mod}' is imported but is a restricted/harmful module — "
                "remove this import (it should have been caught by Stage 3)"
            )
            continue

        # Safe: auto-add to schema dependencies
        pip_name = _import_to_pip_name(mod)
        declared_deps.append(pip_name)
        declared_import_names.add(mod)
        auto_added.append(pip_name)

    if auto_added:
        result.add_warning(
            f"Auto-added undeclared dependencies to schema: {auto_added}"
        )

    return result


# ---------------------------------------------------------------------------
# Pip <-> import name translation tables
# ---------------------------------------------------------------------------

# Modules that should never be auto-added (harmful; Stage 3 should have caught them)
_HARMFUL_MODULES = {"subprocess", "ctypes", "pty", "socket"}

# pip install name  →  import name
_PIP_TO_IMPORT: dict[str, str] = {
    "beautifulsoup4":     "bs4",
    "scikit-learn":       "sklearn",
    "pillow":             "PIL",
    "pyyaml":             "yaml",
    "python-dateutil":    "dateutil",
    "google-generativeai": "google",
    "langchain-groq":     "langchain_groq",
    "langchain-core":     "langchain_core",
    "json-repair":        "json_repair",
    "opencv-python":      "cv2",
    "setuptools":         "pkg_resources",
}

# import name  →  pip install name  (reverse of above)
_IMPORT_TO_PIP: dict[str, str] = {v: k for k, v in _PIP_TO_IMPORT.items()}

# Large but common stdlib fallback for Python < 3.10
_STDLIB_FALLBACK: frozenset[str] = frozenset({
    "abc", "ast", "asyncio", "base64", "binascii", "builtins", "cmath",
    "collections", "concurrent", "contextlib", "copy", "csv", "dataclasses",
    "datetime", "decimal", "email", "enum", "fnmatch", "fractions",
    "ftplib", "functools", "gc", "getpass", "glob", "gzip", "hashlib",
    "heapq", "hmac", "html", "http", "idlelib", "importlib", "inspect",
    "io", "ipaddress", "itertools", "json", "keyword", "linecache",
    "logging", "lzma", "math", "mimetypes", "multiprocessing", "operator",
    "os", "pathlib", "pickle", "pkgutil", "platform", "pprint", "queue",
    "random", "re", "secrets", "select", "shelve", "shutil", "signal",
    "smtplib", "socket", "sqlite3", "stat", "statistics", "string",
    "struct", "subprocess", "sys", "sysconfig", "tarfile", "tempfile",
    "textwrap", "threading", "time", "timeit", "token", "tokenize",
    "tomllib", "traceback", "types", "typing", "unicodedata", "unittest",
    "urllib", "uuid", "warnings", "weakref", "xml", "zipfile", "zlib",
})


def _pip_to_import_name(pip_name: str) -> str:
    """Map a pip package name to the top-level import name used in code."""
    return _PIP_TO_IMPORT.get(pip_name.lower(), pip_name.replace("-", "_"))


def _import_to_pip_name(import_name: str) -> str:
    """Map a top-level import name back to the pip package name."""
    return _IMPORT_TO_PIP.get(import_name, import_name)


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
        str(report) or print(report) outputs the human-readable summary.

    Note
    ----
    Runtime execution is intentionally not performed here.
    Use the executor module for sandboxed execution and function-level testing.
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

    # Stage 1 — Schema Integrity
    s1 = _stage_schema_integrity(tool)
    stages.append(s1)

    # Stage 2 — Syntax Check
    s2 = _stage_syntax_check(code)
    stages.append(s2)

    # Stage 3 — Static Safety
    s3 = _stage_static_safety(code)
    safe = s3.is_safe
    stages.append(s3)

    # Stage 4 — Structural Match
    s4 = _stage_structural_match(tool, code)
    stages.append(s4)

    # Stage 5 — Dependency Check
    s5 = _stage_dependency_check(tool, code)
    stages.append(s5)

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
    )
