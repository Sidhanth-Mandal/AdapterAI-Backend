"""
Docker_exec/runner.py — Loads a compiled tool file and executes a function.

Takes the path written by tool_compiler.py, dynamically imports it,
looks up the requested function, calls it with the provided args, and
returns the result.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from typing import Any, Union


def run_function(
    compiled_file: Path,
    function_name: str,
    args: Union[list, dict, Any],
) -> Any:
    """
    Dynamically import a compiled tool file and call a function from it.

    Parameters
    ----------
    compiled_file : Path
        Absolute path to the compiled ``.py`` file (output of tool_compiler).
    function_name : str
        Name of the function to call inside the compiled file.
    args : list | dict | any
        Arguments to pass to the function:
          - list  →  positional:  fn(*args)
          - dict  →  keyword:     fn(**args)
          - other →  single pos:  fn(args)

    Returns
    -------
    Any
        Whatever the function returns.

    Raises
    ------
    FileNotFoundError
        If ``compiled_file`` does not exist.
    AttributeError
        If ``function_name`` is not defined in the compiled file.
    """
    if not compiled_file.exists():
        raise FileNotFoundError(
            f"Compiled tool file not found: {compiled_file}"
        )

    # ── Dynamic import ───────────────────────────────────────────────────────
    module_name = compiled_file.stem  # e.g. "cricket_stats_tool"

    # Remove any previously cached version so fresh code is always used
    if module_name in sys.modules:
        del sys.modules[module_name]

    spec   = importlib.util.spec_from_file_location(module_name, compiled_file)
    module = importlib.util.module_from_spec(spec)       # type: ignore[arg-type]
    sys.modules[module_name] = module
    spec.loader.exec_module(module)                       # type: ignore[union-attr]

    # ── Look up the function ─────────────────────────────────────────────────
    fn = getattr(module, function_name, None)
    if fn is None or not callable(fn):
        available = [
            name for name in dir(module)
            if callable(getattr(module, name)) and not name.startswith("_")
        ]
        raise AttributeError(
            f"Function '{function_name}' not found in '{compiled_file.name}'. "
            f"Available callables: {available}"
        )

    # ── Call and return ──────────────────────────────────────────────────────
    print(f"[Runner] Calling {function_name} with args={args!r}")

    if isinstance(args, dict):
        return fn(**args)
    elif isinstance(args, list):
        return fn(*args)
    else:
        return fn(args)
