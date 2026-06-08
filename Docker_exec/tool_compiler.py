"""
Docker_exec/tool_compiler.py — Converts tool_json into an executable Python file.

Steps
-----
1. Extract ``tool_json["code"]``  →  write to ``/tmp/compiled_tools/<tool_name>.py``
2. Extract ``tool_json["dependencies"]``  →  ``pip install`` them into the container
3. Return the Path of the compiled file

This runs inside the Docker container, so pip installs are persistent for
the lifetime of the container.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

# Where compiled tool files are written inside the container
_TOOLS_DIR = Path("/tmp/compiled_tools")

# Track which deps have already been installed in this container session
# to avoid redundant pip calls on repeated requests
_installed_deps: set[str] = set()


def compile_tool(tool_json: dict) -> Path:
    """
    Compile a tool_json dict into an executable Python file.

    Parameters
    ----------
    tool_json : dict
        Full content of generated_tool.json.

    Returns
    -------
    Path
        Absolute path to the compiled ``.py`` file inside the container.

    Raises
    ------
    ValueError
        If ``tool_json`` has no ``code`` field or the code is empty.
    subprocess.CalledProcessError
        If ``pip install`` fails for one or more dependencies.
    """
    tool_name = tool_json.get("tool_name", "unnamed_tool")
    code      = tool_json.get("code", "").strip()

    if not code:
        raise ValueError(
            f"tool_json for '{tool_name}' has no 'code' field or code is empty."
        )

    # ── Write compiled file ──────────────────────────────────────────────────
    _TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    compiled_path = _TOOLS_DIR / f"{tool_name}.py"
    compiled_path.write_text(code, encoding="utf-8")
    print(f"[Compiler] Written: {compiled_path}")

    # ── Install dependencies ─────────────────────────────────────────────────
    deps: list[str] = tool_json.get("dependencies", [])
    new_deps = [d for d in deps if d not in _installed_deps]

    if new_deps:
        print(f"[Compiler] Installing dependencies: {new_deps}")
        result = subprocess.run(
            [
                sys.executable, "-m", "pip", "install",
                "--quiet",
                "--no-cache-dir",
                "--disable-pip-version-check",
            ] + new_deps,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"pip install failed for {new_deps}:\n"
                f"{result.stderr or result.stdout}"
            )
        _installed_deps.update(new_deps)
        print(f"[Compiler] Installed: {new_deps}")
    else:
        print(f"[Compiler] All dependencies already installed: {deps}")

    return compiled_path
