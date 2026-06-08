"""
validate_cli.py -- Color-coded CLI runner for the tool validator.

Usage
-----
    python codeexecuter/validate_cli.py <path_to_tool.json>
    python codeexecuter/validate_cli.py codeexecuter/generated_tool.json

Exit codes
----------
    0  — all stages passed
    1  — one or more stages failed
    2  — usage error / file not found
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure codeexecuter/ is importable when run from project root
sys.path.insert(0, str(Path(__file__).parent))

from .validator import validate_tool, ValidationReport, StageResult

# ---------------------------------------------------------------------------
# ANSI colour helpers (work on Windows 10+ with ENABLE_VIRTUAL_TERMINAL)
# ---------------------------------------------------------------------------

try:
    import colorama
    colorama.init(autoreset=True)
    _COLORAMA = True
except ImportError:
    _COLORAMA = False

_RESET  = "\033[0m"
_BOLD   = "\033[1m"
_RED    = "\033[91m"
_GREEN  = "\033[92m"
_YELLOW = "\033[93m"
_CYAN   = "\033[96m"
_WHITE  = "\033[97m"
_DIM    = "\033[2m"

# ASCII-safe symbols (avoids UnicodeEncodeError on Windows cp1252)
_PASS = "[PASS]"
_FAIL = "[FAIL]"
_WARN = "[WARN]"

def _c(text: str, *codes: str) -> str:
    """Wrap text in ANSI escape codes (no-op if colour unavailable)."""
    if not sys.stdout.isatty() and not _COLORAMA:
        return text
    return "".join(codes) + text + _RESET


# ---------------------------------------------------------------------------
# Pretty printer
# ---------------------------------------------------------------------------

def _print_stage(stage: StageResult) -> None:
    if stage.passed:
        badge = _c(_PASS, _GREEN, _BOLD)
        label = _c(stage.name, _GREEN)
    else:
        badge = _c(_FAIL, _RED, _BOLD)
        label = _c(stage.name, _RED)

    print(f"  {badge} {label}")

    for err in stage.errors:
        # Indent multi-line errors nicely
        lines = err.splitlines()
        print(f"         {_c('ERROR', _RED, _BOLD)}   : {lines[0]}")
        for extra in lines[1:]:
            print(f"                   {_c(extra, _DIM)}")

    for warn in stage.warnings:
        print(f"         {_c(_WARN, _YELLOW, _BOLD)} : {warn}")


def _print_report(report: ValidationReport, json_path: str) -> None:
    width = 64

    print()
    print(_c("=" * width, _CYAN))
    title = "  TOOL VALIDATOR REPORT"
    print(_c(title, _CYAN, _BOLD))
    print(_c(f"  File: {json_path}", _DIM))
    print(_c("=" * width, _CYAN))

    print()
    for stage in report.stages:
        _print_stage(stage)
        print()

    print(_c("-" * width, _DIM))

    # Functions verified
    if report.functions_verified:
        fns = ", ".join(report.functions_verified)
        print(f"  {_c('Functions dry-run verified', _WHITE, _BOLD)}: {_c(fns, _CYAN)}")
    else:
        print(f"  {_c('Functions dry-run verified', _WHITE, _BOLD)}: {_c('none', _DIM)}")

    print()

    # Safety badge
    if report.safe:
        safety_badge = _c("  [SAFE]  ", _GREEN, _BOLD)
    else:
        safety_badge = _c("  [UNSAFE] -- dangerous patterns detected  ", _RED, _BOLD)
    print(safety_badge)

    # Overall result
    print()
    if report.passed:
        verdict = _c("  >> VALIDATION PASSED", _GREEN, _BOLD)
    else:
        verdict = _c("  >> VALIDATION FAILED", _RED, _BOLD)
    print(verdict)

    # Summary counts
    n_err = len(report.errors)
    n_warn = len(report.warnings)
    print(_c(f"  Errors: {n_err}   Warnings: {n_warn}", _DIM))
    print(_c("=" * width, _CYAN))
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> int:
    # Ensure stdout can handle any characters even on Windows
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # type: ignore[attr-defined]

    if len(sys.argv) < 2:
        print(
            _c("Usage: ", _BOLD) +
            "python codeexecuter/validate_cli.py <path_to_tool.json>",
            file=sys.stderr,
        )
        return 2

    json_path = sys.argv[1]
    p = Path(json_path)

    if not p.exists():
        print(_c(f"Error: File not found: {json_path}", _RED), file=sys.stderr)
        return 2

    print(_c(f"\n  Running validator on: {json_path} …", _DIM))

    try:
        report = validate_tool(p)
    except json.JSONDecodeError as e:
        print(_c(f"Error: Invalid JSON in '{json_path}': {e}", _RED), file=sys.stderr)
        return 2
    except Exception as e:
        print(_c(f"Unexpected validator error: {e}", _RED), file=sys.stderr)
        return 2

    _print_report(report, json_path)

    return 0 if report.passed else 1


if __name__ == "__main__":
    sys.exit(main())
