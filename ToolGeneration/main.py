"""
main.py — CLI entry point for the ToolGeneration pipeline.

Usage
-----
    python main.py <template_id>

Example
-------
    python main.py tem00001
"""

import json
import sys

from .pipeline import generate_tool


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python main.py <template_id>", file=sys.stderr)
        sys.exit(1)

    template_id: str = sys.argv[1]
    result: dict = generate_tool(template_id)

    print(json.dumps(result, indent=2))

    if result.get("status") != "success":
        sys.exit(1)


if __name__ == "__main__":
    main()