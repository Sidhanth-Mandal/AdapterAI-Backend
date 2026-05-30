"""
metadata.py — Tool metadata extraction for the ToolGeneration pipeline.

Single public function
----------------------
extract_tool_metadata(tool_json) → dict
    Returns name, description, and a structured tool_information string
    that summarises every function's inputs and outputs in plain text.
    This string is stored in Templates.tool_information so the orchestrator
    can understand tool capabilities without parsing the full JSON.
"""

from __future__ import annotations


def extract_tool_metadata(tool_json: dict) -> dict:
    """
    Extract human-readable metadata from a generated tool dict.

    Parameters
    ----------
    tool_json : dict
        A validated tool dict that conforms to ToolSchema.

    Returns
    -------
    dict with keys:
        name             : str   tool_json["tool_name"]
        description      : str   tool_json["tool_description"]
        tool_information : str   Multi-line capability summary of all functions
    """
    name: str = tool_json.get("tool_name", "unknown_tool")
    description: str = tool_json.get("tool_description", "")
    functions: list[dict] = tool_json.get("functions", [])

    sections: list[str] = []

    for fn in functions:
        fn_name: str = fn.get("name", "<unnamed>")
        fn_desc: str = fn.get("description", "")
        parameters: list[dict] = fn.get("parameters", [])
        outputs: list[dict] = fn.get("outputs", [])
        return_type: str = fn.get("return_type", "dict")

        lines: list[str] = [
            f"Function: {fn_name}",
            f"Description:",
            f"  {fn_desc}",
            "",
        ]

        if parameters:
            lines.append("Inputs:")
            for p in parameters:
                p_name = p.get("name", "?")
                p_type = p.get("type", "?")
                p_desc = p.get("description", "")
                required_flag = "" if p.get("required", True) else " (optional)"
                if p_desc:
                    lines.append(f"  - {p_name} ({p_type}){required_flag}: {p_desc}")
                else:
                    lines.append(f"  - {p_name} ({p_type}){required_flag}")
        else:
            lines.append("Inputs:")
            lines.append("  - (none)")

        lines.append("")

        if outputs:
            lines.append("Outputs:")
            for o in outputs:
                o_name = o.get("name", "?")
                o_type = o.get("type", "?")
                o_desc = o.get("description", "")
                if o_desc:
                    lines.append(f"  - {o_name} ({o_type}): {o_desc}")
                else:
                    lines.append(f"  - {o_name} ({o_type})")
        else:
            lines.append("Outputs:")
            lines.append(f"  - result ({return_type})")

        sections.append("\n".join(lines))

    tool_information: str = "\n---\n\n".join(sections)

    return {
        "name": name,
        "description": description,
        "tool_information": tool_information,
    }
