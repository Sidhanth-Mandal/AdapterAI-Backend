"""
apis/default_template.py
------------------------
Defines the built-in DEFAULT template that is returned to every user
on GET /loadtemplate/, regardless of what they have created.

This template lives purely in code — it is never stored in the database —
so there is no seed step, no SYSTEM user FK, and no migration required.

Schema mirrors the TemplateRecord Pydantic model:
    template_id  : str
    name         : str
    description  : str | None
    created_at   : None   (not applicable for a hard-coded template)
    updated_at   : None   (not applicable for a hard-coded template)

The agent uses behaviour_prompt and tool_information at runtime; those
fields are NOT part of TemplateRecord (the list endpoint only returns
display metadata), so they live here purely for reference / future use.
"""

from apis.schemas import TemplateRecord

# ---------------------------------------------------------------------------
# Runtime config (used by the MainAgent when the template is loaded)
# ---------------------------------------------------------------------------

DEFAULT_TEMPLATE_ID = "DEFAULT"

DEFAULT_BEHAVIOUR_PROMPT = """\
You are a helpful, knowledgeable, and friendly AI assistant.

You have no restrictions on topics — you are open to assist with anything the user asks, including:
  • Answering questions on any subject (science, history, arts, technology, etc.)
  • Writing, editing, and proofreading text
  • Explaining concepts clearly at any level of detail
  • Brainstorming ideas and creative tasks
  • Summarising documents or web pages
  • Solving maths problems and logical puzzles
  • Writing and debugging code in any programming language
  • General conversation and advice

Guidelines
----------
- Be concise but thorough. Prefer clear, structured answers (use bullet points or numbered lists where appropriate).
- If you are unsure about something, say so honestly rather than guessing.
- Use the available built-in tools (web search, calculator, document retrieval) whenever they would help you give a more accurate or up-to-date answer.
- Maintain a warm, professional tone at all times.
- Do not refuse reasonable requests; if a topic is sensitive, handle it thoughtfully.
""".strip()

# No custom tools — the agent's built-in tools are used automatically.
DEFAULT_TOOL_INFORMATION = ""

# ---------------------------------------------------------------------------
# TemplateRecord — what the /loadtemplate/ endpoint returns to the frontend
# ---------------------------------------------------------------------------

DEFAULT_TEMPLATE_RECORD = TemplateRecord(
    template_id=DEFAULT_TEMPLATE_ID,
    name="General Purpose Assistant",
    description=(
        "The built-in default template. "
        "An open-ended, general-purpose AI assistant with no domain restrictions. "
        "Uses built-in tools (web search, calculator, document retrieval) as needed. "
        "Available to every user automatically — cannot be deleted."
    ),
    created_at=None,
    updated_at=None,
)
