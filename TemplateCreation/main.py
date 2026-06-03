"""
main.py — REMOVED
-----------------
The CLI entry point has been removed as part of the API-driven refactor.

Use the service layer instead:

    from TemplateCreation.service import chat_template, create_template

    # One conversational turn
    response = chat_template(
        template_id="tmpl-abc123",
        user_id="user-001",
        user_prompt="I need a coding assistant that helps with Python.",
    )

    # Phase 2 is triggered automatically when requirements are satisfied.
    # You can also call create_template() directly if needed:
    create_template(
        user_id="user-001",
        template_id="tmpl-abc123",
        template_conv_history=[...],  # list of {role, content, ...} dicts
    )

See service.py for full API documentation.
"""

raise RuntimeError(
    "main.py has been removed.  "
    "Import from service.py instead: chat_template() and create_template()."
)
