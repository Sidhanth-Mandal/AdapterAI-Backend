# seed.py  — run from the ToolGeneration/ directory
import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from db import get_connection

# ── values ──────────────────────────────────────────────────────────────────
USER = {
    "user_id":       "usr00001",       # VARCHAR — change to 1 if you use INT
    "username":      "dev_user",
    "email":         "dev@adapterai.com",
    "password_hash": "hashed_password_here",   # replace with a real bcrypt hash
}

TEMPLATE = {
    "template_id":            "tem00001",
    "name":                   "Cricket Stats Tool",
    "description":            "Generates a tool that fetches live and historical cricket statistics.",
    "system_prompt":          None,
    "tool_generation_prompt": (
        "I want you to become a cricket expert and provide me with real stats. "
        "Build functions to get player batting/bowling averages, "
        "team head-to-head records, and live match scores using free APIs."
    ),
    "tool_information":       None,
    "created_by":             "usr00001",   # must match USER["user_id"] type
}

# ── insert ───────────────────────────────────────────────────────────────────
INSERT_USER = """
    INSERT INTO users (user_id, username, email, password_hash, created_at, updated_at)
    VALUES (%(user_id)s, %(username)s, %(email)s, %(password_hash)s, NOW(), NOW())
    ON CONFLICT (user_id) DO NOTHING;
"""

INSERT_TEMPLATE = """
    INSERT INTO templates (
        template_id, name, description,
        system_prompt, tool_generation_prompt, tool_information,
        created_by, created_at, updated_at
    )
    VALUES (
        %(template_id)s, %(name)s, %(description)s,
        %(system_prompt)s, %(tool_generation_prompt)s, %(tool_information)s,
        %(created_by)s, NOW(), NOW()
    )
    ON CONFLICT (template_id) DO NOTHING;
"""

def seed():
    with get_connection() as conn:
        with conn.cursor() as cur:
            cur.execute(INSERT_USER, USER)
            print(f"[seed] User '{USER['user_id']}' inserted (or already exists).")

            cur.execute(INSERT_TEMPLATE, TEMPLATE)
            print(f"[seed] Template '{TEMPLATE['template_id']}' inserted (or already exists).")

        conn.commit()
    print("[seed] Done.")


def fetch_and_print():
    """Fetch tem00001 from templates and to00001 from tools, then print their contents."""
    import json
    import psycopg2.extras

    with get_connection() as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:

            # ── Template ──────────────────────────────────────────────────────
            cur.execute("SELECT * FROM templates WHERE template_id = %s", ("tem00001",))
            template = cur.fetchone()
            print("\n" + "=" * 60)
            print("TEMPLATE: tem00001")
            print("=" * 60)
            if template:
                for key, value in template.items():
                    print(f"  {key:<28}: {value}")
            else:
                print("  ⚠  Not found.")

            # ── Tool ──────────────────────────────────────────────────────────
            cur.execute("SELECT * FROM tools WHERE tool_id = %s", ("to00001",))
            tool = cur.fetchone()
            print("\n" + "=" * 60)
            print("TOOL: to00001")
            print("=" * 60)
            if tool:
                for key, value in tool.items():
                    # Pretty-print JSONB field
                    if key == "tool_json" and value is not None:
                        print(f"  {key:<28}:")
                        print(json.dumps(value, indent=4))
                    else:
                        print(f"  {key:<28}: {value}")
            else:
                print("  ⚠  Not found.")

        print("=" * 60 + "\n")


if __name__ == "__main__":
    #seed()
    fetch_and_print()
