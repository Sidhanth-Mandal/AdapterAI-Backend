from schemas import ToolSchema      # direct script execution fallback
from SystemPrompt import system_prompt
from langchain_groq import ChatGroq
from dotenv import load_dotenv
from langchain_core.messages import SystemMessage, HumanMessage
from validator import validate_tool
import json
import sys


user_prompt = """I want you to become cicket expert and provide me with real stats"""

load_dotenv()

model = ChatGroq(model='llama-3.3-70b-versatile')

schema_str = json.dumps(ToolSchema.model_json_schema(), indent=2)

json_instruction = f"""
IMPORTANT: Respond with ONLY a valid JSON object that matches this exact schema. No markdown, no explanation, no code fences.

Schema:
{schema_str}
"""

response = model.invoke([
    SystemMessage(content=system_prompt + json_instruction),
    HumanMessage(content=user_prompt)
])

raw = response.content.strip()
if raw.startswith("```"):
    raw = raw.split("```", 2)[1]          # drop opening fence
    if raw.startswith("json"):
        raw = raw[4:]                      # drop "json" language tag
    raw = raw.rsplit("```", 1)[0].strip() # drop closing fence

try:
    import json_repair
    data = json_repair.loads(raw)
except ImportError:
    data = json.loads(raw)

tool = ToolSchema.model_validate(data)

OUTPUT_PATH = "codeexecuter/generated_tool.json"
with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    f.write(tool.model_dump_json(indent=2))

print("[OK] Tool generated successfully!")
print(tool.model_dump_json(indent=2))

# ── Auto-validate the generated tool ──────────────────────────────────────
print("\n[Validator] Running validation pipeline …")
report = validate_tool(OUTPUT_PATH)
print(report.summary())

if not report.passed:
    print("[Validator] ✗ Validation FAILED — tool was saved but should NOT be registered.", file=sys.stderr)
    sys.exit(1)

if not report.safe:
    print("[Validator] ✗ Tool flagged as UNSAFE — dangerous patterns detected.", file=sys.stderr)
    sys.exit(1)

print("[Validator] ✓ Tool is valid and safe — ready for registration.")

