import os
import subprocess
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from langchain_groq import ChatGroq

PROJECT_PATH = "D:/CODING/Projects/AdapterAI"

# -------- App --------
app = FastAPI()

# -------- CONFIG --------
API_KEY = "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"

client = ChatGroq(
    model='openai/gpt-oss-120b',
    api_key=API_KEY
)

# -------- Request Schema --------
class PromptRequest(BaseModel):
    prompt: str


# -------- Clean AI Code --------
def clean_code(code: str) -> str:
    if "```" in code:
        parts = code.split("```")
        code = parts[1] if len(parts) > 1 else code
        if code.startswith("python"):
            code = code[len("python"):]
    return code.strip()


# -------- Generate Code --------
def generate_code(prompt: str) -> str:
    response = client.invoke(
        [
            ("system", "Generate only safe Python code. No explanation."),
            ("human", prompt)
        ]
    )
    return response.content.strip()


# -------- Save Code --------
def save_code(code: str):
    with open("main_exec.py", "w") as f:
        f.write(code)


# -------- Run in Docker --------
def run_code():
    try:
        result = subprocess.run(
            [
                "docker", "run", "--rm",
                "--network", "none",
                "--memory", "100m",
                "--cpus", "0.5",
                "-v", f"{PROJECT_PATH}:/app",
                "python:3.10",
                "python", "/app/main_exec.py"
            ],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.stdout, result.stderr
    except subprocess.TimeoutExpired:
        return "", "Execution timed out"


# -------- API Endpoint --------
@app.post("/run")
async def run_code_endpoint(req: PromptRequest):
    if not req.prompt:
        raise HTTPException(status_code=400, detail="Prompt is required")

    # 1. Generate code
    raw_code = generate_code(req.prompt)

    # 2. Clean code
    code = clean_code(raw_code)

    # 3. Save
    save_code(code)

    # 4. Execute
    stdout, stderr = run_code()

    return {
        "code": code,
        "output": stdout,
        "error": stderr
    }