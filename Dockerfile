# ============================================================
# AdapterAI — Backend Dockerfile
# Builds the FastAPI backend that serves the whole project
# ============================================================

FROM python:3.11-slim

# System dependencies (for psycopg2-binary, bcrypt, etc.)
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python dependencies first (cached layer) ─────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy all project source code ─────────────────────────────
COPY apis/           ./apis/
COPY MainAgent/      ./MainAgent/
COPY SubAgent/       ./SubAgent/
COPY ToolGeneration/ ./ToolGeneration/
COPY TemplateCreation/ ./TemplateCreation/
COPY builtintools/   ./builtintools/
COPY vector_store/   ./vector_store/
COPY utils/          ./utils/

# ── Expose backend port ───────────────────────────────────────
EXPOSE 8002

# ── Run the FastAPI app ───────────────────────────────────────
CMD ["uvicorn", "apis.main:app", "--host", "0.0.0.0", "--port", "8002"]
