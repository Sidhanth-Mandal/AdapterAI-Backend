# Docker_exec/Dockerfile
# Build:  docker build -t tool-executor .
# Run:    docker run -p 8000:8000 tool-executor

FROM python:3.11-slim

WORKDIR /app

# Install base server dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api.py          .
COPY tool_compiler.py .
COPY runner.py       .

# Create the directory where compiled tools will be written at runtime
RUN mkdir -p /tmp/compiled_tools

EXPOSE 8000

CMD ["python", "api.py"]
