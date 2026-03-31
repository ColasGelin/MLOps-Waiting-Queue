# ── Queue Monitor — Python App ────────────────────────────────────────────────
# Runs YOLOv8 + Flask + LangGraph agent.
# Expects an Ollama service accessible via OLLAMA_HOST.

FROM python:3.11-slim

# System deps required by OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxrender1 \
        libxext6 \
        libgl1 \
        curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (separate layer for caching)
COPY requirements.txt .
RUN pip install uv
RUN uv pip install -r requirements.txt --system

# Copy application source
COPY agent.py detect.py server.py ./
COPY templates/ templates/
COPY static/    static/

# Entrypoint script handles Ollama readiness + model pull
COPY entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

EXPOSE 8000

ENTRYPOINT ["/entrypoint.sh"]
