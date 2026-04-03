#!/usr/bin/env bash
set -e

OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1}"

# ── Wait for Ollama ──────────────────────────────────────────────────────────
echo "[INIT] Waiting for Ollama at $OLLAMA_HOST ..."
for i in $(seq 1 40); do
    if curl -sf "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
        echo "[INIT] Ollama is ready."
        break
    fi
    if [ "$i" -eq 40 ]; then
        echo "[WARN] Ollama not reachable after 40 attempts — starting without LLM."
        export NO_AGENT=true
        break
    fi
    echo "[INIT] Attempt $i/40 — retrying in 3 s..."
    sleep 3
done

# ── Pull model if needed ─────────────────────────────────────────────────────
if [ "${NO_AGENT:-false}" != "true" ]; then
    echo "[INIT] Ensuring model '$OLLAMA_MODEL' is available (first run may take several minutes)..."
    curl -sf "$OLLAMA_HOST/api/pull" \
        -H "Content-Type: application/json" \
        -d "{\"name\":\"$OLLAMA_MODEL\",\"stream\":false}" \
        | python3 -c "import sys,json; d=json.load(sys.stdin); print('[INIT] Model status:', d.get('status','done'))" \
        || echo "[INIT] Pull request sent (model may already be cached)."
fi

# ── Build command from env vars ──────────────────────────────────────────────
VIDEO="${VIDEO_FILE:-/data/video.mp4}"
MODEL="${YOLO_MODEL:-yolov8s.pt}"
CONF="${CONF_THRESH:-0.35}"
DEVICE="${YOLO_DEVICE:-cpu}"
PORT="${PORT:-8000}"

ARGS=(
    python server.py
    --model  "$MODEL"
    --conf   "$CONF"
    --device "$DEVICE"
    --host   "0.0.0.0"
    --port   "$PORT"
)

# Support comma-separated VIDEO_FILE for sequential clip playback
IFS=',' read -ra _VIDEOS <<< "$VIDEO"
for _v in "${_VIDEOS[@]}"; do
    ARGS+=(--video "$_v")
done

[ -n "${ZONE1:-}" ] && ARGS+=(--zone1 "$ZONE1")
[ -n "${ZONE2:-}" ] && ARGS+=(--zone2 "$ZONE2")
[ "${NO_AGENT:-false}" = "true" ] && ARGS+=(--no-agent)

echo "[INIT] Starting: ${ARGS[*]}"
exec "${ARGS[@]}"
