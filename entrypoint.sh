#!/usr/bin/env bash
set -e

OLLAMA_HOST="${OLLAMA_HOST:-http://ollama:11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1}"
OLLAMA_PULL_ON_START="${OLLAMA_PULL_ON_START:-false}"
OLLAMA_REQUIRE_MODEL="${OLLAMA_REQUIRE_MODEL:-false}"

# ── Wait for Ollama ──────────────────────────────────────────────────────────
echo "[INIT] Waiting for Ollama at $OLLAMA_HOST ..."
for i in $(seq 1 40); do
    if curl -sf "$OLLAMA_HOST/api/tags" > /dev/null 2>&1; then
        echo "[INIT] Ollama is ready."
        break
    fi
    if [ "$i" -eq 40 ]; then
        if [ "$OLLAMA_REQUIRE_MODEL" = "true" ]; then
            echo "[ERROR] Ollama not reachable after 40 attempts and OLLAMA_REQUIRE_MODEL=true."
            exit 1
        else
            echo "[WARN] Ollama not reachable after 40 attempts — starting without LLM."
            export NO_AGENT=true
            break
        fi
    fi
    echo "[INIT] Attempt $i/40 — retrying in 3 s..."
    sleep 3
done

# ── Pull model if needed ─────────────────────────────────────────────────────
if [ "${NO_AGENT:-false}" != "true" ]; then
    if [ "$OLLAMA_PULL_ON_START" = "true" ]; then
        if [ "$OLLAMA_REQUIRE_MODEL" = "true" ]; then
            echo "[INIT] Blocking startup until model '$OLLAMA_MODEL' is available..."
            curl -sf "$OLLAMA_HOST/api/pull" \
                -H "Content-Type: application/json" \
                -d "{\"name\":\"$OLLAMA_MODEL\",\"stream\":false}" \
                | python3 -c "import sys,json; d=json.load(sys.stdin); print('[INIT] Model status:', d.get('status','done'))" \
                || { echo "[ERROR] Model pull failed while OLLAMA_REQUIRE_MODEL=true."; exit 1; }
        else
            echo "[INIT] Pulling model '$OLLAMA_MODEL' in background..."
            (
                curl -sf "$OLLAMA_HOST/api/pull" \
                    -H "Content-Type: application/json" \
                    -d "{\"name\":\"$OLLAMA_MODEL\",\"stream\":false}" \
                    | python3 -c "import sys,json; d=json.load(sys.stdin); print('[INIT] Background model status:', d.get('status','done'))" \
                    || echo "[WARN] Background model pull failed or still in progress."
            ) &
        fi
    else
        if [ "$OLLAMA_REQUIRE_MODEL" = "true" ]; then
            echo "[ERROR] OLLAMA_REQUIRE_MODEL=true requires OLLAMA_PULL_ON_START=true."
            exit 1
        fi
        echo "[INIT] Skipping model pull on startup (OLLAMA_PULL_ON_START=false)."
    fi
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
ARGS+=(--video)
for _v in "${_VIDEOS[@]}"; do
    ARGS+=("$_v")
done

[ -n "${ZONE1:-}" ] && ARGS+=(--zone1 "$ZONE1")
[ -n "${ZONE2:-}" ] && ARGS+=(--zone2 "$ZONE2")
[ "${NO_AGENT:-false}" = "true" ] && ARGS+=(--no-agent)

echo "[INIT] Starting: ${ARGS[*]}"
exec "${ARGS[@]}"
