# Supermarket Queue Manager

A fully self-hosted AI pipeline that monitors supermarket checkout queues in real time using computer vision and an LLM agent. No external APIs. Everything runs locally inside Docker.

A YOLOv8 model — fine-tuned on footage from a video game simulating a top-down supermarket view — watches a camera feed, counts customers per lane, and tracks wait times. When thresholds are crossed, a LangGraph agent powered by Ollama (llama3.1) decides whether to open a register, close one, redirect customers, or call a supervisor. Every suggestion goes through a human operator. The system never acts on its own.

<!-- TODO: drop a 30-second screen recording here showing the live UI + Grafana side by side -->
<!-- ![Demo](docs/demo.gif) -->

---

## Quick Start

```bash
git clone <repo-url> && cd Supermlarket

# CPU only
docker compose up --build

# With GPU (requires nvidia-container-toolkit)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up --build

# With Grafana + Prometheus + Loki
docker compose --profile observability up --build
```

Open `http://localhost:8000` for the live UI. Grafana is at `http://localhost:3000` (admin/admin).

---

## Architecture

```
┌─────────────┐    MJPEG     ┌──────────────────────────────────────────────┐
│  Top-down   │───stream────▶│              Flask Server (8000)             │
│  Camera /   │              │                                              │
│  Video file │              │  ┌──────────┐  ┌──────────┐  ┌───────────┐  │
└─────────────┘              │  │ YOLOv8   │  │ Metrics  │  │  Agent    │  │
                             │  │ Detector │─▶│ Engine   │─▶│  Worker   │  │
                             │  │ Thread   │  │          │  │ (LangGraph│  │
                             │  └──────────┘  └────┬─────┘  │ + Ollama) │  │
                             │                     │        └─────┬─────┘  │
                             └─────────────────────┼──────────────┼────────┘
                                                   │              │
                               ┌───────────────────┤              │ SSE
                               │                   │              ▼
                          ┌────▼─────┐      ┌──────▼───┐   ┌──────────┐
                          │Prometheus│      │ /metrics │   │  Web UI  │
                          │  (5s)    │      │  (JSON)  │   │ (Browser)│
                          └────┬─────┘      └──────────┘   └──────────┘
                               │
                          ┌────▼─────┐
                          │ Grafana  │
                          │ (3000)   │
                          └──────────┘
```

**Human-in-the-Loop by design**: the agent suggests actions but cannot execute them. A human operator sees the suggestion and decides whether to act. The system has no actuators.

**Why self-hosted**: customer behaviour data (video feeds, queue patterns, dwell times) is sensitive. Nothing leaves the local network.

---

## Pipeline

Three concurrent threads run inside a single Flask process:

### 1. Detection — YOLOv8 + ByteTrack

Reads frames from the video source, runs YOLOv8 inference, and feeds detections into a ByteTrack tracker. For each tracked person, the system checks whether their centroid falls inside a queue zone polygon (OpenCV `pointPolygonTest`). When someone enters a zone, their dwell timer starts. When they leave, the wait time is recorded.

**Outputs per frame:** person count per lane, average wait time, queue trend (growing/stable/declining), total clients and employees visible.

### 2. Agent — LangGraph + Ollama

Sits idle until an alert fires:

| Alert | Trigger | Cooldown |
|-------|---------|----------|
| **Queue alert** | A lane hits 5+ people for 3+ seconds | 20s |
| **Close alert** | Dynamic lane (3/4) has ≤2 people and total load ≤8, held 10s | 40s |

The agent receives a full metrics snapshot and responds with a structured decision:

```
SITUATION: Lane 1 has 6 customers and growing, lanes 3 and 4 are closed.
REASONING: 6 people in lane 1 with an upward trend — lane 3 is free and should absorb overflow.
ACTION: open_register(3)
URGENCY: high
```

### 3. Report — Scheduled Summaries

Every 60 seconds, independently of alerts, the agent generates a short operational summary of current conditions.

---

## Fine-tuning the CV Model

### Why fine-tune?

Off-the-shelf YOLO detects people well in street scenes, but struggles with top-down camera angles. People look different from above — no visible face, foreshortened body, shopping carts occluding limbs.

### Training data from a video game

The training dataset comes from a supermarket simulation video game, captured from a top-down perspective. Frames were extracted and annotated manually using Label Studio. Two classes: **Client** (0) and **Employee** (1). 36 annotated frames split into train (~28) and validation (~8).

<!-- TODO: grid of 4–6 sample frames from the training set showing the top-down view with bounding boxes -->

### Training configuration

```python
# train.py
BASE_MODEL  = "yolov8s.pt"   # YOLOv8 small as starting point
EPOCHS      = 100
BATCH       = 8
IMG_SIZE    = 640
FREEZE      = 10              # Freeze backbone — preserve pretrained features
```

Heavy augmentation compensates for the small dataset: mosaic (0.8), mixup (0.1), flip, rotation (±10°), scale (0.3), HSV jitter.

Output: `runs/detect/supermarket_finetune/weights/best.pt`

### Zone configuration

Queue zones are 4-corner polygons. Use `pick_zones.py` to draw them interactively on the first frame of your video:

```bash
python pick_zones.py --video topclip1.mp4
```

---

## The LLM Agent

Built with LangGraph as a two-node state graph:

```
[analyze] ──condition──▶ [execute_tool] ──▶ END
    │                                        ▲
    └──── action == "none" ──────────────────┘
```

### Tools

| Tool | Description |
|------|-------------|
| `open_register(lane_id)` | Opens a dynamic checkout (lane 3 or 4) |
| `close_register(lane_id)` | Closes a dynamic checkout |
| `redirect_customers(from, to)` | Suggests moving people between lanes |
| `alert_supervisor(msg, urgency)` | Escalates to floor staff |
| `flag_anomaly(description)` | Flags unusual patterns for review |
| `generate_shift_report()` | Produces a management summary |

### Decision rules (from the system prompt, in priority order)

1. Lane has 5+ people AND a free lane exists → `open_register`
2. Lane has 5+ people AND no free lane AND another lane has 0–2 people → `redirect_customers`
3. Dynamic lane (3 or 4) has 0–1 people AND no lane is overloaded → `close_register`
4. Lane has 6+ people and no safe open/redirect move exists → `alert_supervisor`
5. Otherwise → `none`

### Store layout

Lanes 1 and 2 are always staffed. Lanes 3 and 4 are opened or closed by the agent based on demand. Minimum 2 checkouts open, maximum 4.

### Guardrails

- The server enforces hard limits (can't close lane 1/2, can't exceed 4 open) regardless of what the LLM says
- Malformed LLM output falls back to `action: none`
- Tool names are normalised — `call_supervisor`, `notify_supervisor`, and natural language like "calling the supervisor" all map to `alert_supervisor`

---

## Web UI

<!-- TODO: full screenshot of the Monitor tab — video feed on the left, event cards on the right -->

- **Live video**: MJPEG stream annotated with bounding boxes and zone overlays
- **Metrics bar**: queue counts per lane, open/closed indicators, average wait times, in-store count, trends
- **Agent panel**: scrolling feed of decisions, colour-coded by urgency (green/orange/red)
- **Alert popup**: full-screen overlay on high-urgency decisions
- **Real-time**: metrics poll every 2s, agent decisions stream via SSE

---

## Grafana & Observability

Pre-provisioned dashboard ("Queue Monitor - Realtime Ops"), refreshes every 2 seconds.

<!-- TODO: screenshot of the full Grafana dashboard -->

### Panels

| Panel | What it shows |
|-------|---------------|
| **Queue Length by Checkout** | People per lane over time, with annotation markers for agent actions |
| **Average Wait Time** | Wait time per lane in seconds |
| **Visible People in Camera** | Clients vs. employees detected |
| **LLM Processing Time** | Duration of each agent call |
| **Store Occupancy** | Total people in store vs. visible in camera |

### Annotation colours

Agent decisions show as coloured vertical markers on the time-series graphs:

| Colour | Action |
|--------|--------|
| Green | Open register |
| Red | Close register |
| Orange | Redirect customers |
| White | No action |

### Logs

Promtail ships Docker container logs to Loki. The detector emits one `TELEMETRY` log line per second with all current metrics, searchable from Grafana.

---

## Configuration

All via environment variables in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO_FILE` | `/data/testclip2.mp4` | Comma-separated video paths (played sequentially, then loops) |
| `YOLO_MODEL` | Fine-tuned `best.pt` | Path to YOLO weights |
| `CONF_THRESH` | `0.35` | Detection confidence threshold |
| `YOLO_DEVICE` | `cpu` | `cpu` or `0` for GPU |
| `ZONE1` / `ZONE2` | Predefined polygons | Queue zone corners: `x1,y1,x2,y2,x3,y3,x4,y4` |
| `OLLAMA_MODEL` | `llama3.1` | LLM model for the agent |
| `NO_AGENT` | unset | Set to `1` to disable the LLM agent (CV-only mode) |

---

## Known Limitations

**Video game training data doesn't fully reflect real geometry.** The simulated scenes approximate a top-down supermarket view, but real stores have different lighting, camera distortion, and occlusion from shelving. The model needs retraining on real footage for production use.

**Small training dataset.** 36 annotated frames is enough for a proof-of-concept with aggressive augmentation, but not for robust generalisation. A production model would need 500+ frames across different stores and conditions.

**Ollama introduces latency and inconsistency.** Running llama3.1 locally means each agent call takes 2–8 seconds. The model occasionally returns malformed responses or ignores the rules. The parser handles this gracefully, but some decisions get dropped.

**4 lanes don't justify an LLM.** With only 4 checkout lanes and 5 clear rules, a deterministic state machine would be faster, cheaper, and more predictable. The LLM agent makes more sense in data-rich environments — 20+ lanes, variable layouts, multi-floor routing — where rules become unwieldy.

**No feedback loop.** The system suggests actions but can't verify whether a human followed the suggestion, or whether it actually improved queue times. There's no reinforcement signal flowing back to the agent.

---

## What a Production Version Would Look Like

| Area | Current | Production |
|------|---------|------------|
| **Camera** | Pre-recorded video files | RTSP streams from ceiling-mounted cameras |
| **CV model** | YOLOv8s on 36 frames | YOLOv8m/l trained on 1000+ real frames per store |
| **Tracking** | ByteTrack with fixed zones | Multi-camera re-ID, automatic zone calibration |
| **LLM** | llama3.1 via Ollama | Fine-tuned model with structured output guarantees |
| **Feedback** | None | Closed-loop: track if suggestions were followed + outcome |
| **Alerting** | Browser popup | Store intercom, mobile push, POS integration |
| **Scale** | Single store, 4 lanes | Multi-store fleet with per-store agents |
| **Data** | Ephemeral | Time-series DB, decision audit log, shift analytics |
