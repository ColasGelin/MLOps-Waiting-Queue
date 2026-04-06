# Supermarket Queue Manager

A fully self-hosted AI pipeline that monitors supermarket checkout queues in real time using computer vision and an LLM agent. No external APIs. Everything runs locally inside Docker.

A YOLOv8 model — fine-tuned on synthetic 3D supermarket footage — watches a top-down camera feed, counts customers per lane, and tracks wait times. When thresholds are crossed, a LangGraph agent powered by Ollama (llama3.1) decides whether to open a register, close one, redirect customers, or call a supervisor. Every suggestion goes through a human operator. The system never acts on its own.

<!-- TODO: drop a 30-second screen recording here showing the live UI + Grafana side by side -->
<!-- ![Demo](docs/demo.gif) -->

---

## Table of Contents

- [Architecture](#architecture)
- [Pipeline](#pipeline)
- [Fine-tuning the CV Model](#fine-tuning-the-cv-model)
- [The LLM Agent](#the-llm-agent)
- [Web UI](#web-ui)
- [Grafana & Observability](#grafana--observability)
- [Running with Docker](#running-with-docker)
- [Configuration Reference](#configuration-reference)
- [Known Limitations](#known-limitations)
- [What a Production Version Would Look Like](#what-a-production-version-would-look-like)

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

<!-- TODO: screenshot of the architecture in action — the web UI on the left, Grafana on the right -->

**Key design decision**: this is a **Human-in-the-Loop** system. The agent suggests actions (open lane 3, redirect customers from lane 1 to lane 2, call a supervisor). A human operator sees the suggestion and decides whether to act on it. The system has no actuators — it cannot open a register by itself.

**Why self-hosted**: customer behaviour data (video feeds, queue patterns, dwell times) is sensitive. Nothing leaves the local network. Ollama runs the LLM on-premise, and the entire stack fits inside a single `docker compose up`.

---

## Pipeline

The system runs as three concurrent threads inside a single Flask process:

### 1. Detection Thread — YOLOv8 + ByteTrack

Reads frames from the video source, runs YOLOv8 inference, and feeds detections into a ByteTrack tracker. For each tracked person, the system checks whether their centroid falls inside a queue zone polygon (using OpenCV's `pointPolygonTest`). When someone enters a zone, their dwell timer starts. When they leave, the wait time is recorded.

**Outputs per frame:**
- Person count per lane (queue1–queue4)
- Average wait time per lane
- Queue trend (growing / stable / declining)
- Total clients visible, employees visible, estimated store count

### 2. Agent Worker — LangGraph + Ollama

Sits idle until an alert fires. Two alert types:

| Alert | Trigger | Cooldown |
|-------|---------|----------|
| **Queue alert** | A lane hits 5+ people for 3+ seconds | 20s |
| **Close alert** | Dynamic lane (3/4) has ≤2 people and total load ≤8, held 10s | 40s |

When triggered, the agent receives a full metrics snapshot and responds with a structured decision:

```
SITUATION: Lane 1 has 6 customers and growing, lanes 3 and 4 are closed.
REASONING: 6 people in lane 1 with an upward trend — lane 3 is free and should absorb overflow.
ACTION: open_register(3)
URGENCY: high
```

### 3. Report Worker — Scheduled Summaries

Every 60 seconds, independently of alerts, the agent generates a 2–3 sentence operational summary of current conditions. These show up in the UI as trend reports.

---

## Fine-tuning the CV Model

### The dataset problem

Off-the-shelf YOLO models detect people well in street scenes, but struggle with top-down supermarket camera angles. People look different from above — no visible face, foreshortened body, shopping carts occluding limbs.

### Synthetic 3D training data

The training dataset was generated from synthetic 3D-rendered supermarket scenes filmed from a top-down perspective. This gave us:
- Full control over camera angle, lighting, and crowd density
- Automatic bounding box annotations (no manual labelling)
- The ability to generate edge cases (crowded lanes, employees restocking near queues)

The dataset lives in `datasettopview/` with 2 classes:

| Class ID | Label |
|----------|-------|
| 0 | Client |
| 1 | Employee |

36 annotated frames were split into train (~28) and validation (~8) sets, annotated via Label Studio in YOLO format.

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

Heavy augmentation compensates for the small dataset:
- Mosaic (0.8), mixup (0.1), flip, rotation (±10°), scale (0.3)
- HSV jitter to handle varying supermarket lighting

Output: `runs/detect/supermarket_finetune/weights/best.pt`

### Zone configuration

Queue zones are defined as 4-corner polygons. The `pick_zones.py` tool opens the first frame of your video, overlays a coordinate grid, and lets you draw the zone boundaries interactively:

```bash
python pick_zones.py --video topclip1.mp4
```

Zones are passed as 8 comma-separated values: `x1,y1,x2,y2,x3,y3,x4,y4`

---

## The LLM Agent

The agent is built with LangGraph as a two-node state graph:

```
[analyze] ──condition──▶ [execute_tool] ──▶ END
    │                                        ▲
    └──── action == "none" ──────────────────┘
```

### Available tools

| Tool | Description |
|------|-------------|
| `open_register(lane_id)` | Opens a dynamic checkout (lane 3 or 4) |
| `close_register(lane_id)` | Closes a dynamic checkout |
| `redirect_customers(from, to)` | Suggests moving people between lanes |
| `alert_supervisor(msg, urgency)` | Escalates to floor staff |
| `flag_anomaly(description)` | Flags unusual patterns for review |
| `generate_shift_report()` | Produces a management summary |

### Decision rules (priority order)

1. Lane has 5+ people AND a free lane exists → `open_register`
2. Lane has 5+ people AND no free lane AND another lane has 0–2 people → `redirect_customers`
3. Dynamic lane has 0–1 people AND no lane is overloaded → `close_register`
4. Lane has 6+ people and no safe move exists → `alert_supervisor`
5. Otherwise → `none`

### Store layout

```
 ┌─────────────────────────────────────────┐
 │              STORE FLOOR                │
 │                                         │
 │   Lane 1 [always open]  ████████░░░░   │
 │   Lane 2 [always open]  ██████░░░░░░   │
 │   Lane 3 [dynamic]      ░░░░░░░░░░░░   │
 │   Lane 4 [dynamic]      ░░░░░░░░░░░░   │
 │                                         │
 └─────────────────────────────────────────┘
     █ = customers in queue    ░ = empty
```

Lanes 1 and 2 are always staffed. Lanes 3 and 4 are opened or closed by the agent based on demand. Minimum 2 checkouts open, maximum 4.

### Guardrails

- The server enforces hard limits (can't close lane 1/2, can't exceed 4 open checkouts) regardless of what the LLM says
- If the LLM returns malformed output, the parser falls back gracefully with `action: none`
- Tool names are normalised — `call_supervisor`, `notify_supervisor`, and natural language like "calling the supervisor" all map to `alert_supervisor`

---

## Web UI

The browser UI is a single-page app served by Flask at port 8000.

<!-- TODO: full screenshot of the Monitor tab — video feed on the left, event cards on the right -->

### Monitor tab

- **Live video**: MJPEG stream from the detection thread, annotated with bounding boxes and zone overlays
- **Metrics bar**: queue counts per lane, open/closed checkout indicators, average wait times, in-store count, trends
- **Agent panel**: scrolling feed of agent decisions, colour-coded by urgency (green/orange/red), with situation + reasoning + action
- **Alert popup**: full-screen overlay when a high-urgency decision fires

### Dashboard tab

Links to Grafana for historical data and deeper analysis.

### Real-time updates

- Metrics poll every 2 seconds via `/metrics`
- Agent decisions stream instantly via Server-Sent Events (`/events`)

---

## Grafana & Observability

Grafana is pre-provisioned with a "Queue Monitor - Realtime Ops" dashboard that refreshes every 2 seconds.

<!-- TODO: screenshot of the full Grafana dashboard -->

### Panels

| Panel | What it shows |
|-------|---------------|
| **Queue Length by Checkout** | People per lane over time, with annotation markers for agent actions |
| **Average Wait Time** | Wait time per lane in seconds |
| **Visible People in Camera** | Clients vs. employees detected |
| **LLM Processing Time** | Duration of each agent call, with threshold lines |
| **Store Occupancy** | Total people in store vs. visible in camera, with annotation markers |

### Annotations

Agent decisions show as coloured vertical markers on the time-series graphs:

| Colour | Action |
|--------|--------|
| Green | Open register |
| Red | Close register |
| Orange | Redirect customers |
| White | No action |

### Log aggregation

Promtail ships Docker container logs to Loki. The detector thread emits one `TELEMETRY` log line per second with all current metrics, searchable from within Grafana.

---

## Running with Docker

### Prerequisites

- Docker & Docker Compose
- ~8 GB RAM (Ollama + llama3.1 needs ~4.5 GB)
- NVIDIA GPU + [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) (optional, for GPU acceleration)

### Quick start (CPU)

```bash
git clone <repo-url> && cd Supermlarket

# Place your video files in the project root, or use the included test clips

docker compose up
```

The app will:
1. Start Ollama and pull `llama3.1` on first run
2. Build the Flask app container
3. Start processing the default video file

Open `http://localhost:8000` for the live UI.

### With GPU

```bash
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up
```

This enables NVIDIA GPU for both YOLO inference and Ollama.

### With Grafana & full observability

```bash
docker compose --profile observability up
```

Then open `http://localhost:3000` for Grafana (default login: admin/admin).

### With everything + Nginx reverse proxy

```bash
docker compose --profile observability --profile gateway up
```

Access everything through Nginx:
- `/app` → Web UI
- `/dashboard/` → Grafana
- `/video` → Raw MJPEG stream

---

## Configuration Reference

All configuration is via environment variables in `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `VIDEO_FILE` | `/data/testclip2.mp4` | Comma-separated video paths (played sequentially, then loops) |
| `YOLO_MODEL` | Fine-tuned `best.pt` | Path to YOLO weights |
| `CONF_THRESH` | `0.35` | Detection confidence threshold |
| `YOLO_DEVICE` | `cpu` | `cpu` or `0` for GPU |
| `ZONE1` / `ZONE2` | Predefined polygons | Queue zone corners: `x1,y1,x2,y2,x3,y3,x4,y4` |
| `OLLAMA_MODEL` | `llama3.1` | LLM model for the agent |
| `OLLAMA_PULL_ON_START` | `true` | Auto-pull model on first boot |
| `NO_AGENT` | unset | Set to `1` to disable the LLM agent (CV-only mode) |
| `PORT` | `8000` | Flask server port |

### Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /` | Web UI |
| `GET /video` | MJPEG live stream |
| `GET /events` | SSE stream (agent decisions, alerts) |
| `GET /metrics` | JSON metrics snapshot |
| `GET /statistics` | Prometheus-format metrics |
| `POST /add_checkout` | Open a checkout |
| `POST /remove_checkout` | Close a checkout |
| `POST /reset` | Rewind video and reset state |

---

## Known Limitations

**Synthetic training data doesn't fully reflect real geometry.**
The 3D-rendered scenes approximate a top-down supermarket view, but real stores have variable lighting, camera distortion, and occlusion from shelving. The model will need retraining on real footage for production use.

**Ollama introduces latency and inconsistency.**
Running llama3.1 locally means each agent call takes 2–8 seconds depending on hardware. The model occasionally returns malformed responses or ignores the rules. The parser handles this gracefully, but it means some decisions are dropped.

**4 lanes don't justify an LLM.**
With only 4 checkout lanes and 5 clear rules, a deterministic state machine would be faster, cheaper, and more predictable. The LLM agent is more meaningful in data-rich environments — 20+ lanes, variable store layouts, multi-floor routing — where rules become unwieldy.

**No feedback loop.**
The system suggests actions but has no way to verify whether a human followed the suggestion, or whether the suggestion actually improved queue times. There's no reinforcement signal flowing back to the agent.

**Small training dataset.**
36 annotated frames is enough for a proof-of-concept with aggressive augmentation, but not enough for robust generalisation. A production model would need 500+ annotated frames across different stores, lighting conditions, and crowd densities.

---

## What a Production Version Would Look Like

| Area | Current | Production |
|------|---------|------------|
| **Camera** | Pre-recorded video files | RTSP streams from ceiling-mounted cameras |
| **CV model** | YOLOv8s fine-tuned on 36 frames | YOLOv8m/l trained on 1000+ real annotated frames per store |
| **Tracking** | ByteTrack with fixed zones | Multi-camera re-identification, automatic zone calibration |
| **LLM** | llama3.1 via Ollama (local) | Fine-tuned model or larger model with structured output guarantees |
| **Agent** | 5 hardcoded rules | Dynamic rule engine, learned from historical data |
| **Feedback** | None | Closed-loop: track whether suggestions were followed + outcome |
| **Alerting** | Browser popup | Integration with store intercom, mobile push, POS system |
| **Scale** | Single store, 4 lanes | Multi-store fleet, centralized dashboard, per-store agents |
| **Data** | Ephemeral (no persistence) | Time-series DB, decision audit log, shift analytics |

---

## Project Structure

```
.
├── server.py                  # Flask app, detector thread, agent worker, metrics
├── agent.py                   # LangGraph agent, tools, LLM prompt, parser
├── detect.py                  # YOLOv8 inference, ByteTrack, zone dwell tracking
├── train.py                   # Fine-tuning script for YOLOv8
├── pick_zones.py              # Interactive zone polygon selector
├── templates/index.html       # Single-page web UI
├── static/
│   ├── style.css              # Apple-inspired UI styling
│   └── app.js                 # Real-time polling, SSE handling, UI updates
├── docker-compose.yml         # Main orchestration (CPU)
├── docker-compose.gpu.yml     # GPU override for NVIDIA
├── Dockerfile                 # Python 3.11 + OpenCV + YOLO
├── entrypoint.sh              # Container init: wait for Ollama, pull model, start server
├── requirements.txt           # Python dependencies
├── datasettopview/            # Training dataset
│   ├── data.yaml              # YOLO dataset config
│   ├── train/                 # Training images + labels
│   └── val/                   # Validation images + labels
├── runs/                      # Training output (weights, metrics, curves)
├── grafana/
│   ├── dashboards/            # Pre-built Grafana dashboard JSON
│   └── provisioning/          # Datasource + dashboard provisioning
├── prometheus/
│   └── prometheus.yml         # Scrape config (5s interval)
├── promtail/
│   └── config.yml             # Log shipping to Loki
└── nginx/
    └── nginx.conf             # Reverse proxy + WebSocket support
```

---

## License

<!-- TODO: add license -->
