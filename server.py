"""
Flask Server — Queue Monitor Web UI
Runs the YOLOv8 detector in a background thread, streams MJPEG video,
pushes agent decision events via SSE, and serves the frontend.

Usage:
    python server.py --video testclip.mp4 --zone1 x1,y1,...,x4,y4 --zone2 x1,y1,...,x4,y4
"""

import argparse
import json
import queue
import random
import threading
import time
from datetime import datetime
from types import SimpleNamespace

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template, request, stream_with_context
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker

from agent import run_agent, run_report
from detect import (
    BOX_COLOR,
    CLIENT_CLASS,
    EMPLOYEE_CLASS,
    EMPLOYEE_COLOR,
    TRACKER_ARGS,
    ZONE_COLORS,
    StoreCounter,
    TrackRegistry,
    centroid_in_zone,
    compute_trend,
    draw_box,
    draw_hud,
    draw_zone,
    parse_zone,
)

# ── Config ───────────────────────────────────────────────────────────────────

CONF_THRESH          = 0.35
ZONE_MIN_DWELL_SEC   = 1.0
ALERT_THRESHOLD      = 5     # people in a lane to trigger alert
ALERT_DURATION       = 3.0   # seconds the threshold must be held before alert fires
ALERT_COOLDOWN       = 20.0  # seconds before a new alert can fire globally
POST_ACTION_COOLDOWN = 20.0  # seconds to suppress alerts after a tool action

# ── Shared state ─────────────────────────────────────────────────────────────

# Thread-safe queue for SSE events (agent decisions)
sse_queue = queue.Queue(maxsize=200)

# Urgent trigger queue: detector pushes a metrics snapshot here when a
# threshold alert fires so the agent worker calls the LLM immediately
urgent_queue = queue.Queue(maxsize=5)

# True while the agent is actively processing a request — blocks new alerts
agent_busy = False
agent_busy_lock = threading.Lock()

# Rolling log of events for the minute report (entries are plain strings)
minute_log = []
minute_log_lock = threading.Lock()

def log_event(msg: str):
    """Append a timestamped entry to the rolling minute log."""
    with minute_log_lock:
        minute_log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")

# List of SSE subscriber queues (one per connected client)
sse_clients = []
sse_clients_lock = threading.Lock()

# Latest metrics (written by detector, read by /metrics and agent)
latest_metrics = {
    "queue1": 0,
    "queue2": 0,
    "queue1_avg_wait": None,
    "queue2_avg_wait": None,
    "queue1_trend": "stable",
    "queue2_trend": "stable",
    "queue3": 0,         # Extra checkout 3 (simulated)
    "queue4": 0,         # Extra checkout 4 (simulated)
    "employees": 0,
    "clients_visible": 0,
    "store_count": 0,
    "checkouts_open": 2,
    "last_action": None,       # e.g. "open_register(1)"
    "last_action_time": 0.0,   # unix timestamp of last tool action
    "status": "initializing",
    "fps": 0.0,
    "frame": 0,
    "total_frames": 0,
}
metrics_lock = threading.Lock()

# Simulated activity state for dynamic checkouts (3 and 4), backend-owned.
# Each checkout has its own next_update timestamp for irregular timing.
extra_checkout_state = {
    3: {"count": 0, "trend": 1, "next_update": 0.0},
    4: {"count": 0, "trend": 1, "next_update": 0.0},
}


def _update_extra_checkouts_locked(now_ts: float):
    """Update queue3/queue4 on independent irregular intervals (1.5–9s each)."""
    open_count = int(latest_metrics.get("checkouts_open", 2))
    for checkout_num in (3, 4):
        state = extra_checkout_state[checkout_num]
        is_open = open_count >= checkout_num
        key = f"queue{checkout_num}"

        if not is_open:
            state["count"] = 0
            state["trend"] = 1
            state["next_update"] = 0.0
            latest_metrics[key] = 0
            continue

        if now_ts < state["next_update"]:
            continue

        # Each checkout picks its own next update time independently
        state["next_update"] = now_ts + random.uniform(1.5, 9.0)

        if random.random() < 0.6:
            state["trend"] = 1 if random.random() < 0.5 else -1

        state["count"] += state["trend"]
        state["count"] = max(0, min(5, state["count"]))

        if state["count"] == 0:
            state["trend"] = 1
        elif state["count"] == 5:
            state["trend"] = -1

        latest_metrics[key] = int(state["count"])


def broadcast_event(event_data: dict):
    """Push an event to all connected SSE clients."""
    msg = f"data: {json.dumps(event_data)}\n\n"
    with sse_clients_lock:
        dead = []
        for q in sse_clients:
            try:
                q.put_nowait(msg)
            except queue.Full:
                dead.append(q)
        for q in dead:
            sse_clients.remove(q)


# ── Detector worker ─────────────────────────────────────────────────────────

class DetectorState:
    def __init__(self, video_path, model_name, conf, zone1, zone2, device, loop_video):
        self.video_path = video_path
        self.model_name = model_name
        self.conf = conf
        self.zone1 = zone1
        self.zone2 = zone2
        self.device = device
        self.loop_video = loop_video
        self.lock = threading.Lock()
        self.latest_jpeg = None
        self.stop_event = threading.Event()


def detector_worker(state: DetectorState):
    """Run YOLO + ByteTrack on the video, produce JPEG frames and metrics."""
    try:
        model = YOLO(state.model_name)
        tracker = BYTETracker(TRACKER_ARGS, frame_rate=30)
        cap = cv2.VideoCapture(state.video_path)
        if not cap.isOpened():
            with metrics_lock:
                latest_metrics["status"] = "error"
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        registry = TrackRegistry(video_fps)
        store_counter = StoreCounter()
        frame_idx = 0
        proc_t0 = time.time()
        zone_dwell = [{}, {}]
        zone_entry_frame = [{}, {}]
        zone_completed_waits = [[], []]

        # History for trend computation
        q1_history = []
        q2_history = []

        # Alert state: high_since per lane, single global cooldown + counter
        lane_alert = [{"high_since": None}, {"high_since": None}]
        last_global_alert = 0.0
        alert_counter = 0

        with metrics_lock:
            latest_metrics["total_frames"] = total_frames
            latest_metrics["status"] = "running"

        while not state.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if state.loop_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    tracker = BYTETracker(TRACKER_ARGS, frame_rate=30)
                    registry = TrackRegistry(video_fps)
                    zone_dwell = [{}, {}]
                    zone_entry_frame = [{}, {}]
                    zone_completed_waits = [[], []]
                    continue
                with metrics_lock:
                    latest_metrics["status"] = "ended"
                break

            frame_idx += 1
            frame_start = time.time()

            with metrics_lock:
                _update_extra_checkouts_locked(frame_start)

            # Detection
            results = model(
                frame,
                classes=[CLIENT_CLASS, EMPLOYEE_CLASS],
                conf=state.conf,
                device=state.device,
                verbose=False,
            )[0]

            all_boxes = results.boxes.cpu() if results.boxes is not None else None
            if all_boxes is not None and len(all_boxes) > 0:
                client_boxes = all_boxes[all_boxes.cls == CLIENT_CLASS]
                employee_boxes = all_boxes[all_boxes.cls == EMPLOYEE_CLASS]
            else:
                client_boxes = None
                employee_boxes = None

            raw_tracks = (
                tracker.update(client_boxes, frame)
                if client_boxes is not None and len(client_boxes) > 0
                else []
            )
            visible = registry.update(raw_tracks, frame_idx)
            employee_count = 0 if employee_boxes is None else len(employee_boxes)

            # Zone dwell tracking
            min_frames = ZONE_MIN_DWELL_SEC * video_fps
            visible_cids = {cid for _, cid in visible}
            for i, zone in enumerate([state.zone1, state.zone2]):
                if zone is None:
                    continue
                for box, cid in visible:
                    if centroid_in_zone(box, zone):
                        prev = zone_dwell[i].get(cid, 0)
                        zone_dwell[i][cid] = prev + 1
                        if prev == 0:
                            zone_entry_frame[i][cid] = frame_idx
                    else:
                        if cid in zone_dwell[i]:
                            if zone_dwell[i][cid] >= min_frames:
                                wait = (frame_idx - zone_entry_frame[i].get(cid, frame_idx)) / video_fps
                                zone_completed_waits[i].append(wait)
                            zone_dwell[i].pop(cid, None)
                            zone_entry_frame[i].pop(cid, None)
                for cid in list(zone_dwell[i]):
                    if cid not in visible_cids:
                        if zone_dwell[i][cid] >= min_frames:
                            wait = (frame_idx - zone_entry_frame[i].get(cid, frame_idx)) / video_fps
                            zone_completed_waits[i].append(wait)
                        zone_dwell[i].pop(cid, None)
                        zone_entry_frame[i].pop(cid, None)

            q1_count = sum(1 for _, f in zone_dwell[0].items() if f >= min_frames)
            q2_count = sum(1 for _, f in zone_dwell[1].items() if f >= min_frames)

            # ── Threshold alerts ─────────────────────────────────────────────
            now_t = time.time()
            for lane_idx, count in enumerate([q1_count, q2_count]):
                lane_state = lane_alert[lane_idx]
                lane_num = lane_idx + 1
                if count >= ALERT_THRESHOLD:
                    if lane_state["high_since"] is None:
                        lane_state["high_since"] = now_t
                    elif (now_t - lane_state["high_since"] >= ALERT_DURATION
                          and now_t - last_global_alert >= ALERT_COOLDOWN):
                        # Skip if LLM is busy or a tool action was recently taken
                        with agent_busy_lock:
                            if agent_busy:
                                continue
                        with metrics_lock:
                            last_action_time = latest_metrics["last_action_time"]
                        if now_t - last_action_time < POST_ACTION_COOLDOWN:
                            continue
                        alert_counter += 1
                        aid = f"alert-{alert_counter}"
                        broadcast_event({
                            "type": "queue_alert",
                            "alert_id": aid,
                            "lane": lane_num,
                            "count": count,
                            "timestamp": time.strftime("%H:%M:%S"),
                            "message": f"Lane {lane_num} has {count} people waiting.",
                        })
                        log_event(f"Alert fired — Lane {lane_num}: {count} people")
                        with metrics_lock:
                            snapshot = dict(latest_metrics)
                        snapshot["_alert_id"] = aid
                        try:
                            urgent_queue.put_nowait(snapshot)
                        except queue.Full:
                            pass
                        last_global_alert = now_t
                else:
                    lane_state["high_since"] = None

            # Avg waits
            q1_avg, q2_avg = None, None
            if state.zone1 is not None:
                all_w = zone_completed_waits[0] + [
                    (frame_idx - zone_entry_frame[0].get(cid, frame_idx)) / video_fps
                    for cid, f in zone_dwell[0].items() if f >= min_frames
                ]
                q1_avg = (sum(all_w) / len(all_w)) if all_w else None
            if state.zone2 is not None:
                all_w = zone_completed_waits[1] + [
                    (frame_idx - zone_entry_frame[1].get(cid, frame_idx)) / video_fps
                    for cid, f in zone_dwell[1].items() if f >= min_frames
                ]
                q2_avg = (sum(all_w) / len(all_w)) if all_w else None

            # Store count
            store_count = store_counter.update(len(visible))

            # Trend history (record every ~1 second of video)
            if frame_idx % max(1, int(video_fps)) == 0:
                q1_history.append(q1_count)
                q2_history.append(q2_count)

            q1_trend = compute_trend(q1_history)
            q2_trend = compute_trend(q2_history)

            # Determine alert status
            is_alert = q1_count >= 4 or q2_count >= 4 or q1_trend == "growing" or q2_trend == "growing"

            # Draw annotations
            for (x1, y1, x2, y2), cid in visible:
                if zone_dwell[0].get(cid, 0) >= min_frames:
                    color = ZONE_COLORS[0]
                    wait_s = (frame_idx - zone_entry_frame[0].get(cid, frame_idx)) / video_fps
                    label = f"Client {wait_s:.0f}s"
                elif zone_dwell[1].get(cid, 0) >= min_frames:
                    color = ZONE_COLORS[1]
                    wait_s = (frame_idx - zone_entry_frame[1].get(cid, frame_idx)) / video_fps
                    label = f"Client {wait_s:.0f}s"
                else:
                    color = BOX_COLOR
                    label = "Client"
                draw_box(frame, x1, y1, x2, y2, color=color, label=label)

            if employee_boxes is not None:
                for box in employee_boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    draw_box(frame, x1, y1, x2, y2, color=EMPLOYEE_COLOR, label="Employee")

            queue_counts = []
            avg_waits = []
            if state.zone1 is not None:
                queue_counts.append(("Queue 1", q1_count))
                avg_waits.append(q1_avg)
            if state.zone2 is not None:
                queue_counts.append(("Queue 2", q2_count))
                avg_waits.append(q2_avg)
            if not queue_counts:
                queue_counts.append(("Clients", len(visible)))
            draw_hud(frame, employee_count, video_fps, queue_counts, avg_waits)

            # Encode JPEG
            ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                continue

            now = time.time()
            processing_fps = frame_idx / max(1e-6, now - proc_t0)

            with state.lock:
                state.latest_jpeg = jpeg.tobytes()

            with metrics_lock:
                latest_metrics.update({
                    "queue1": int(q1_count),
                    "queue2": int(q2_count),
                    "queue1_avg_wait": round(q1_avg, 1) if q1_avg else None,
                    "queue2_avg_wait": round(q2_avg, 1) if q2_avg else None,
                    "queue1_trend": q1_trend,
                    "queue2_trend": q2_trend,
                    "employees": int(employee_count),
                    "clients_visible": int(len(visible)),
                    "store_count": int(store_count),
                    "status": "ALERT" if is_alert else "MONITORING",
                    "fps": round(processing_fps, 1),
                    "frame": int(frame_idx),
                })

            # Pace frame reads to match the source video's FPS.
            # If inference took longer than one frame period, skip the sleep
            # so we don't fall further behind; if it was faster, wait out
            # the remainder so the video plays at real-time speed.
            elapsed = time.time() - frame_start
            target  = 1.0 / video_fps
            if elapsed < target:
                time.sleep(target - elapsed)

        cap.release()
    except Exception as exc:
        with metrics_lock:
            latest_metrics["status"] = "error"
        print(f"[ERROR] Detector: {exc}")


# ── Agent worker ─────────────────────────────────────────────────────────────

def agent_worker(stop_event: threading.Event):
    """Call the LangGraph agent on a routine interval OR immediately when
    the detector fires a threshold alert via urgent_queue."""
    time.sleep(3)  # let detector warm up

    while not stop_event.is_set():
        # Only fire when an alert has triggered an urgent call
        try:
            snapshot = urgent_queue.get(timeout=1.0)
            print("[AGENT] Urgent call triggered by queue alert")
        except queue.Empty:
            continue

        alert_id = snapshot.pop("_alert_id", None)

        with agent_busy_lock:
            global agent_busy
            agent_busy = True

        print(f"[AGENT] Calling agent — Q1:{snapshot.get('queue1',0)} "
              f"Q2:{snapshot.get('queue2',0)} Store:{snapshot.get('store_count',0)}")
        result = run_agent(snapshot)
        print(f"[AGENT] {result.get('urgency','?').upper()} — {result.get('situation','')}")

        # Log the action taken (tool calls update metrics via /add_checkout or /remove_checkout)
        action = result.get("action", "none").lower()
        if "open_register" in action or "close_register" in action:
            log_event(f"Action taken: {result.get('action', 'none')}")

        with agent_busy_lock:
            agent_busy = False

        broadcast_event({
            "type": "agent_decision",
            "trigger": "cv_alert",
            "alert_id": alert_id,
            "timestamp": result.get("timestamp", time.strftime("%H:%M:%S")),
            "situation": result.get("situation", ""),
            "reasoning": result.get("reasoning", ""),
            "action": result.get("action", "none"),
            "urgency": result.get("urgency", "low"),
            "tool_result": result.get("tool_result"),
            "metrics": {
                "queue1":          snapshot.get("queue1", 0),
                "queue2":          snapshot.get("queue2", 0),
                "queue3":          snapshot.get("queue3", 0),
                "queue4":          snapshot.get("queue4", 0),
                "store_count":     snapshot.get("store_count", 0),
                "checkouts_open":  snapshot.get("checkouts_open", 2),
                "queue1_avg_wait": snapshot.get("queue1_avg_wait"),
                "queue2_avg_wait": snapshot.get("queue2_avg_wait"),
                "queue1_trend":    snapshot.get("queue1_trend", "stable"),
                "queue2_trend":    snapshot.get("queue2_trend", "stable"),
            },
        })


# ── Report worker ────────────────────────────────────────────────────────────

def report_worker(stop_event: threading.Event):
    """Every 60 seconds, run a scheduled LLM check and publish a blue event card."""
    stop_event.wait(60)  # wait for first full minute
    while not stop_event.is_set():
        with metrics_lock:
            snapshot = dict(latest_metrics)
        with minute_log_lock:
            log_copy = list(minute_log)
            minute_log.clear()

        if snapshot.get("status") not in ("running", "MONITORING", "ALERT"):
            stop_event.wait(60)
            continue

        print("[SCHEDULED] Running 60s LLM check...")
        result = run_agent(snapshot)
        print(f"[SCHEDULED] {result.get('urgency', '?').upper()} — {result.get('situation', '')}")

        action = result.get("action", "none").lower()
        if "open_register" in action or "close_register" in action:
            log_event(f"Scheduled action taken: {result.get('action', 'none')}")

        broadcast_event({
            "type": "scheduled_llm_call",
            "trigger": "scheduled",
            "timestamp": result.get("timestamp", time.strftime("%H:%M:%S")),
            "situation": result.get("situation", ""),
            "reasoning": result.get("reasoning", ""),
            "action": result.get("action", "none"),
            "urgency": result.get("urgency", "low"),
            "tool_result": result.get("tool_result"),
            "metrics": {
                "queue1":          snapshot.get("queue1", 0),
                "queue2":          snapshot.get("queue2", 0),
                "queue3":          snapshot.get("queue3", 0),
                "queue4":          snapshot.get("queue4", 0),
                "store_count":     snapshot.get("store_count", 0),
                "checkouts_open":  snapshot.get("checkouts_open", 2),
                "queue1_avg_wait": snapshot.get("queue1_avg_wait"),
                "queue2_avg_wait": snapshot.get("queue2_avg_wait"),
                "queue1_trend":    snapshot.get("queue1_trend", "stable"),
                "queue2_trend":    snapshot.get("queue2_trend", "stable"),
            },
        })
        stop_event.wait(60)


# ── Flask app ────────────────────────────────────────────────────────────────

def create_app(det_state: DetectorState):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/video")
    def video_stream():
        def generate():
            while not det_state.stop_event.is_set():
                with det_state.lock:
                    frame = det_state.latest_jpeg
                if frame is None:
                    time.sleep(0.03)
                    continue
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"
                time.sleep(0.016)  # ~60fps cap to avoid flooding
        return Response(stream_with_context(generate()), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/events")
    def sse_stream():
        """SSE endpoint — pushes agent decision events to the browser."""
        def generate():
            q = queue.Queue(maxsize=50)
            with sse_clients_lock:
                sse_clients.append(q)
            try:
                while True:
                    try:
                        msg = q.get(timeout=30)
                        yield msg
                    except queue.Empty:
                        yield ": keepalive\n\n"
            except GeneratorExit:
                with sse_clients_lock:
                    if q in sse_clients:
                        sse_clients.remove(q)

        return Response(stream_with_context(generate()), mimetype="text/event-stream",
                        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})

    @app.route("/push_event", methods=["POST"])
    def push_event():
        """Accept agent events from external processes (e.g. standalone detect.py)."""
        data = request.get_json(silent=True)
        if not data:
            return jsonify({"error": "invalid json"}), 400
        broadcast_event(data)
        return jsonify({"ok": True})

    @app.route("/metrics")
    def metrics():
        with metrics_lock:
            return jsonify(dict(latest_metrics))

    @app.route("/add_checkout", methods=["POST"])
    def add_checkout():
        """Open a new checkout (tool call from agent)."""
        data = request.get_json(silent=True) or {}
        lane_id = data.get("lane_id")
        
        with metrics_lock:
            # Check if already at max (2 base + 2 extra = 4)
            max_checkouts = 4
            if latest_metrics["checkouts_open"] >= max_checkouts:
                return jsonify({
                    "error": f"Maximum {max_checkouts} checkouts already open",
                    "checkouts_open": latest_metrics["checkouts_open"]
                }), 409
            
            # Add new checkout
            latest_metrics["checkouts_open"] += 1
            opened_checkout = latest_metrics["checkouts_open"]
            if opened_checkout in (3, 4):
                extra_checkout_state[opened_checkout]["count"] = 0
                extra_checkout_state[opened_checkout]["trend"] = 1
                latest_metrics[f"queue{opened_checkout}"] = 0
            latest_metrics["last_action"] = f"open_register({lane_id})"
            latest_metrics["last_action_time"] = time.time()
        
        log_event(f"Checkout opened (lane {lane_id}) — Total: {latest_metrics['checkouts_open']}")
        
        return jsonify({
            "ok": True,
            "checkouts_open": latest_metrics["checkouts_open"],
            "message": f"Checkout {latest_metrics['checkouts_open']} opened"
        })

    @app.route("/remove_checkout", methods=["POST"])
    def remove_checkout():
        """Close a checkout (tool call from agent)."""
        data = request.get_json(silent=True) or {}
        lane_id = data.get("lane_id")
        
        with metrics_lock:
            # Ensure we don't go below 2 (the base checkouts)
            min_checkouts = 2
            if latest_metrics["checkouts_open"] <= min_checkouts:
                return jsonify({
                    "error": f"Minimum {min_checkouts} checkouts must stay open",
                    "checkouts_open": latest_metrics["checkouts_open"]
                }), 409
            
            # Remove checkout
            latest_metrics["checkouts_open"] -= 1
            for checkout_num in (3, 4):
                if latest_metrics["checkouts_open"] < checkout_num:
                    extra_checkout_state[checkout_num]["count"] = 0
                    extra_checkout_state[checkout_num]["trend"] = 1
                    latest_metrics[f"queue{checkout_num}"] = 0
            latest_metrics["last_action"] = f"close_register({lane_id})"
            latest_metrics["last_action_time"] = time.time()
        
        log_event(f"Checkout closed (lane {lane_id}) — Total: {latest_metrics['checkouts_open']}")
        
        return jsonify({
            "ok": True,
            "checkouts_open": latest_metrics["checkouts_open"],
            "message": f"Checkout closed — Total: {latest_metrics['checkouts_open']}"
        })

    return app


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Queue Monitor — Flask + YOLO + Agent")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--model", default="yolov8s.pt")
    parser.add_argument("--conf", type=float, default=CONF_THRESH)
    parser.add_argument("--device", default="0", help="YOLO device: cpu, 0, cuda:0")
    parser.add_argument("--zone1", default=None, help="Queue 1 polygon: x1,y1,...,x4,y4")
    parser.add_argument("--zone2", default=None, help="Queue 2 polygon: x1,y1,...,x4,y4")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-loop", action="store_true", help="Stop at end of video")
    parser.add_argument("--no-agent", action="store_true", help="Disable LLM agent")
    args = parser.parse_args()

    zone1 = parse_zone(args.zone1) if args.zone1 else None
    zone2 = parse_zone(args.zone2) if args.zone2 else None

    det_state = DetectorState(
        args.video, args.model, args.conf, zone1, zone2, args.device, not args.no_loop
    )

    # Start detector thread
    det_thread = threading.Thread(target=detector_worker, args=(det_state,), daemon=True)
    det_thread.start()

    # Start agent thread
    if not args.no_agent:
        agent_thread = threading.Thread(target=agent_worker, args=(det_state.stop_event,), daemon=True)
        agent_thread.start()
        print("[INFO] Agent worker started")
        report_thread = threading.Thread(target=report_worker, args=(det_state.stop_event,), daemon=True)
        report_thread.start()
        print("[INFO] Report worker started (every 60s)")
    else:
        print("[INFO] Agent disabled (--no-agent)")

    app = create_app(det_state)
    print(f"[INFO] Open http://localhost:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
