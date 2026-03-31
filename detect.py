"""
Queue Detector — Simple Video Output
Runs YOLOv8 person detection on a video file and saves
an annotated output video.

Usage:
    python detect.py --video path/to/your/footage.mp4 --output out.mp4
    python detect.py --video footage.mp4 --zone1 100,200,400,600 --zone2 450,200,750,600
"""

import cv2
import json
import numpy as np
import random
import requests
import threading
from types import SimpleNamespace
from ultralytics.trackers.byte_tracker import BYTETracker
import argparse
import time
import os

# ── track registry ────────────────────────────────────────────────────────────
STABLE_FRAMES        = 3     # frames before a new track is drawn
GRACE_FRAMES         = 20    # frames a track stays visible after going missing
REID_IOU_THRESH      = 0.15  # minimum IoU to re-identify a lost track
REID_MAX_AGE         = 90    # frames to keep a lost track candidate for re-ID
REID_CENTROID_FACTOR = 0.8   # centroid fallback: max dist = factor × avg(box_w, box_h)
ZONE_MIN_DWELL_SEC   = 1.0   # seconds a person must stay in a zone before being counted


def _iou(a, b):
    """IoU between two boxes [x1,y1,x2,y2]."""
    ix1 = max(a[0], b[0]); iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2]); iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2]-a[0]) * (a[3]-a[1])
    area_b = (b[2]-b[0]) * (b[3]-b[1])
    return inter / (area_a + area_b - inter)


class TrackRegistry:
    def __init__(self, video_fps: float = 30.0):
        self._fps        = max(video_fps, 1.0)
        # tracker_id  → canonical_id (our stable ID, may differ from tracker's)
        self._id_map     = {}
        # canonical_id → consecutive frames seen
        self._hits       = {}
        # canonical_id → last frame seen
        self._last_seen  = {}
        # canonical_id → last box  (for re-ID)
        self._last_box   = {}
        # canonical_id → smoothed box (EMA to reduce jitter)
        self._smooth_box = {}
        self._next_id    = 1

    def _new_canonical(self):
        cid = self._next_id
        self._next_id += 1
        self._hits[cid] = 0
        return cid

    def _try_reid(self, box, frame_idx):
        """Return canonical_id of best matching lost track, or None."""
        best_iou, best_cid = 0.0, None
        for cid, last_box in self._last_box.items():
            age = frame_idx - self._last_seen.get(cid, 0)
            if age == 0 or age > REID_MAX_AGE:
                continue
            iou = _iou(box, last_box)
            if iou > best_iou:
                best_iou, best_cid = iou, cid
        if best_iou >= REID_IOU_THRESH:
            return best_cid

        # Fallback: centroid proximity (tolerates moderate positional shift)
        best_dist, best_cid2 = float('inf'), None
        cx = (box[0] + box[2]) / 2;  cy = (box[1] + box[3]) / 2
        bw = box[2] - box[0];        bh = box[3] - box[1]
        ref = (bw + bh) / 2
        for cid, last_box in self._last_box.items():
            age = frame_idx - self._last_seen.get(cid, 0)
            if age == 0 or age > REID_MAX_AGE:
                continue
            lcx = (last_box[0] + last_box[2]) / 2
            lcy = (last_box[1] + last_box[3]) / 2
            dist = ((cx - lcx) ** 2 + (cy - lcy) ** 2) ** 0.5
            if dist < best_dist and dist < ref * REID_CENTROID_FACTOR:
                best_dist, best_cid2 = dist, cid
        return best_cid2

    def update(self, raw_tracks, frame_idx):
        """
        Feed BYTETracker output for this frame.
        Returns list of (box, canonical_id).
        """
        active_tracker_ids = set()

        for track in raw_tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            tid  = int(track[4])
            box  = (x1, y1, x2, y2)
            active_tracker_ids.add(tid)

            if tid not in self._id_map:
                cid = self._try_reid(box, frame_idx)
                if cid is not None:
                    self._hits[cid] = STABLE_FRAMES  # immediately stable
                else:
                    cid = self._new_canonical()
                self._id_map[tid] = cid

            cid = self._id_map[tid]
            self._hits[cid]      = self._hits.get(cid, 0) + 1
            self._last_seen[cid] = frame_idx
            self._last_box[cid]  = box

            # Exponential moving average on box coords to reduce jitter
            alpha = 0.9
            if cid in self._smooth_box:
                sb = self._smooth_box[cid]
                self._smooth_box[cid] = tuple(
                    int(alpha * b + (1 - alpha) * s)
                    for b, s in zip(box, sb)
                )
            else:
                self._smooth_box[cid] = box

        # Build visible output: stable tracks + grace-period tracks
        output = []
        seen_cids = {self._id_map[tid] for tid in active_tracker_ids if tid in self._id_map}

        for cid in list(self._hits.keys()):
            age_unseen = frame_idx - self._last_seen.get(cid, frame_idx)
            hits       = self._hits.get(cid, 0)

            if cid in seen_cids:
                if hits < STABLE_FRAMES:
                    continue  # not yet stable, skip (anti-flicker)
            else:
                if age_unseen > GRACE_FRAMES:
                    continue  # outside grace window, skip

            box = self._smooth_box.get(cid, self._last_box.get(cid))
            if box:
                output.append((box, cid))

        return output

    def active_count(self, frame_idx):
        """Number of stable, currently-visible tracks."""
        count = 0
        for cid, last in self._last_seen.items():
            if frame_idx - last <= GRACE_FRAMES and self._hits.get(cid, 0) >= STABLE_FRAMES:
                count += 1
        return count


# ── store counter ────────────────────────────────────────────────────────────
class StoreCounter:
    """Simulates total store occupancy based on detected people in view.
    Steps by at most ±1 every 3 seconds toward a target, keeping the
    displayed value stable and realistic (target range 20-30)."""

    def __init__(self):
        self._current = 22          # start mid-range
        self._last_step = time.time()
        self.STEP_INTERVAL = 3.0    # seconds between ±1 steps

    def update(self, detected_count: int) -> int:
        # Target: detected people in checkout × 2 + small base, clamped 20-30
        target = max(20, min(40, detected_count * 1.2 + 12))

        now = time.time()
        if now - self._last_step >= self.STEP_INTERVAL:
            if self._current < target:
                self._current += 1
            elif self._current > target:
                self._current -= 1
            self._last_step = now

        return self._current


# ── trend helper ─────────────────────────────────────────────────────────────
def compute_trend(history: list) -> str:
    """Return 'growing', 'stable', or 'declining' based on the last 3 readings."""
    if len(history) < 3:
        return "stable"
    last3 = history[-3:]
    if last3[-1] > last3[0] + 1:
        return "growing"
    elif last3[-1] < last3[0] - 1:
        return "declining"
    return "stable"


# ── metrics reporter ─────────────────────────────────────────────────────────
class MetricsReporter:
    """Background thread that POSTs metrics + agent events to the Flask server."""

    def __init__(self, server_url: str, interval: float = 5.0):
        self._url = server_url.rstrip("/") + "/push_event"
        self._interval = interval
        self._lock = threading.Lock()
        self._latest = {}
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._thread.start()

    def stop(self):
        self._stop.set()

    def update(self, metrics: dict):
        with self._lock:
            self._latest = dict(metrics)

    def _run(self):
        while not self._stop.is_set():
            self._stop.wait(self._interval)
            if self._stop.is_set():
                break
            with self._lock:
                snapshot = dict(self._latest)
            if not snapshot:
                continue
            try:
                # Try to call the agent first
                try:
                    from agent import run_agent
                    result = run_agent(snapshot)
                except Exception:
                    result = {
                        "type": "agent_decision",
                        "timestamp": time.strftime("%H:%M:%S"),
                        "situation": "Agent unavailable",
                        "reasoning": "Ollama not reachable",
                        "action": "none",
                        "urgency": "low",
                        "tool_result": None,
                    }

                event = {
                    "type": "agent_decision",
                    "timestamp": result.get("timestamp", time.strftime("%H:%M:%S")),
                    "situation": result.get("situation", ""),
                    "reasoning": result.get("reasoning", ""),
                    "action": result.get("action", "none"),
                    "urgency": result.get("urgency", "low"),
                    "tool_result": result.get("tool_result"),
                }
                requests.post(self._url, json=event, timeout=2)
            except Exception as e:
                print(f"[REPORTER] POST failed: {e}")


# ── dependency check ──────────────────────────────────────────────────────────
try:
    from ultralytics import YOLO
except ImportError:
    raise SystemExit(
        "\n[ERROR] ultralytics not installed.\n"
        "Run:  pip install ultralytics\n"
    )


# ── config ────────────────────────────────────────────────────────────────────
MODEL_NAME     = "yolov8s.pt"
CLIENT_CLASS   = 0   # 'Client'   in the fine-tuned model
EMPLOYEE_CLASS = 1   # 'Employee' in the fine-tuned model
CONF_THRESH    = 0.35
BOX_COLOR      = (0, 255, 120)    # green  — client
EMPLOYEE_COLOR = (255, 100, 50)   # orange — employee
ZONE_COLORS    = [(60, 180, 255), (255, 80, 180)]  # zone1=yellow-ish, zone2=pink
TEXT_COLOR     = (255, 255, 255)
OVERLAY_BG     = (0, 0, 0)
FPS_TARGET     = 30

TRACKER_ARGS = SimpleNamespace(
    track_high_thresh=0.35,
    track_low_thresh=0.25,
    new_track_thresh=0.45,
    track_buffer=150,
    match_thresh=0.5,
    fuse_score=True,
)


# ── zone helpers ──────────────────────────────────────────────────────────────
def parse_zone(s):
    """Parse 'x1,y1,x2,y2,x3,y3,x4,y4' into a numpy (4,2) int32 polygon."""
    parts = [int(v) for v in s.split(",")]
    if len(parts) != 8:
        raise ValueError(f"Zone must be 8 values x1,y1,x2,y2,x3,y3,x4,y4 — got: {s}")
    return np.array(parts, dtype="int32").reshape((4, 2))


def centroid_in_zone(box, zone):
    """Return True if the box centroid is inside the zone polygon."""
    if zone is None:
        return False
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return cv2.pointPolygonTest(zone, (cx, cy), measureDist=False) >= 0


def draw_zone(frame, zone, color, label):
    """Draw a semi-transparent filled zone polygon with a label."""
    if zone is None:
        return
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone], color)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.polylines(frame, [zone], isClosed=True, color=color, thickness=2)
    lx, ly = zone[0]
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (lx, ly - th - 8), (lx + tw + 6, ly), color, -1)
    cv2.putText(frame, label, (lx + 3, ly - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, OVERLAY_BG, 2, cv2.LINE_AA)


# ── drawing helpers ───────────────────────────────────────────────────────────
def draw_box(frame, x1, y1, x2, y2, color=None, label=""):
    c = color or BOX_COLOR
    cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
    text = label or "?"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), c, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, OVERLAY_BG, 1, cv2.LINE_AA)


def draw_hud(frame, employee_count, fps, queue_counts, avg_waits=None):
    lines = [f"FPS: {fps:.1f}", f"Employees: {employee_count}"]
    for i, (label, count) in enumerate(queue_counts):
        lines.append(f"{label}: {count}")
        if avg_waits and i < len(avg_waits) and avg_waits[i] is not None:
            lines.append(f"  avg wait: {avg_waits[i]:.1f}s")

    pad    = 10
    line_h = 28
    box_h  = pad * 2 + line_h * len(lines)
    cv2.rectangle(frame, (10, 10), (280, 10 + box_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (280, 10 + box_h), BOX_COLOR, 1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line,
                    (10 + pad, 10 + pad + line_h * i + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)


# ── main pipeline ─────────────────────────────────────────────────────────────
def run(video_path: str, output_path: str, preview: bool = True,
        zone1=None, zone2=None, server_url: str = None):
    t_start = time.time()

    print(f"\n[INFO] Loading model: {MODEL_NAME}")
    model   = YOLO(MODEL_NAME)
    tracker = BYTETracker(TRACKER_ARGS, frame_rate=30)

    print(f"[INFO] Opening video: {video_path}")
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Cannot open video: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    video_fps    = cap.get(cv2.CAP_PROP_FPS) or FPS_TARGET
    width        = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height       = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    registry = TrackRegistry(video_fps)
    store_counter = StoreCounter()
    q1_history = []
    q2_history = []

    # Start metrics reporter if server URL provided
    reporter = None
    if server_url:
        reporter = MetricsReporter(server_url, interval=5.0)
        reporter.start()
        print(f"[INFO] Reporting metrics to {server_url} every 5s")

    print(f"[INFO] Video: {total_frames} frames @ {video_fps:.1f} fps")
    if zone1 is not None:
        print(f"[INFO] Zone 1: {zone1.tolist()}")
    if zone2 is not None:
        print(f"[INFO] Zone 2: {zone2.tolist()}")
    print(f"[INFO] Writing output to: {output_path}\n")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise SystemExit(f"[ERROR] Cannot create output video: {output_path}")

    frame_idx      = 0
    visible        = []
    employee_count = 0
    employee_boxes = None
    zone_dwell          = [{}, {}]   # zone_dwell[i][cid] = consecutive frames in zone i
    zone_entry_frame    = [{}, {}]   # zone_entry_frame[i][cid] = frame when cid entered zone i
    zone_completed_waits= [[], []]   # finished wait durations in seconds, per zone
    preview_enabled = preview
    delay_ms        = max(1, int(1000 / max(video_fps, 1)))

    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    if preview_enabled and not has_display:
        print("[WARN] No display detected — running headless.")
        preview_enabled = False
    if preview_enabled:
        try:
            cv2.namedWindow("Queue Detector", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Queue Detector", 1280, 720)
            print("[INFO] Preview enabled. Press 'q' to quit early.")
        except cv2.error as exc:
            print(f"[WARN] Preview failed: {exc} — running headless.")
            preview_enabled = False

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video")
            break

        frame_idx += 1

        # ── detection ─────────────────────────────────────────────────────────
        results = model(
            frame,
            classes=[CLIENT_CLASS, EMPLOYEE_CLASS],
            conf=CONF_THRESH,
            device=0,
            verbose=False,
        )[0]

        all_boxes = results.boxes.cpu() if results.boxes is not None else None
        if all_boxes is not None and len(all_boxes) > 0:
            client_boxes   = all_boxes[all_boxes.cls == CLIENT_CLASS]
            employee_boxes = all_boxes[all_boxes.cls == EMPLOYEE_CLASS]
        else:
            client_boxes   = None
            employee_boxes = None

        raw_tracks     = tracker.update(client_boxes, frame) if client_boxes is not None and len(client_boxes) > 0 else []
        visible        = registry.update(raw_tracks, frame_idx)
        employee_count = 0 if employee_boxes is None else len(employee_boxes)

        # ── zone counts ───────────────────────────────────────────────────────
        # ── dwell tracking — only count after ZONE_MIN_DWELL_SEC in zone ────────
        min_frames = ZONE_MIN_DWELL_SEC * video_fps
        visible_cids = {cid for _, cid in visible}
        for i, zone in enumerate([zone1, zone2]):
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
            # clean up tracks no longer visible
            for cid in list(zone_dwell[i]):
                if cid not in visible_cids:
                    if zone_dwell[i][cid] >= min_frames:
                        wait = (frame_idx - zone_entry_frame[i].get(cid, frame_idx)) / video_fps
                        zone_completed_waits[i].append(wait)
                    zone_dwell[i].pop(cid, None)
                    zone_entry_frame[i].pop(cid, None)

        q1_count = sum(1 for cid, f in zone_dwell[0].items() if f >= min_frames)
        q2_count = sum(1 for cid, f in zone_dwell[1].items() if f >= min_frames)

        # ── store count + trends ─────────────────────────────────────────────
        store_count = store_counter.update(len(visible))
        if frame_idx % max(1, int(video_fps)) == 0:
            q1_history.append(q1_count)
            q2_history.append(q2_count)
        q1_trend = compute_trend(q1_history)
        q2_trend = compute_trend(q2_history)

        # ── report metrics to server ─────────────────────────────────────────
        if reporter and frame_idx % max(1, int(video_fps)) == 0:
            # compute avg waits for reporter
            _q1_avg, _q2_avg = None, None
            if zone1 is not None:
                _w = zone_completed_waits[0] + [
                    (frame_idx - zone_entry_frame[0].get(c, frame_idx)) / video_fps
                    for c, f in zone_dwell[0].items() if f >= min_frames
                ]
                _q1_avg = round(sum(_w) / len(_w), 1) if _w else None
            if zone2 is not None:
                _w = zone_completed_waits[1] + [
                    (frame_idx - zone_entry_frame[1].get(c, frame_idx)) / video_fps
                    for c, f in zone_dwell[1].items() if f >= min_frames
                ]
                _q2_avg = round(sum(_w) / len(_w), 1) if _w else None
            reporter.update({
                "queue1": q1_count,
                "queue2": q2_count,
                "queue1_avg_wait": _q1_avg,
                "queue2_avg_wait": _q2_avg,
                "queue1_trend": q1_trend,
                "queue2_trend": q2_trend,
                "employees": employee_count,
                "clients_visible": len(visible),
                "store_count": store_count,
            })

        # ── draw client tracks ────────────────────────────────────────────────
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

        # ── draw employees ────────────────────────────────────────────────────
        if employee_boxes is not None:
            for box in employee_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                draw_box(frame, x1, y1, x2, y2, color=EMPLOYEE_COLOR, label="Employee")

        # ── HUD ───────────────────────────────────────────────────────────────
        queue_counts = []
        avg_waits    = []
        if zone1 is not None:
            queue_counts.append(("Queue 1", q1_count))
            all_w = zone_completed_waits[0] + [
                (frame_idx - zone_entry_frame[0].get(cid, frame_idx)) / video_fps
                for cid, f in zone_dwell[0].items() if f >= min_frames
            ]
            avg_waits.append(sum(all_w) / len(all_w) if all_w else None)
        if zone2 is not None:
            queue_counts.append(("Queue 2", q2_count))
            all_w = zone_completed_waits[1] + [
                (frame_idx - zone_entry_frame[1].get(cid, frame_idx)) / video_fps
                for cid, f in zone_dwell[1].items() if f >= min_frames
            ]
            avg_waits.append(sum(all_w) / len(all_w) if all_w else None)
        if not queue_counts:
            queue_counts.append(("Clients", len(visible)))
        draw_hud(frame, employee_count, video_fps, queue_counts, avg_waits)

        out.write(frame)

        if preview_enabled:
            cv2.imshow("Queue Detector", frame)
            if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                print("[INFO] Stopped by user.")
                break

        if frame_idx % 30 == 0:
            elapsed = time.time() - t_start
            print(f"[{elapsed:.1f}]s Frames:{frame_idx}/{total_frames} "
                  f"| Q1:{q1_count} Q2:{q2_count} Employees:{employee_count}")

    cap.release()
    out.release()
    if reporter:
        reporter.stop()
    if preview_enabled:
        cv2.destroyAllWindows()

    total = time.time() - t_start
    print(f"[INFO] Done in {int(total//60)}m {int(total%60)}s.")

    for i, label in enumerate(["Queue 1", "Queue 2"]):
        waits = zone_completed_waits[i]
        if waits:
            print(f"[STATS] {label}: {len(waits)} visits | "
                  f"avg wait {sum(waits)/len(waits):.1f}s | "
                  f"min {min(waits):.1f}s | max {max(waits):.1f}s")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 queue detector")
    parser.add_argument("--video",      required=True)
    parser.add_argument("--output",     default="output.mp4")
    parser.add_argument("--conf",       type=float, default=CONF_THRESH)
    parser.add_argument("--model",      default=MODEL_NAME)
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument("--zone1", default=None, help="Queue 1 as 4 points: x1,y1,x2,y2,x3,y3,x4,y4")
    parser.add_argument("--zone2", default=None, help="Queue 2 as 4 points: x1,y1,x2,y2,x3,y3,x4,y4")
    parser.add_argument("--server-url", default=None,
                        help="Flask server URL to POST agent events to (e.g. http://localhost:8000)")
    args = parser.parse_args()

    CONF_THRESH = args.conf
    MODEL_NAME  = args.model

    zone1 = parse_zone(args.zone1) if args.zone1 else None
    zone2 = parse_zone(args.zone2) if args.zone2 else None

    run(args.video, args.output, preview=not args.no_preview,
        zone1=zone1, zone2=zone2, server_url=args.server_url)
