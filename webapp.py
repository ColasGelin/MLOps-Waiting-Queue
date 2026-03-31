import argparse
import threading
import time
from types import SimpleNamespace

import cv2
import numpy as np
from flask import Flask, Response, jsonify, render_template
from ultralytics import YOLO
from ultralytics.trackers.byte_tracker import BYTETracker


STABLE_FRAMES = 3
GRACE_FRAMES = 20
REID_IOU_THRESH = 0.15
REID_MAX_AGE = 90
REID_CENTROID_FACTOR = 0.8
ZONE_MIN_DWELL_SEC = 1.0

MODEL_NAME = "yolov8s.pt"
CLIENT_CLASS = 0
EMPLOYEE_CLASS = 1
CONF_THRESH = 0.35
BOX_COLOR = (0, 255, 120)
EMPLOYEE_COLOR = (255, 100, 50)
ZONE_COLORS = [(60, 180, 255), (255, 80, 180)]
TEXT_COLOR = (255, 255, 255)
OVERLAY_BG = (0, 0, 0)

TRACKER_ARGS = SimpleNamespace(
    track_high_thresh=0.35,
    track_low_thresh=0.25,
    new_track_thresh=0.45,
    track_buffer=150,
    match_thresh=0.5,
    fuse_score=True,
)


def _iou(a, b):
    ix1 = max(a[0], b[0])
    iy1 = max(a[1], b[1])
    ix2 = min(a[2], b[2])
    iy2 = min(a[3], b[3])
    inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if inter == 0:
        return 0.0
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    return inter / (area_a + area_b - inter)


def parse_zone(s):
    parts = [int(v) for v in s.split(",")]
    if len(parts) != 8:
        raise ValueError(f"Zone must be 8 values x1,y1,x2,y2,x3,y3,x4,y4, got: {s}")
    return np.array(parts, dtype="int32").reshape((4, 2))


def centroid_in_zone(box, zone):
    if zone is None:
        return False
    cx = (box[0] + box[2]) / 2
    cy = (box[1] + box[3]) / 2
    return cv2.pointPolygonTest(zone, (cx, cy), measureDist=False) >= 0


def draw_zone(frame, zone, color, label):
    if zone is None:
        return
    overlay = frame.copy()
    cv2.fillPoly(overlay, [zone], color)
    cv2.addWeighted(overlay, 0.15, frame, 0.85, 0, frame)
    cv2.polylines(frame, [zone], isClosed=True, color=color, thickness=2)
    lx, ly = zone[0]
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
    cv2.rectangle(frame, (lx, ly - th - 8), (lx + tw + 6, ly), color, -1)
    cv2.putText(frame, label, (lx + 3, ly - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, OVERLAY_BG, 2, cv2.LINE_AA)


def draw_box(frame, x1, y1, x2, y2, color=None, label=""):
    c = color or BOX_COLOR
    cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)
    text = label or "?"
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), c, -1)
    cv2.putText(frame, text, (x1 + 3, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, OVERLAY_BG, 1, cv2.LINE_AA)


def draw_hud(frame, employee_count, fps, queue_counts, avg_waits=None):
    lines = [f"FPS: {fps:.1f}", f"Employees: {employee_count}"]
    for i, (label, count) in enumerate(queue_counts):
        lines.append(f"{label}: {count}")
        if avg_waits and i < len(avg_waits) and avg_waits[i] is not None:
            lines.append(f"  avg wait: {avg_waits[i]:.1f}s")

    pad = 10
    line_h = 28
    box_h = pad * 2 + line_h * len(lines)
    cv2.rectangle(frame, (10, 10), (320, 10 + box_h), (0, 0, 0), -1)
    cv2.rectangle(frame, (10, 10), (320, 10 + box_h), BOX_COLOR, 1)
    for i, line in enumerate(lines):
        cv2.putText(frame, line, (10 + pad, 10 + pad + line_h * i + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6, TEXT_COLOR, 1, cv2.LINE_AA)


class TrackRegistry:
    def __init__(self, video_fps=30.0):
        self._id_map = {}
        self._hits = {}
        self._last_seen = {}
        self._last_box = {}
        self._smooth_box = {}
        self._next_id = 1

    def _new_canonical(self):
        cid = self._next_id
        self._next_id += 1
        self._hits[cid] = 0
        return cid

    def _try_reid(self, box, frame_idx):
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

        best_dist, best_cid2 = float("inf"), None
        cx = (box[0] + box[2]) / 2
        cy = (box[1] + box[3]) / 2
        bw = box[2] - box[0]
        bh = box[3] - box[1]
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
        active_tracker_ids = set()
        for track in raw_tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            tid = int(track[4])
            box = (x1, y1, x2, y2)
            active_tracker_ids.add(tid)

            if tid not in self._id_map:
                cid = self._try_reid(box, frame_idx)
                if cid is not None:
                    self._hits[cid] = STABLE_FRAMES
                else:
                    cid = self._new_canonical()
                self._id_map[tid] = cid

            cid = self._id_map[tid]
            self._hits[cid] = self._hits.get(cid, 0) + 1
            self._last_seen[cid] = frame_idx
            self._last_box[cid] = box

            alpha = 0.9
            if cid in self._smooth_box:
                sb = self._smooth_box[cid]
                self._smooth_box[cid] = tuple(int(alpha * b + (1 - alpha) * s) for b, s in zip(box, sb))
            else:
                self._smooth_box[cid] = box

        output = []
        seen_cids = {self._id_map[tid] for tid in active_tracker_ids if tid in self._id_map}
        for cid in list(self._hits.keys()):
            age_unseen = frame_idx - self._last_seen.get(cid, frame_idx)
            hits = self._hits.get(cid, 0)
            if cid in seen_cids:
                if hits < STABLE_FRAMES:
                    continue
            else:
                if age_unseen > GRACE_FRAMES:
                    continue
            box = self._smooth_box.get(cid, self._last_box.get(cid))
            if box:
                output.append((box, cid))
        return output


class StreamState:
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
        self.start_time = time.time()
        self.stop_event = threading.Event()
        self.metrics = {
            "fps": 0.0,
            "processing_fps": 0.0,
            "employees": 0,
            "queue1": 0,
            "queue2": 0,
            "queue1_avg_wait": None,
            "queue2_avg_wait": None,
            "clients_visible": 0,
            "frame": 0,
            "total_frames": 0,
            "status": "initializing",
            "uptime_sec": 0,
            "model": model_name,
            "source": video_path,
            "stream_loops": 0,
            "error": None,
        }


def detector_worker(state):
    try:
        model = YOLO(state.model_name)
        tracker = BYTETracker(TRACKER_ARGS, frame_rate=30)
        cap = cv2.VideoCapture(state.video_path)
        if not cap.isOpened():
            with state.lock:
                state.metrics["status"] = "error"
                state.metrics["error"] = f"Cannot open video: {state.video_path}"
            return

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_idx = 0
        proc_t0 = time.time()
        registry = TrackRegistry(video_fps)
        zone_dwell = [{}, {}]
        zone_entry_frame = [{}, {}]
        zone_completed_waits = [[], []]

        with state.lock:
            state.metrics["total_frames"] = total_frames
            state.metrics["status"] = "running"

        while not state.stop_event.is_set():
            ret, frame = cap.read()
            if not ret:
                if state.loop_video:
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    frame_idx = 0
                    zone_dwell = [{}, {}]
                    zone_entry_frame = [{}, {}]
                    zone_completed_waits = [[], []]
                    with state.lock:
                        state.metrics["stream_loops"] += 1
                    continue
                with state.lock:
                    state.metrics["status"] = "ended"
                break

            frame_idx += 1
            results = model(frame, classes=[CLIENT_CLASS, EMPLOYEE_CLASS], conf=state.conf, device=state.device, verbose=False)[0]
            all_boxes = results.boxes.cpu() if results.boxes is not None else None
            if all_boxes is not None and len(all_boxes) > 0:
                client_boxes = all_boxes[all_boxes.cls == CLIENT_CLASS]
                employee_boxes = all_boxes[all_boxes.cls == EMPLOYEE_CLASS]
            else:
                client_boxes = None
                employee_boxes = None

            raw_tracks = tracker.update(client_boxes, frame) if client_boxes is not None and len(client_boxes) > 0 else []
            visible = registry.update(raw_tracks, frame_idx)
            employee_count = 0 if employee_boxes is None else len(employee_boxes)

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

            draw_zone(frame, state.zone1, ZONE_COLORS[0], "Queue 1")
            draw_zone(frame, state.zone2, ZONE_COLORS[1], "Queue 2")

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
            q1_avg = None
            q2_avg = None
            if state.zone1 is not None:
                queue_counts.append(("Queue 1", q1_count))
                all_w = zone_completed_waits[0] + [
                    (frame_idx - zone_entry_frame[0].get(cid, frame_idx)) / video_fps
                    for cid, f in zone_dwell[0].items() if f >= min_frames
                ]
                q1_avg = (sum(all_w) / len(all_w)) if all_w else None
                avg_waits.append(q1_avg)
            if state.zone2 is not None:
                queue_counts.append(("Queue 2", q2_count))
                all_w = zone_completed_waits[1] + [
                    (frame_idx - zone_entry_frame[1].get(cid, frame_idx)) / video_fps
                    for cid, f in zone_dwell[1].items() if f >= min_frames
                ]
                q2_avg = (sum(all_w) / len(all_w)) if all_w else None
                avg_waits.append(q2_avg)
            if not queue_counts:
                queue_counts.append(("Clients", len(visible)))

            draw_hud(frame, employee_count, video_fps, queue_counts, avg_waits)
            ok, jpeg = cv2.imencode(".jpg", frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
            if not ok:
                continue

            now = time.time()
            processing_fps = frame_idx / max(1e-6, now - proc_t0)
            with state.lock:
                state.latest_jpeg = jpeg.tobytes()
                state.metrics.update({
                    "fps": float(video_fps),
                    "processing_fps": float(processing_fps),
                    "employees": int(employee_count),
                    "queue1": int(q1_count),
                    "queue2": int(q2_count),
                    "queue1_avg_wait": q1_avg,
                    "queue2_avg_wait": q2_avg,
                    "clients_visible": int(len(visible)),
                    "frame": int(frame_idx),
                    "uptime_sec": int(now - state.start_time),
                    "status": "running",
                })
        cap.release()
    except Exception as exc:
        with state.lock:
            state.metrics["status"] = "error"
            state.metrics["error"] = str(exc)


def create_app(state):
    app = Flask(__name__)

    @app.route("/")
    def index():
        return render_template("index.html")

    @app.route("/metrics")
    def metrics():
        with state.lock:
            return jsonify(state.metrics)

    @app.route("/video")
    def stream_video():
        def generate():
            while not state.stop_event.is_set():
                with state.lock:
                    frame = state.latest_jpeg
                if frame is None:
                    time.sleep(0.05)
                    continue
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    @app.route("/stream.mjpg")
    def stream_mjpg():
        def generate():
            while not state.stop_event.is_set():
                with state.lock:
                    frame = state.latest_jpeg
                if frame is None:
                    time.sleep(0.05)
                    continue
                yield b"--frame\r\nContent-Type: image/jpeg\r\n\r\n" + frame + b"\r\n"

        return Response(generate(), mimetype="multipart/x-mixed-replace; boundary=frame")

    return app


def main():
    parser = argparse.ArgumentParser(description="Queue detector livestream web UI")
    parser.add_argument("--video", required=True, help="Path to video file")
    parser.add_argument("--model", default=MODEL_NAME)
    parser.add_argument("--conf", type=float, default=CONF_THRESH)
    parser.add_argument("--device", default="cpu", help="YOLO device: cpu, 0, etc")
    parser.add_argument("--zone1", default=None, help="x1,y1,x2,y2,x3,y3,x4,y4")
    parser.add_argument("--zone2", default=None, help="x1,y1,x2,y2,x3,y3,x4,y4")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--no-loop", action="store_true", help="Stop when video ends")
    args = parser.parse_args()

    zone1 = parse_zone(args.zone1) if args.zone1 else None
    zone2 = parse_zone(args.zone2) if args.zone2 else None

    state = StreamState(args.video, args.model, args.conf, zone1, zone2, args.device, not args.no_loop)
    worker = threading.Thread(target=detector_worker, args=(state,), daemon=True)
    worker.start()

    app = create_app(state)
    print(f"[INFO] Open http://{args.host}:{args.port}")
    app.run(host=args.host, port=args.port, threaded=True)


if __name__ == "__main__":
    main()
