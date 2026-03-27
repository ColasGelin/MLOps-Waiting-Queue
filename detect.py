"""
Queue Detector — Simple Video Output
Runs YOLOv8 person detection on a video file and saves
an annotated output video.

Usage:
    python detect.py --video path/to/your/footage.mp4 --output out.mp4
"""

import cv2
from types import SimpleNamespace
from ultralytics.trackers.byte_tracker import BYTETracker
import argparse
import time
import os

# ── track registry ────────────────────────────────────────────────────────────
# Fixes two problems:
#   1. Flickering  — only render a track after STABLE_FRAMES consecutive hits;
#                    keep rendering GRACE_FRAMES after last seen.
#   2. ID loss     — when a new track appears, check if it overlaps an recently
#                    lost track (IoU ≥ REID_IOU_THRESH). If yes, inherit the old
#                    ID so wait-time is preserved.

STABLE_FRAMES    = 3     # frames before a new track is drawn
GRACE_FRAMES     = 10    # frames a track stays visible after going missing
REID_IOU_THRESH  = 0.35  # minimum IoU to re-identify a lost track
REID_MAX_AGE     = 90    # frames to keep a lost track candidate for re-ID


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
        # canonical_id → entry frame (when client first appeared)
        self._entry      = {}
        # canonical_id → consecutive frames seen
        self._hits       = {}
        # canonical_id → last frame seen
        self._last_seen  = {}
        # canonical_id → last box  (for re-ID)
        self._last_box   = {}
        # canonical_id → smoothed box (EMA to reduce jitter)
        self._smooth_box = {}
        self._next_id    = 1
        self._frame      = 0

    def _new_canonical(self, entry_frame):
        cid = self._next_id
        self._next_id += 1
        self._entry[cid]  = entry_frame
        self._hits[cid]   = 0
        return cid

    def _try_reid(self, box, frame_idx):
        """Return canonical_id of best matching lost track, or None."""
        best_iou, best_cid = 0.0, None
        for cid, last_box in self._last_box.items():
            age = frame_idx - self._last_seen.get(cid, 0)
            if age == 0 or age > REID_MAX_AGE:
                continue  # still active or too old
            iou = _iou(box, last_box)
            if iou > best_iou:
                best_iou, best_cid = iou, cid
        if best_iou >= REID_IOU_THRESH:
            return best_cid
        return None

    def update(self, raw_tracks, frame_idx):
        """
        Feed BYTETracker output for this frame.
        Returns list of (box, canonical_id, wait_seconds, smoothed_box).
        """
        self._frame = frame_idx
        active_tracker_ids = set()

        for track in raw_tracks:
            x1, y1, x2, y2 = map(int, track[:4])
            tid  = int(track[4])
            box  = (x1, y1, x2, y2)
            active_tracker_ids.add(tid)

            if tid not in self._id_map:
                # New tracker ID — try to re-identify from recently lost track
                cid = self._try_reid(box, frame_idx)
                if cid is not None:
                    # Restore the old canonical ID
                    self._hits[cid] = STABLE_FRAMES  # immediately stable
                else:
                    cid = self._new_canonical(frame_idx)
                self._id_map[tid] = cid

            cid = self._id_map[tid]
            self._hits[cid]      = self._hits.get(cid, 0) + 1
            self._last_seen[cid] = frame_idx
            self._last_box[cid]  = box

            # Exponential moving average on box coords to reduce jitter
            alpha = 0.6
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

        for cid in list(self._entry.keys()):
            age_unseen = frame_idx - self._last_seen.get(cid, frame_idx)
            hits       = self._hits.get(cid, 0)

            if cid in seen_cids:
                if hits < STABLE_FRAMES:
                    continue  # not yet stable, skip (anti-flicker)
            else:
                if age_unseen > GRACE_FRAMES:
                    continue  # outside grace window, skip
                # still in grace: don't increment hits

            wait_sec = (frame_idx - self._entry[cid]) / self._fps
            box      = self._smooth_box.get(cid, self._last_box.get(cid))
            if box:
                output.append((box, cid, wait_sec))

        return output

    def active_count(self, frame_idx):
        """Number of stable, currently-visible tracks."""
        count = 0
        for cid, last in self._last_seen.items():
            if frame_idx - last <= GRACE_FRAMES and self._hits.get(cid, 0) >= STABLE_FRAMES:
                count += 1
        return count

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
BOX_COLOR      = (0, 255, 120)   # green — active client detection
EMPLOYEE_COLOR = (255, 100, 50)  # blue  — employee (drawn, not counted)
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


# ── drawing helpers ───────────────────────────────────────────────────────────
def draw_box(frame, x1, y1, x2, y2, track_id=None, wait_sec=None, color=None, label_suffix=""):
    c = color or BOX_COLOR

    cv2.rectangle(frame, (x1, y1), (x2, y2), c, 2)

    if track_id is not None:
        wait_str = f" {int(wait_sec//60):02d}:{int(wait_sec%60):02d}" if wait_sec is not None else ""
        label = f"#{track_id}{wait_str}{label_suffix}"
    else:
        label = label_suffix.strip() or "?"
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)

    cv2.rectangle(frame, (x1, y1 - th - 8), (x1 + tw + 6, y1), c, -1)
    cv2.putText(frame, label, (x1 + 3, y1 - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, OVERLAY_BG, 1, cv2.LINE_AA)


def draw_hud(frame, client_count, employee_count, fps, avg_wait_sec=None):
    if avg_wait_sec is not None:
        avg_str = f"{int(avg_wait_sec//60):02d}:{int(avg_wait_sec%60):02d}"
    else:
        avg_str = "--:--"
    lines = [
        f"Clients:   {client_count}",
        f"Employees: {employee_count}",
        f"Avg wait:  {avg_str}",
        f"FPS: {fps:.1f}",
    ]

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
def run(video_path: str, output_path: str, preview: bool = True):
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

    print(f"[INFO] Video: {total_frames} frames @ {video_fps:.1f} fps")
    print(f"[INFO] Writing output to: {output_path}\n")

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out    = cv2.VideoWriter(output_path, fourcc, video_fps, (width, height))
    if not out.isOpened():
        cap.release()
        raise SystemExit(f"[ERROR] Cannot create output video: {output_path}")

    frame_idx      = 0
    visible        = []
    client_count   = 0
    employee_count = 0
    employee_boxes = None
    preview_enabled = preview
    delay_ms        = max(1, int(1000 / max(video_fps, 1)))

    if preview_enabled:
        has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
        if not has_display:
            print("[WARN] No display detected — running headless.")
            preview_enabled = False
        else:
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

        # ── detection (every other frame) ─────────────────────────────────────
        if frame_idx % 2 == 0:
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

            raw_tracks = tracker.update(client_boxes, frame) if client_boxes is not None and len(client_boxes) > 0 else []
            visible        = registry.update(raw_tracks, frame_idx)
            client_count   = len(visible)
            employee_count = 0 if employee_boxes is None else len(employee_boxes)

        # ── draw active client tracks ──────────────────────────────────────────
        for (x1, y1, x2, y2), cid, wait_sec in visible:
            draw_box(frame, x1, y1, x2, y2,
                     track_id=cid, wait_sec=wait_sec, color=BOX_COLOR)

        # ── draw employees (visible but not counted) ───────────────────────────
        if employee_boxes is not None:
            for box in employee_boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                draw_box(frame, x1, y1, x2, y2,
                         color=EMPLOYEE_COLOR, label_suffix="emp")

        # ── HUD ───────────────────────────────────────────────────────────────
        avg_wait = sum(w for _, _, w in visible) / len(visible) if visible else None
        draw_hud(frame, client_count, employee_count, video_fps, avg_wait)

        out.write(frame)

        if preview_enabled:
            cv2.imshow("Queue Detector", frame)
            if cv2.waitKey(delay_ms) & 0xFF == ord("q"):
                print("[INFO] Stopped by user.")
                break

        if frame_idx % 30 == 0:
            elapsed = time.time() - t_start
            print(  f"[{elapsed:.1f}]s "
                    f"Frames:{frame_idx}/{total_frames} "
                    f"| Clients: {client_count}  Employees: {employee_count}")

    cap.release()
    out.release()
    if preview_enabled:
        cv2.destroyAllWindows()

    total = time.time() - t_start
    print(f"[INFO] Done in {int(total//60)}m {int(total%60)}s.")


# ── entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="YOLOv8 queue detector")
    parser.add_argument("--video",      required=True)
    parser.add_argument("--output",     default="output.mp4")
    parser.add_argument("--conf",       type=float, default=CONF_THRESH)
    parser.add_argument("--model",      default=MODEL_NAME)
    parser.add_argument("--no-preview", action="store_true")
    args = parser.parse_args()

    CONF_THRESH = args.conf
    MODEL_NAME  = args.model

    run(args.video, args.output, preview=not args.no_preview)
