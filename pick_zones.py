"""
Zone Picker — extract a frame, open it in Windows, type the coordinates.

Usage:
    python pick_zones.py --video footage.mp4
"""

import cv2
import argparse
import os
import subprocess


def prompt_zone(label):
    print(f"\n  [{label}]")
    print("  Click the 4 corners of the zone on the grid image (any order, go clockwise).")
    print("  Enter as 8 comma-separated values: x1,y1,x2,y2,x3,y3,x4,y4")
    print("  Example: 120,80, 540,80, 560,620, 100,620")
    while True:
        raw = input("  > ").strip()
        try:
            parts = [int(v) for v in raw.replace(" ", "").split(",")]
            if len(parts) != 8:
                raise ValueError
            # Return as list of (x,y) pairs
            return [(parts[i], parts[i+1]) for i in range(0, 8, 2)]
        except ValueError:
            print("  Invalid format — please enter exactly 8 integers: x1,y1,x2,y2,x3,y3,x4,y4")


def draw_zone(frame, zone, color, label):
    if zone is None:
        return
    import numpy as np
    pts = np.array(zone, dtype="int32")
    overlay = frame.copy()
    cv2.fillPoly(overlay, [pts], color)
    cv2.addWeighted(overlay, 0.2, frame, 0.8, 0, frame)
    cv2.polylines(frame, [pts], isClosed=True, color=color, thickness=2)
    cv2.putText(frame, label, (zone[0][0] + 4, zone[0][1] + 24),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)


def draw_grid(frame, step=100):
    """Overlay a coordinate grid with labels every `step` pixels."""
    h, w = frame.shape[:2]
    grid_color  = (255, 255, 255)
    label_color = (0, 255, 255)
    shadow      = (0, 0, 0)
    font        = cv2.FONT_HERSHEY_SIMPLEX

    for x in range(0, w, step):
        cv2.line(frame, (x, 0), (x, h), grid_color, 1, cv2.LINE_AA)
        label = str(x)
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:  # shadow
            cv2.putText(frame, label, (x + 2 + dx, 16 + dy), font, 0.4, shadow, 1)
        cv2.putText(frame, label, (x + 2, 16), font, 0.4, label_color, 1, cv2.LINE_AA)

    for y in range(0, h, step):
        cv2.line(frame, (0, y), (w, y), grid_color, 1, cv2.LINE_AA)
        label = str(y)
        for dy, dx in [(-1,0),(1,0),(0,-1),(0,1)]:
            cv2.putText(frame, label, (2 + dx, y + 14 + dy), font, 0.4, shadow, 1)
        cv2.putText(frame, label, (2, y + 14), font, 0.4, label_color, 1, cv2.LINE_AA)


def open_in_windows(path):
    """Try to open an image file using the Windows shell from WSL."""
    win_path = subprocess.run(
        ["wslpath", "-w", os.path.abspath(path)],
        capture_output=True, text=True
    ).stdout.strip()
    if win_path:
        subprocess.Popen(["explorer.exe", win_path])


def main():
    parser = argparse.ArgumentParser(description="Pick queue zones from a video frame")
    parser.add_argument("--video",  required=True)
    parser.add_argument("--frame",  type=int, default=0, help="Frame index to use (default: 0)")
    parser.add_argument("--output", default="zones.txt")
    args = parser.parse_args()

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise SystemExit(f"[ERROR] Cannot open video: {args.video}")

    total  = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target = min(args.frame, total - 1)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        raise SystemExit("[ERROR] Could not read frame.")

    h, w = frame.shape[:2]

    # Save frame with coordinate grid overlay
    grid_frame = frame.copy()
    draw_grid(grid_frame)
    frame_path = "zone_frame.jpg"
    cv2.imwrite(frame_path, grid_frame)
    print(f"\n[INFO] Frame saved: {frame_path}  ({w}x{h} px)")
    print("[INFO] Attempting to open it in Windows...")
    open_in_windows(frame_path)
    print("[INFO] If it didn't open automatically, open it manually from Windows Explorer.")
    print(f"       Path: \\\\wsl.localhost\\Ubuntu{os.path.abspath(frame_path)}")
    print("\n[INFO] Hover over the corners of each queue zone to read the pixel coordinates.")
    print("       Most image viewers show coordinates in the status bar (bottom of the window).")

    zone1 = prompt_zone("Queue 1")
    zone2 = prompt_zone("Queue 2")

    # Save preview with both zones drawn
    preview = frame.copy()
    draw_zone(preview, zone1, (60, 180, 255),  "Queue 1")
    draw_zone(preview, zone2, (255, 80, 180), "Queue 2")
    preview_path = "zone_preview.jpg"
    cv2.imwrite(preview_path, preview)
    print(f"\n[INFO] Preview saved: {preview_path}")
    open_in_windows(preview_path)
    print(f"       Path: \\\\wsl.localhost\\Ubuntu{os.path.abspath(preview_path)}")

    z1_str = ",".join(f"{x},{y}" for x, y in zone1)
    z2_str = ",".join(f"{x},{y}" for x, y in zone2)
    cmd = f"python detect.py --video <your_video.mp4> --zone1 {z1_str} --zone2 {z2_str}"

    print("\n── Result ────────────────────────────────────────")
    print(f"  --zone1 {z1_str}")
    print(f"  --zone2 {z2_str}")
    print(f"\n  Full command:\n  {cmd}")

    with open(args.output, "w") as f:
        f.write(f"--zone1 {z1_str}\n--zone2 {z2_str}\n\n# Full command:\n# {cmd}\n")

    print(f"\n[INFO] Saved to {args.output}")


if __name__ == "__main__":
    main()
