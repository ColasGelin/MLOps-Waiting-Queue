#!/usr/bin/env python3
"""Extract one frame every X seconds from a video."""

import argparse
import os
import subprocess
import sys


def extract_with_ffmpeg(video_path: str, interval: float, output_dir: str) -> None:
    os.makedirs(output_dir, exist_ok=True)
    cmd = [
        "ffmpeg", "-i", video_path,
        "-vf", f"fps=1/{interval}",
        "-q:v", "2",
        os.path.join(output_dir, "frame_%04d.jpg"),
    ]
    subprocess.run(cmd, check=True)


def extract_with_opencv(video_path: str, interval: float, output_dir: str) -> None:
    import cv2

    os.makedirs(output_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        sys.exit(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_interval = int(fps * interval)
    frame_index = 0
    saved = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_index % frame_interval == 0:
            out_path = os.path.join(output_dir, f"frame_{saved:04d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1
        frame_index += 1

    cap.release()
    print(f"Saved {saved} frames to {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract one frame every X seconds from a video.")
    parser.add_argument("videos", nargs="+", help="Path(s) to input video file(s)")
    parser.add_argument("interval", type=float, help="Interval in seconds between frames")
    parser.add_argument("-o", "--output", default="frames", help="Output directory (default: frames)")
    args = parser.parse_args()

    use_ffmpeg = subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode == 0
    print("Using ffmpeg" if use_ffmpeg else "ffmpeg not found, falling back to opencv")

    for video in args.videos:
        if not os.path.isfile(video):
            print(f"Skipping: file not found: {video}", file=sys.stderr)
            continue

        # Each video gets its own subdirectory named after the video file
        video_name = os.path.splitext(os.path.basename(video))[0]
        out_dir = os.path.join(args.output, video_name) if len(args.videos) > 1 else args.output

        print(f"Processing {video} -> {out_dir}")
        if use_ffmpeg:
            extract_with_ffmpeg(video, args.interval, out_dir)
        else:
            extract_with_opencv(video, args.interval, out_dir)


if __name__ == "__main__":
    main()
