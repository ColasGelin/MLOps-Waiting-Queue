# Queue Detector — Step 1: Person Detection

Runs YOLOv8 on supermarket footage, shows a semi-live preview window
frame-by-frame, and saves an annotated output video.

---

## Setup

```bash
pip install -r requirements.txt
```

No manual model download needed — YOLOv8 downloads automatically
on first run.

---

## Run

```bash
python detect.py --video path/to/your/footage.mp4 --output output.mp4
```

Press `q` in the preview window to stop early.

### Options

| Flag | Default | Description |
|------|---------|-------------|
| `--video` | required | Path to your footage file |
| `--output` | output.mp4 | Path to output video file |
| `--conf` | 0.35 | Confidence threshold (lower = more detections) |
| `--model` | yolov8n.pt | Model size (see below) |
| `--no-preview` | false | Disable on-screen preview (headless mode) |

### Model sizes

| Model | Speed | Accuracy | Use when |
|-------|-------|----------|----------|
| `yolov8n.pt` | fastest | good | testing / low-end machine |
| `yolov8s.pt` | fast | better | default recommendation |
| `yolov8m.pt` | medium | best | final demo |

---

## What you'll see

- A preview window that updates frame-by-frame (semi-live playback)
- An output video file with green bounding boxes around each detected person
- Confidence score on each box
- Top-left HUD: person count + FPS + processing status

---

## WSL notes

The preview window needs Linux GUI support.

- On Windows 11, use WSLg (recommended).
- If you're using plain WSL2 without WSLg, run an X server (for example VcXsrv)
	and set `DISPLAY` correctly.

If no GUI display is available, the script automatically continues in headless mode
and still writes the output video.

---

## Troubleshooting

**Too many false detections** → raise `--conf` to 0.5 or 0.6

**Missing people** → lower `--conf` to 0.25, or switch to `yolov8s.pt`

**Very slow** → make sure you're on `yolov8n.pt` (fastest)

**Video won't open** → check the path, try converting to .mp4 with ffmpeg:
```bash
ffmpeg -i input.mov -c:v libx264 output.mp4
```

---

## Next steps (coming)

- Step 2: ByteTrack person tracking (assign IDs, track across frames)
- Step 3: Queue zone detection + wait time estimation
- Step 4: LangGraph agent layer
- Step 5: Grafana monitoring dashboard
