"""
Fine-tune YOLOv8s on the supermarket client/employee dataset.

Usage:
    python train.py
    python train.py --epochs 150 --batch 8 --model yolov8s.pt
"""

import argparse
from ultralytics import YOLO

DATA_YAML  = "datasettopview/data.yaml"
BASE_MODEL = "yolov8s.pt"
EPOCHS     = 100
BATCH      = 8
IMG_SIZE   = 640
# Freeze the first N backbone layers — important for small datasets
# so the pretrained features are preserved and only the head adapts.
FREEZE     = 10


def train(model_path: str, epochs: int, batch: int):
    model = YOLO(model_path)

    model.train(
        data=DATA_YAML,
        epochs=epochs,
        imgsz=IMG_SIZE,
        batch=batch,
        freeze=FREEZE,

        # ── learning rate ────────────────────────────────────────────────────
        lr0=0.001,
        lrf=0.01,       # final lr = lr0 * lrf
        warmup_epochs=5,

        # ── augmentation (compensates for the small dataset) ─────────────────
        fliplr=0.5,
        degrees=10.0,
        translate=0.1,
        scale=0.3,
        mosaic=0.8,
        mixup=0.1,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,

        # ── output ───────────────────────────────────────────────────────────
        project="runs",
        name="supermarket_finetune",
        exist_ok=True,
        save=True,
        plots=False,

        # ── early stopping ───────────────────────────────────────────────────
        patience=30,

        device=0,       # GPU 0; change to "cpu" if no GPU
        workers=0,
        verbose=True,
    )

    print("\n[INFO] Training complete.")
    print("[INFO] Best weights: runs/detect/supermarket_finetune/weights/best.pt")
    print("[INFO] Run detection with:")
    print("       python detect.py --video footage.mp4 "
          "--model runs/detect/supermarket_finetune/weights/best.pt")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLOv8 for client/employee detection")
    parser.add_argument("--model",  default=BASE_MODEL)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch",  type=int, default=BATCH)
    args = parser.parse_args()

    train(args.model, args.epochs, args.batch)
