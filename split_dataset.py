import os
import shutil
import random
from pathlib import Path

# Config
SOURCE = Path("datasettopview")
OUTPUT = Path("datasettopview")
TRAIN_RATIO = 0.8
SEED = 42

random.seed(SEED)

# Gather all image stems that have a matching label
images_dir = SOURCE / "images"
labels_dir = SOURCE / "labels"

stems = [
    f.stem for f in images_dir.iterdir()
    if f.is_file() and (labels_dir / (f.stem + ".txt")).exists()
]

random.shuffle(stems)
split = int(len(stems) * TRAIN_RATIO)
train_stems = stems[:split]
val_stems = stems[split:]

print(f"Total: {len(stems)} | Train: {len(train_stems)} | Val: {len(val_stems)}")

# Create output dirs
for split_name in ("train", "val"):
    (OUTPUT / split_name / "images").mkdir(parents=True, exist_ok=True)
    (OUTPUT / split_name / "labels").mkdir(parents=True, exist_ok=True)

def copy_pair(stem, split_name):
    img_src = next(images_dir.glob(f"{stem}.*"))
    lbl_src = labels_dir / (stem + ".txt")
    shutil.copy2(img_src, OUTPUT / split_name / "images" / img_src.name)
    shutil.copy2(lbl_src, OUTPUT / split_name / "labels" / lbl_src.name)

for stem in train_stems:
    copy_pair(stem, "train")
for stem in val_stems:
    copy_pair(stem, "val")

# Copy classes.txt
if SOURCE != OUTPUT:
    shutil.copy2(SOURCE / "classes.txt", OUTPUT / "classes.txt")

# Write data.yaml
classes = [l.strip() for l in (SOURCE / "classes.txt").read_text().splitlines() if l.strip()]
yaml_content = f"""path: {OUTPUT.resolve()}
train: train/images
val: val/images

nc: {len(classes)}
names: {classes}
"""
(OUTPUT / "data.yaml").write_text(yaml_content)

print("Done! Dataset written to:", OUTPUT.resolve())
print(f"  train/images: {len(list((OUTPUT / 'train' / 'images').iterdir()))} files")
print(f"  val/images:   {len(list((OUTPUT / 'val' / 'images').iterdir()))} files")
