"""Convert DefectDet COCO annotations to YOLO format with train/val split."""
import sys
import json
import shutil
import random
from pathlib import Path
from collections import defaultdict

random.seed(42)

base = Path(__file__).resolve().parent.parent / "datasets" / "DefectDet" / "DefectDet"
ann_file = base / "annotation" / "defect_annotations.json"
img_dir = base / "images"

if not ann_file.exists():
    print(f"[ERROR] Annotation file not found: {ann_file}")
    print("Please download the DefectDet dataset first and place it under datasets/DefectDet/")
    sys.exit(1)

with open(ann_file, "r") as f:
    coco = json.load(f)

out_base = Path(__file__).resolve().parent.parent.parent / "datasets" / "DefectDet"
for split in ["train", "val"]:
    (out_base / "images" / split).mkdir(parents=True, exist_ok=True)
    (out_base / "labels" / split).mkdir(parents=True, exist_ok=True)

img_map = {img["id"]: img for img in coco["images"]}

ann_map = defaultdict(list)
for ann in coco["annotations"]:
    ann_map[ann["image_id"]].append(ann)

img_ids = list(img_map.keys())
random.shuffle(img_ids)
split_idx = int(len(img_ids) * 0.8)
train_ids = set(img_ids[:split_idx])
val_ids = set(img_ids[split_idx:])

print(f"Total images: {len(img_ids)}")
print(f"Train: {len(train_ids)}, Val: {len(val_ids)}")
print(f"Categories: {coco['categories']}")

total_anns = 0
for img_id, img_info in img_map.items():
    split = "train" if img_id in train_ids else "val"
    fname = img_info["file_name"]
    w_img = img_info["width"]
    h_img = img_info["height"]

    src = img_dir / fname
    dst = out_base / "images" / split / fname
    if src.exists():
        shutil.copy2(src, dst)

    label_name = Path(fname).stem + ".txt"
    label_path = out_base / "labels" / split / label_name

    anns = ann_map.get(img_id, [])
    with open(label_path, "w") as f:
        for ann in anns:
            cat_id = ann["category_id"] - 1
            bx, by, bw, bh = ann["bbox"]
            cx = (bx + bw / 2) / w_img
            cy = (by + bh / 2) / h_img
            nw = bw / w_img
            nh = bh / h_img
            f.write(f"{cat_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
            total_anns += 1

print(f"Total annotations converted: {total_anns}")
print(f"Output: {out_base}")
