import random
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
from PIL import Image
from ultralytics import YOLO

ROOT = Path("/home/admin1/Abishek-Intern/new_atr")
MODEL_PATH = ROOT / "models/yolov8s_832_79class_no_multiscale/weights/best.pt"
IMAGE_DIR = ROOT / "data/yolo_79class_balanced_aug/images/test"
IMGSZ = 832
CONF = 0.25
DEVICE = 0

if not torch.cuda.is_available():
    raise SystemExit("CUDA is unavailable.")

model = YOLO(str(MODEL_PATH))
net = model.model.eval()
device = torch.device("cuda:0")
names = model.names

paths = sorted(
    p for p in IMAGE_DIR.iterdir()
    if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
)
random.Random(42).shuffle(paths)
paths = paths[:50]

def prepare(rgb):
    h, w = rgb.shape[:2]
    scale = IMGSZ / max(h, w)
    nh, nw = round(h * scale), round(w * scale)
    resized = cv2.resize(rgb, (nw, nh), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)
    top = (IMGSZ - nh) // 2
    left = (IMGSZ - nw) // 2
    canvas[top:top + nh, left:left + nw] = resized
    tensor = torch.from_numpy(canvas).permute(2, 0, 1).float()
    tensor = (tensor / 255.0).unsqueeze(0).to(device)
    return tensor, scale, top, left

def get_raw(tensor):
    with torch.no_grad():
        out = net(tensor)
    if isinstance(out, (tuple, list)):
        out = out[0]
    return out[0].float()

def calculate_iou(boxes, box):
    x1 = torch.maximum(boxes[:, 0], box[0])
    y1 = torch.maximum(boxes[:, 1], box[1])
    x2 = torch.minimum(boxes[:, 2], box[2])
    y2 = torch.minimum(boxes[:, 3], box[3])
    inter = (x2 - x1).clamp(min=0) * (y2 - y1).clamp(min=0)
    area1 = (boxes[:, 2] - boxes[:, 0]).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=0)
    area2 = (box[2] - box[0]).clamp(min=0) * (
        box[3] - box[1]
    ).clamp(min=0)
    return inter / (area1 + area2 - inter + 1e-9)

rows = []

for image_no, path in enumerate(paths, 1):
    with Image.open(path) as im:
        rgb = np.array(im.convert("RGB"))

    # Force square inference so it can be compared with the 832x832 tensor.
    result = model.predict(
        source=rgb, conf=CONF, imgsz=IMGSZ,
        rect=False, device=DEVICE, verbose=False
    )[0]

    if result.boxes is None or len(result.boxes) == 0:
        continue

    tensor, scale, top, left = prepare(rgb)
    raw = get_raw(tensor)

    cx, cy, bw, bh = raw[:4]
    raw_boxes = torch.stack(
        [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], dim=1
    )
    nc = len(names)

    for i in range(len(result.boxes)):
        cls = int(result.boxes.cls[i].item())
        conf = float(result.boxes.conf[i].item())
        x1, y1, x2, y2 = result.boxes.xyxy[i].tolist()

        box = torch.tensor(
            [x1 * scale + left, y1 * scale + top,
             x2 * scale + left, y2 * scale + top],
            device=device, dtype=raw.dtype
        )

        ious = calculate_iou(raw_boxes, box)
        scores = raw[4 + cls]
        best_iou_idx = int(ious.argmax())

        candidates = torch.where(ious >= 0.5)[0]
        fallback = len(candidates) == 0
        if fallback:
            candidates = torch.topk(ious, min(5, len(ious))).indices

        chosen = int(candidates[scores[candidates].argmax()])

        rows.append({
            "image": path.name,
            "aircraft": names[cls],
            "det_conf": conf,
            "max_box_iou": float(ious[best_iou_idx]),
            "chosen_anchor_iou": float(ious[chosen]),
            "chosen_class_score": float(scores[chosen]),
            "max_class_score": float(scores.max()),
            "fallback_top5": fallback,
        })

    print(f"Processed {image_no}/{len(paths)}: {path.name}")

df = pd.DataFrame(rows)
if df.empty:
    raise SystemExit("No detections found in the selected sample.")

out = ROOT / "anchor_alignment_diagnostic.csv"
df.to_csv(out, index=False)

print("\n========== ANCHOR ALIGNMENT SUMMARY ==========")
print("Detections:", len(df))
print("Mean highest box IoU:", round(df.max_box_iou.mean(), 4))
print("Mean selected-anchor IoU:", round(df.chosen_anchor_iou.mean(), 4))
print("Selected anchor IoU < 0.5:", int((df.chosen_anchor_iou < 0.5).sum()))
print("Used top-5 fallback:", int(df.fallback_top5.sum()))
print("Mean detection confidence:", round(df.det_conf.mean(), 4))
print("Mean selected class score:", round(df.chosen_class_score.mean(), 4))
print("Mean maximum class score:", round(df.max_class_score.mean(), 4))

print("\nLargest confidence vs selected-score differences:")
df["score_difference"] = abs(df.det_conf - df.chosen_class_score)
print(df.nlargest(15, "score_difference")[
    ["image", "aircraft", "det_conf", "max_box_iou",
     "chosen_anchor_iou", "chosen_class_score",
     "max_class_score", "fallback_top5"]
].to_string(index=False))

print("\nFull diagnostic saved to:", out)
