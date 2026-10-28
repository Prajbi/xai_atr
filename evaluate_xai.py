#!/usr/bin/env python3
"""
Evaluate the four XAI methods used by 4.py on a reproducible sample of test images.

Run from the project root:
    python evaluate_xai.py

This script does not modify 4.py or the model weights. It reuses the XAI scoring
and anchor-matching logic from 4.py, adapted into a standalone evaluation script.
"""
import csv
import random
import time
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM, XGradCAM

# ---------------- Configuration ----------------
PROJECT_ROOT = Path("/home/admin1/Abishek-Intern/new_atr")
MODEL_PATH = PROJECT_ROOT / "models/yolov8s_832_79class_no_multiscale/weights/best.pt"
TEST_IMAGES = PROJECT_ROOT / "data/yolo_79class_balanced_aug/images/test"
MAX_IMAGES = 50
SEED = 42
IMGSZ = 832
CONFIDENCE = 0.25
GRAY = 114 / 255.0
W_LOC = 0.5
EIGEN_LAYERS = [-2, -3, -4]
GRAD_LAYERS = [-8, -9, -10]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# ---------------- Device / model ----------------
if not torch.cuda.is_available():
    raise SystemExit("CUDA GPU is not available. This evaluation follows 4.py's GPU-only setup.")

DEVICE = 0
CUDA = torch.device("cuda:0")
torch.backends.cudnn.benchmark = True

if not MODEL_PATH.is_file():
    raise SystemExit(f"Model weights not found: {MODEL_PATH}")
if not TEST_IMAGES.is_dir():
    raise SystemExit(f"Test image directory not found: {TEST_IMAGES}")

print(f"Loading model: {MODEL_PATH}")
model = YOLO(str(MODEL_PATH))
model_names = model.names

# ---------------- Core functions from 4.py ----------------
CAM_REGISTRY = {"EigenCAM": EigenCAM, "XGradCAM": XGradCAM}
ALL_METHODS = ["EigenCAM", "XGradCAM", "SmoothGrad", "Occlusion"]


class YoloCamWrapper(nn.Module):
    def __init__(self, yolo_model):
        super().__init__()
        self.model = yolo_model.model

    def forward(self, x):
        out = self.model(x)
        return out[0] if isinstance(out, (tuple, list)) else out


wrapped = YoloCamWrapper(model).to(CUDA)
wrapped.eval()


class AnchorClassTarget:
    """Class score for one internal prediction candidate."""
    def __init__(self, cls_id, anchor_idx):
        self.cls_id = cls_id
        self.anchor_idx = anchor_idx

    def __call__(self, model_output):
        return model_output[4 + self.cls_id, self.anchor_idx]


def score_fn(out, cls_id, idxs):
    """Maximum class score among the internal candidates matched to a detection."""
    return out[:, 4 + cls_id, :][:, idxs].max(dim=1).values


def prepare_input(image_rgb, imgsz=IMGSZ):
    h, w = image_rgb.shape[:2]
    scale = imgsz / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    pad_top, pad_left = (imgsz - new_h) // 2, (imgsz - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    tensor = (
        torch.from_numpy(canvas.astype(np.float32) / 255.0)
        .permute(2, 0, 1).unsqueeze(0).to(CUDA)
    )
    return tensor, resized, pad_top, pad_left, new_h, new_w, scale


@torch.no_grad()
def match_anchors(tensor, cls_id, box_lb, iou_thr=0.5):
    out = wrapped(tensor)[0]
    cx, cy, bw, bh = out[0], out[1], out[2], out[3]
    boxes = torch.stack([cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2], 1)
    b = torch.tensor(box_lb, device=out.device, dtype=out.dtype)
    iw = (torch.minimum(boxes[:, 2], b[2]) - torch.maximum(boxes[:, 0], b[0])).clamp(min=0)
    ih = (torch.minimum(boxes[:, 3], b[3]) - torch.maximum(boxes[:, 1], b[1])).clamp(min=0)
    inter = iw * ih
    union = (
        (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
        + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9
    )
    iou = inter / union
    candidates = torch.nonzero(iou >= iou_thr).squeeze(1)
    if candidates.numel() == 0:
        candidates = torch.topk(iou, min(5, iou.numel())).indices
    cls_scores = out[4 + cls_id]
    best_idx = int(candidates[cls_scores[candidates].argmax()])
    return best_idx, candidates, float(cls_scores[best_idx])


def norm01(m, clip_pct=None):
    m = np.nan_to_num(m.astype(np.float32))
    if clip_pct:
        m = np.clip(m, 0, np.percentile(m, clip_pct))
    m = m - m.min()
    mx = m.max()
    return m / mx if mx > 1e-8 else m


def run_cam(method, layer, tensor, target):
    cam_cls = CAM_REGISTRY[method]
    x = tensor.clone().detach().requires_grad_(True)
    layers = [wrapped.model.model[layer]]
    cam_obj = cam_cls(model=wrapped, target_layers=layers)
    with cam_obj as cam:
        if method == "EigenCAM":
            result = cam(input_tensor=x, targets=None, eigen_smooth=True)
        else:
            result = cam(input_tensor=x, targets=[target])
    return result[0]


def input_grad(x, target):
    x = x.clone().detach().requires_grad_(True)
    wrapped.zero_grad(set_to_none=True)
    with torch.enable_grad():
        out = wrapped(x)[0]
        target(out).backward()
    return x.grad.detach()


def smoothgrad_map(tensor, target, n=16, sigma=0.08):
    acc = torch.zeros_like(tensor)
    for _ in range(n):
        acc += input_grad(tensor + sigma * torch.randn_like(tensor), target).abs()
    return acc.amax(1)[0].cpu().numpy()


@torch.no_grad()
def occlusion_map(tensor, cls_id, idxs, base, patch=96, stride=48, batch_size=16):
    height, width = tensor.shape[-2:]
    ys = list(range(0, height - patch + 1, stride))
    xs = list(range(0, width - patch + 1, stride))
    if ys[-1] != height - patch:
        ys.append(height - patch)
    if xs[-1] != width - patch:
        xs.append(width - patch)
    positions = [(y, x) for y in ys for x in xs]
    heat = np.zeros((height, width), np.float32)
    count = np.zeros((height, width), np.float32)

    for start in range(0, len(positions), batch_size):
        chunk = positions[start:start + batch_size]
        batch = tensor.repeat(len(chunk), 1, 1, 1)
        for k, (y, x) in enumerate(chunk):
            batch[k, :, y:y + patch, x:x + patch] = GRAY
        scores = score_fn(wrapped(batch), cls_id, idxs).float().cpu().numpy()
        for (y, x), score in zip(chunk, scores):
            heat[y:y + patch, x:x + patch] += max(base - float(score), 0.0)
            count[y:y + patch, x:x + patch] += 1
    return heat / np.maximum(count, 1)


def compute_heat(method, layer, tensor, target, cls_id, idxs, base):
    if method in CAM_REGISTRY:
        return norm01(run_cam(method, layer, tensor, target))
    if method == "SmoothGrad":
        raw = smoothgrad_map(tensor, target)
        return norm01(cv2.GaussianBlur(raw, (0, 0), 6), 99.5)
    if method == "Occlusion":
        raw = occlusion_map(tensor, cls_id, idxs, base)
        return norm01(cv2.GaussianBlur(raw, (0, 0), 12))
    raise ValueError(method)


def localization(heat, box_lb):
    height, width = heat.shape
    x1, y1, x2, y2 = [int(round(v)) for v in box_lb]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(width, x2), min(height, y2)
    total = heat.sum()
    if total <= 1e-8 or x2 <= x1 or y2 <= y1:
        return 0.0, 0
    inside = heat[y1:y2, x1:x2].sum() / total
    area = (x2 - x1) * (y2 - y1) / (height * width)
    loc = float(np.clip((inside - area) / (1 - area + 1e-9), 0, 1))
    iy, ix = np.unravel_index(np.argmax(heat), heat.shape)
    return loc, int(x1 <= ix < x2 and y1 <= iy < y2)


@torch.no_grad()
def faithfulness(tensor, heat, cls_id, idxs, base, fracs=(0.05, 0.10, 0.20, 0.30)):
    if base <= 1e-6:
        return 0.0
    height, width = heat.shape
    order = np.argsort(-heat.ravel(), kind="stable")
    gray = torch.full_like(tensor, GRAY)
    deleted, inserted = [], []
    for fraction in fracs:
        mask = np.zeros(height * width, np.float32)
        mask[order[:max(1, int(fraction * height * width))]] = 1
        mask_t = torch.from_numpy(mask.reshape(1, 1, height, width)).to(tensor.device)
        deleted.append(tensor * (1 - mask_t) + gray * mask_t)
        inserted.append(tensor * mask_t + gray * (1 - mask_t))
    scores = (
        score_fn(wrapped(torch.cat(deleted + inserted, 0)), cls_id, idxs)
        .float().cpu().numpy() / base
    )
    scores = np.clip(scores, 0, 1)
    n = len(fracs)
    return float(np.clip(scores[n:].mean() - scores[:n].mean(), 0, 1))


def explain_one(image_rgb, cls_id, box_xyxy):
    tensor, resized, pad_top, pad_left, new_h, new_w, scale = prepare_input(image_rgb)
    x1, y1, x2, y2 = box_xyxy
    # Map all four original-image coordinates onto the padded 832x832 canvas.
    # Both y coordinates must include pad_top.
    box_lb = (
        x1 * scale + pad_left,
        y1 * scale + pad_top,
        x2 * scale + pad_left,
        y2 * scale + pad_top,
    )
    anchor_idx, candidate_idxs, anchor_score = match_anchors(tensor, cls_id, box_lb)
    target = AnchorClassTarget(cls_id, anchor_idx)
    with torch.no_grad():
        base = float(score_fn(wrapped(tensor), cls_id, candidate_idxs)[0])

    method_rows = []
    for method in ALL_METHODS:
        candidate_layers = EIGEN_LAYERS if method == "EigenCAM" else (
            GRAD_LAYERS if method == "XGradCAM" else [None]
        )
        successes = []
        errors = []
        for layer in candidate_layers:
            started = time.time()
            try:
                heat = compute_heat(
                    method, layer, tensor, target, cls_id, candidate_idxs, base
                )
                loc, hit = localization(heat, box_lb)
                faith = faithfulness(tensor, heat, cls_id, candidate_idxs, base)
                combined = W_LOC * loc + (1 - W_LOC) * faith
                successes.append({
                    "method": method,
                    "layer": layer,
                    "localization": loc,
                    "peak_inside_box": hit,
                    "faithfulness": faith,
                    "combined_score": combined,
                    "seconds": time.time() - started,
                    "error": "",
                })
            except Exception as exc:
                errors.append(f"{method} layer {layer}: {type(exc).__name__}: {exc}")
            finally:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
        if successes:
            method_rows.append(max(successes, key=lambda row: row["combined_score"]))
        else:
            method_rows.append({
                "method": method, "layer": None, "localization": None,
                "peak_inside_box": None, "faithfulness": None,
                "combined_score": None, "seconds": None,
                "error": " | ".join(errors)[:1000],
            })
    valid = [
        r for r in method_rows
        if r["combined_score"] is not None
        and np.isfinite(r["combined_score"])
    ]

    # Never award a win for an all-zero result. If multiple methods tie for
    # the highest positive score, record the tie instead of favoring list order.
    winner = ""
    tied_methods = set()
    if valid:
        best_score = max(float(r["combined_score"]) for r in valid)
        if best_score > 0:
            tied_methods = {
                r["method"] for r in valid
                if np.isclose(float(r["combined_score"]), best_score, rtol=0.0, atol=1e-8)
            }
            if len(tied_methods) == 1:
                winner = next(iter(tied_methods))

    for row in method_rows:
        if winner and row["method"] == winner:
            status = "winner"
        elif tied_methods and len(tied_methods) > 1 and row["method"] in tied_methods:
            status = "tied_best"
        elif not valid:
            status = "all_methods_failed"
        elif max(float(r["combined_score"]) for r in valid) <= 0:
            status = "no_positive_score"
        else:
            status = "not_selected"

        row["selected_for_detection"] = int(bool(winner) and row["method"] == winner)
        row["selection_status"] = status
        row["base_matched_class_score"] = base
        row["matched_anchor_score"] = anchor_score
    return method_rows


def main():
    image_paths = sorted(
        p for p in TEST_IMAGES.iterdir()
        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS
    )
    if not image_paths:
        raise SystemExit(f"No images found in {TEST_IMAGES}")

    rng = random.Random(SEED)
    selected = sorted(rng.sample(image_paths, min(MAX_IMAGES, len(image_paths))))
    run_dir = PROJECT_ROOT / "runs" / f"xai_eval_{len(selected)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=False)

    print(f"Test images available: {len(image_paths)}")
    print(f"Selected for XAI evaluation: {len(selected)} (seed={SEED})")
    print(f"Output directory: {run_dir}")
    print("Evaluating every YOLO detection at confidence >= ", CONFIDENCE)

    all_rows = []
    detection_count = 0
    for image_no, path in enumerate(selected, start=1):
        print(f"\n[{image_no}/{len(selected)}] {path.name}")
        try:
            with Image.open(path) as im:
                image = im.convert("RGB")
                image_rgb = np.array(image)
            predictions = model.predict(
                source=image, conf=CONFIDENCE, imgsz=IMGSZ,
                device=DEVICE, verbose=False
            )
            result = predictions[0]
            if result.boxes is None or len(result.boxes) == 0:
                print("  No detections.")
                continue

            boxes = result.boxes
            for det_no in range(len(boxes)):
                cls_id = int(boxes.cls[det_no].item())
                confidence = float(boxes.conf[det_no].item())
                box = tuple(float(v) for v in boxes.xyxy[det_no].tolist())
                aircraft = model_names[cls_id]
                detection_count += 1
                print(f"  Detection {det_no + 1}: {aircraft} ({confidence:.3f})")
                try:
                    rows = explain_one(image_rgb, cls_id, box)
                    for row in rows:
                        all_rows.append({
                            "image": path.name,
                            "detection_index_in_image": det_no,
                            "aircraft": aircraft,
                            "class_id": cls_id,
                            "detection_confidence": confidence,
                            **row,
                        })
                    winners = [r["method"] for r in rows if r["selected_for_detection"]]
                    print("    Winner:", winners[0] if winners else "none (all methods failed)")
                except Exception as exc:
                    error = f"{type(exc).__name__}: {exc}"
                    print("    Detection evaluation failed:", error)
                    for method in ALL_METHODS:
                        all_rows.append({
                            "image": path.name,
                            "detection_index_in_image": det_no,
                            "aircraft": aircraft,
                            "class_id": cls_id,
                            "detection_confidence": confidence,
                            "method": method,
                            "layer": None,
                            "localization": None,
                            "peak_inside_box": None,
                            "faithfulness": None,
                            "combined_score": None,
                            "seconds": None,
                            "error": error,
                            "selected_for_detection": 0,
                            "selection_status": "evaluation_error",
                            "base_matched_class_score": None,
                            "matched_anchor_score": None,
                        })
        except Exception as exc:
            print(f"  IMAGE ERROR: {type(exc).__name__}: {exc}")

    detail_path = run_dir / "xai_per_detection.csv"
    fieldnames = [
        "image", "detection_index_in_image", "aircraft", "class_id",
        "detection_confidence", "method", "layer", "localization",
        "peak_inside_box", "faithfulness", "combined_score", "seconds",
        "error", "selected_for_detection", "selection_status",
        "base_matched_class_score", "matched_anchor_score",
    ]
    with detail_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_rows)

    summary_path = run_dir / "xai_method_summary.csv"
    summary_rows = []
    evaluated_detections = {
        (r["image"], r["detection_index_in_image"])
        for r in all_rows
    }
    valid_rows = [
        r for r in all_rows
        if r.get("combined_score") is not None and r.get("error", "") == ""
    ]
    winner_counts = {}
    for r in all_rows:
        if r.get("selected_for_detection") == 1:
            winner_counts[r["method"]] = winner_counts.get(r["method"], 0) + 1

    detections_with_unique_winner = sum(
        1 for key in evaluated_detections
        if any(
            r["image"] == key[0]
            and r["detection_index_in_image"] == key[1]
            and r.get("selected_for_detection") == 1
            for r in all_rows
        )
    )
    detections_with_valid_method = sum(
        1 for key in evaluated_detections
        if any(
            r["image"] == key[0]
            and r["detection_index_in_image"] == key[1]
            and r.get("combined_score") is not None
            and r.get("error", "") == ""
            for r in all_rows
        )
    )
    for method in ALL_METHODS:
        rows = [r for r in valid_rows if r["method"] == method]
        wins = winner_counts.get(method, 0)
        summary_rows.append({
            "method": method,
            "successful_explanations": len(rows),
            "failed_explanations": sum(
                1 for r in all_rows
                if r["method"] == method and r.get("error", "")
            ),
            "mean_localization": float(np.mean([r["localization"] for r in rows])) if rows else "",
            "mean_faithfulness": float(np.mean([r["faithfulness"] for r in rows])) if rows else "",
            "mean_combined_score": float(np.mean([r["combined_score"] for r in rows])) if rows else "",
            "selected_detections": wins,
            "selection_frequency_percent": (
                100 * wins / detections_with_valid_method
                if detections_with_valid_method else 0.0
            ),
        })

    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
        writer.writeheader()
        writer.writerows(summary_rows)

    print("\n================ SUMMARY ================")
    print("Images selected:", len(selected))
    print("YOLO detections processed:", detection_count)
    print("Detection entries in XAI CSV:", len(evaluated_detections))
    print("Detections with at least one valid method:", detections_with_valid_method)
    print("Detections with a unique positive-score winner:", detections_with_unique_winner)
    print("Detections without a unique positive-score winner:",
          detections_with_valid_method - detections_with_unique_winner)
    print("\nMethod summary:")
    for row in summary_rows:
        print(
            f"{row['method']:10s} | "
            f"mean loc={row['mean_localization']} | "
            f"mean faith={row['mean_faithfulness']} | "
            f"mean combined={row['mean_combined_score']} | "
            f"wins={row['selected_detections']} | "
            f"selection={row['selection_frequency_percent']:.1f}% | "
            f"failures={row['failed_explanations']}"
        )
    print("\nSaved detailed results:", detail_path)
    print("Saved summary:", summary_path)
    print("\nInterpretation notes:")
    print("- Box coordinates are mapped to the padded model canvas, including top padding for both y coordinates.")
    print("- A winner is assigned only for a unique, strictly positive highest combined score.")
    print("- Ties and all-zero scores are not counted as wins.")
    print("- Localization is measured against the predicted box; faithfulness uses matched internal class scores.")
    print("- These are proxy metrics, not human-grounded explanation labels.")
    print("- Inspect matched-anchor scores and validate anchor alignment before making strong faithfulness claims.")


if __name__ == "__main__":
    main()