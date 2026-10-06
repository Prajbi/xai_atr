import time
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn.functional as F
from pathlib import Path
from PIL import Image
from ultralytics import YOLO
from pytorch_grad_cam import (EigenCAM, EigenGradCAM, GradCAM, GradCAMPlusPlus,
                              XGradCAM, HiResCAM, LayerCAM, ScoreCAM, AblationCAM)
from pytorch_grad_cam.utils.image import show_cam_on_image

# PAGE CONFIG
st.set_page_config(page_title="Aircraft Detection Dashboard", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")

# PROJECT PATH
PROJECT_ROOT = Path("/home/admin1/Abishek-Intern/new_atr")
MODEL_PATH = PROJECT_ROOT /"models" / "yolov8s_832_targeted_aug" / "weights" / "best.pt"
#/ "main" /"models" / "yolo_aircraft.pt"
#/"models" / "yolov8s_832_targeted_aug" / "weights" / "best.pt"
# GPU CHECK
if not torch.cuda.is_available():
    st.error("CUDA GPU is not available. This application is configured for GPU-only inference.")
    st.stop()
DEVICE = 0
CUDA = torch.device("cuda:0")
GPU_NAME = torch.cuda.get_device_name(0)
IMGSZ = 832
GRAY = 114 / 255.0

# LOAD MODEL
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found:\n\n{MODEL_PATH}")
        st.stop()
    return YOLO(str(MODEL_PATH))
model = load_model()

# METHOD REGISTRY + AUTOMATIC LAYER CANDIDATES
CAM_REGISTRY = {
    "EigenCAM": EigenCAM, "EigenGradCAM": EigenGradCAM, "GradCAM": GradCAM,
    "GradCAM++": GradCAMPlusPlus, "XGradCAM": XGradCAM, "HiResCAM": HiResCAM,
    "LayerCAM": LayerCAM, "ScoreCAM": ScoreCAM, "AblationCAM": AblationCAM,
}
OTHER_METHODS = ["Saliency", "SmoothGrad", "IntegratedGradients", "Occlusion", "RISE"]
ALL_METHODS = list(CAM_REGISTRY) + OTHER_METHODS
DEFAULT_METHODS = [m for m in ALL_METHODS if m != "AblationCAM"]   # AblationCAM often fails on YOLO routing

EIGEN_LAYERS = [-2, -3, -4]        # tried automatically for EigenCAM
GRAD_LAYERS = [-8, -9, -10]        # tried automatically for gradient CAMs
SLOW_METHODS = {"ScoreCAM", "AblationCAM"}
SLOW_LAYERS = [-9]                 # slow methods only try one layer
W_LOC = 0.5                        # fixed 50/50 balance: localization vs faithfulness

def layers_for(method):
    if method == "EigenCAM":
        return EIGEN_LAYERS
    if method in SLOW_METHODS:
        return SLOW_LAYERS
    return GRAD_LAYERS

# SIDEBAR
st.sidebar.title("Detection Settings")
confidence = st.sidebar.slider("Confidence", min_value=0.01, max_value=1.00, value=0.25, step=0.01)
st.sidebar.divider()
st.sidebar.markdown("### Model Information")
st.sidebar.info("GPU-only inference")
st.sidebar.write(f"**Model:** `{MODEL_PATH.name}`")
st.sidebar.write("**Architecture:** YOLOv8s")
if hasattr(model, "names"):
    st.sidebar.write(f"**Number of Classes:** {len(model.names)}")
st.sidebar.write(f"**GPU:** {GPU_NAME}")
st.sidebar.divider()
st.sidebar.markdown("### Explainability")
run_xai = st.sidebar.checkbox("Run explainability comparison", value=True)
selected_methods = st.sidebar.multiselect("Methods", ALL_METHODS, default=DEFAULT_METHODS)

# HEADER
st.title("Military Aircraft Detection")
st.markdown("Upload aircraft images to detect them with YOLOv8s and see exactly where the model looks.")
st.divider()

# IMG UPLOAD
st.subheader("Upload Aircraft Images")
uploaded_files = st.file_uploader("Choose one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
if not uploaded_files:
    st.info("Upload one or more aircraft images to start detection.")
    st.stop()

# =========================== XAI CORE ===========================
class YoloCamWrapper(torch.nn.Module):
    def __init__(self, yolo_model):
        super().__init__()
        self.model = yolo_model.model

    def forward(self, x):
        out = self.model(x)
        return out[0] if isinstance(out, (tuple, list)) else out   # (B, 4+nc, N)

@st.cache_resource
def get_wrapper(_yolo_model):
    w = YoloCamWrapper(_yolo_model).to(CUDA)
    w.eval()
    return w

class AnchorClassTarget:
    """Class score of ONE specific anchor = the prediction for the chosen aircraft."""
    def __init__(self, cls_id, anchor_idx):
        self.cls_id, self.anchor_idx = cls_id, anchor_idx
    def __call__(self, model_output):          # (4+nc, N)
        return model_output[4 + self.cls_id, self.anchor_idx]

def score_fn(out, cls_id, idxs):
    """Best class score among the anchors that belong to the chosen detection. (B,)"""
    return out[:, 4 + cls_id, :][:, idxs].max(dim=1).values

def prepare_input(image_rgb, imgsz=IMGSZ):
    h, w = image_rgb.shape[:2]
    scale = imgsz / max(h, w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    pad_top, pad_left = (imgsz - new_h) // 2, (imgsz - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    tensor = torch.from_numpy(canvas.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(CUDA)
    return tensor, resized, pad_top, pad_left, new_h, new_w, scale

@torch.no_grad()
def match_anchors(wrapped, tensor, cls_id, box_lb, iou_thr=0.5):
    out = wrapped(tensor)[0]
    cx, cy, w, h = out[0], out[1], out[2], out[3]
    boxes = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    b = torch.tensor(box_lb, device=out.device, dtype=out.dtype)
    iw = (torch.minimum(boxes[:, 2], b[2]) - torch.maximum(boxes[:, 0], b[0])).clamp(min=0)
    ih = (torch.minimum(boxes[:, 3], b[3]) - torch.maximum(boxes[:, 1], b[1])).clamp(min=0)
    inter = iw * ih
    iou = inter / ((boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter + 1e-9)
    cand = torch.nonzero(iou >= iou_thr).squeeze(1)
    if cand.numel() == 0:
        cand = torch.topk(iou, 5).indices
    cls_scores = out[4 + cls_id]
    best = int(cand[cls_scores[cand].argmax()])
    return best, cand, float(cls_scores[best])

def norm01(m, clip_pct=None):
    m = np.nan_to_num(m.astype(np.float32))
    if clip_pct:
        m = np.clip(m, 0, np.percentile(m, clip_pct))
    m = m - m.min()
    mx = m.max()
    return m / mx if mx > 1e-8 else m

# ---------- CAM family ----------
def run_cam(method, layer, wrapped, tensor, target):
    cam_cls = CAM_REGISTRY[method]
    x = tensor.clone().detach().requires_grad_(True)
    layers = [wrapped.model.model[layer]]
    try:
        cam_obj = cam_cls(model=wrapped, target_layers=layers,
                          **({"ratio_channels_to_ablate": 0.25} if method == "AblationCAM" else {}))
    except TypeError:
        cam_obj = cam_cls(model=wrapped, target_layers=layers)
    with cam_obj as cam:
        if method == "EigenCAM":                  # class-agnostic by design
            g = cam(input_tensor=x, targets=None, eigen_smooth=True)
        else:
            g = cam(input_tensor=x, targets=[target])
    return g[0]

# ---------- input-gradient family ----------
def _input_grad(wrapped, x, target):
    x = x.clone().detach().requires_grad_(True)
    wrapped.zero_grad(set_to_none=True)
    with torch.enable_grad():
        out = wrapped(x)[0]
        target(out).backward()
    return x.grad.detach()

def saliency_map(wrapped, tensor, target):
    return _input_grad(wrapped, tensor, target).abs().amax(1)[0].cpu().numpy()

def smoothgrad_map(wrapped, tensor, target, n=16, sigma=0.08):
    acc = torch.zeros_like(tensor)
    for _ in range(n):
        acc += _input_grad(wrapped, tensor + sigma * torch.randn_like(tensor), target).abs()
    return acc.amax(1)[0].cpu().numpy()

def ig_map(wrapped, tensor, target, steps=16):
    base = torch.full_like(tensor, GRAY)
    avg = torch.zeros_like(tensor)
    for a in np.linspace(1 / steps, 1.0, steps):
        avg += _input_grad(wrapped, base + float(a) * (tensor - base), target)
    return ((tensor - base) * avg / steps).abs().sum(1)[0].cpu().numpy()

# ---------- perturbation family ----------
@torch.no_grad()
def occlusion_map(wrapped, tensor, cls_id, idxs, base, patch=96, stride=48, bs=16):
    H = tensor.shape[-1]
    ys = list(range(0, H - patch + 1, stride))
    if ys[-1] != H - patch:
        ys.append(H - patch)
    pos = [(y, x) for y in ys for x in ys]
    heat, cnt = np.zeros((H, H), np.float32), np.zeros((H, H), np.float32)
    for i in range(0, len(pos), bs):
        chunk = pos[i:i + bs]
        batch = tensor.repeat(len(chunk), 1, 1, 1)
        for k, (y, x) in enumerate(chunk):
            batch[k, :, y:y + patch, x:x + patch] = GRAY
        sc = score_fn(wrapped(batch), cls_id, idxs).float().cpu().numpy()
        for (y, x), s in zip(chunk, sc):
            heat[y:y + patch, x:x + patch] += max(base - s, 0.0)
            cnt[y:y + patch, x:x + patch] += 1
    return heat / np.maximum(cnt, 1)

@torch.no_grad()
def rise_map(wrapped, tensor, cls_id, idxs, n=400, s=10, p=0.5, bs=20):
    H = tensor.shape[-1]
    gray = torch.full_like(tensor, GRAY)
    cell = int(np.ceil(H / s)); up = (s + 1) * cell
    heat = torch.zeros(H, H, device=tensor.device)
    rng = np.random.RandomState(0)
    for i in range(0, n, bs):
        b = min(bs, n - i)
        grid = (torch.rand(b, 1, s, s, device=tensor.device) < p).float()
        big = F.interpolate(grid, size=(up, up), mode="bilinear", align_corners=False)
        masks = torch.empty(b, 1, H, H, device=tensor.device)
        for k in range(b):
            dy, dx = rng.randint(0, cell), rng.randint(0, cell)
            masks[k] = big[k, :, dy:dy + H, dx:dx + H]
        sc = score_fn(wrapped(tensor * masks + gray * (1 - masks)), cls_id, idxs).float()
        heat += (sc.view(b, 1, 1, 1) * masks).sum(0)[0]
    return (heat / (n * p)).cpu().numpy()

def compute_heat(method, layer, wrapped, tensor, target, cls_id, idxs, base):
    if method in CAM_REGISTRY:
        return norm01(run_cam(method, layer, wrapped, tensor, target))
    if method == "Saliency":
        return norm01(cv2.GaussianBlur(saliency_map(wrapped, tensor, target), (0, 0), 6), 99.5)
    if method == "SmoothGrad":
        return norm01(cv2.GaussianBlur(smoothgrad_map(wrapped, tensor, target), (0, 0), 6), 99.5)
    if method == "IntegratedGradients":
        return norm01(cv2.GaussianBlur(ig_map(wrapped, tensor, target), (0, 0), 6), 99.5)
    if method == "Occlusion":
        return norm01(cv2.GaussianBlur(occlusion_map(wrapped, tensor, cls_id, idxs, base), (0, 0), 12))
    if method == "RISE":
        return norm01(rise_map(wrapped, tensor, cls_id, idxs))
    raise ValueError(method)

# ---------- metrics (used internally to pick the best method) ----------
def localization(heat, box_lb):
    H, W = heat.shape
    x1, y1, x2, y2 = [int(round(v)) for v in box_lb]
    x1, y1, x2, y2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    total = heat.sum()
    if total <= 1e-8 or x2 <= x1 or y2 <= y1:
        return 0.0, 0
    inside = heat[y1:y2, x1:x2].sum() / total
    area = (x2 - x1) * (y2 - y1) / (H * W)
    loc = float(np.clip((inside - area) / (1 - area + 1e-9), 0, 1))
    iy, ix = np.unravel_index(np.argmax(heat), heat.shape)
    return loc, int(x1 <= ix < x2 and y1 <= iy < y2)

@torch.no_grad()
def faithfulness(wrapped, tensor, heat, cls_id, idxs, base, fracs=(0.05, 0.10, 0.20, 0.30)):
    if base <= 1e-6:
        return 0.0
    H, W = heat.shape
    order = np.argsort(-heat.ravel(), kind="stable")
    gray = torch.full_like(tensor, GRAY)
    dels, inss = [], []
    for f in fracs:
        m = np.zeros(H * W, np.float32); m[order[:int(f * H * W)]] = 1
        m = torch.from_numpy(m.reshape(1, 1, H, W)).to(tensor.device)
        dels.append(tensor * (1 - m) + gray * m)
        inss.append(tensor * m + gray * (1 - m))
    sc = score_fn(wrapped(torch.cat(dels + inss, 0)), cls_id, idxs).float().cpu().numpy() / base
    sc = np.clip(sc, 0, 1)
    n = len(fracs)
    return float(np.clip(sc[n:].mean() - sc[:n].mean(), 0, 1))

# ---------- orchestration (cached per image + detection + methods) ----------
@st.cache_data(show_spinner=False, max_entries=64)
def explain_detection(_model, image_np, cls_id, box_xyxy, methods):
    wrapped = get_wrapper(_model)
    tensor, resized, pad_top, pad_left, new_h, new_w, scale = prepare_input(image_np)
    x1, y1, x2, y2 = box_xyxy
    box_lb = (x1 * scale + pad_left, y1 * scale + pad_top, x2 * scale + pad_left, y2 * scale + pad_top)
    anchor_idx, idxs, anchor_score = match_anchors(wrapped, tensor, cls_id, box_lb)
    target = AnchorClassTarget(cls_id, anchor_idx)
    with torch.no_grad():
        base = float(score_fn(wrapped(tensor), cls_id, idxs)[0])

    results = []
    for m in methods:
        tried, last_err = [], ""
        for L in (layers_for(m) if m in CAM_REGISTRY else [None]):
            t0 = time.time()
            try:
                heat = compute_heat(m, L, wrapped, tensor, target, cls_id, idxs, base)
                loc, hit = localization(heat, box_lb)
                faith = faithfulness(wrapped, tensor, heat, cls_id, idxs, base)
                tried.append({"name": m if L is None else f"{m} [L{L}]", "method": m, "layer": L,
                              "heat": heat[pad_top:pad_top + new_h, pad_left:pad_left + new_w].astype(np.float16),
                              "loc": loc, "hit": hit, "faith": faith,
                              "combined": W_LOC * loc + (1 - W_LOC) * faith, "time": time.time() - t0})
            except Exception as e:
                last_err = str(e)[:200]
            torch.cuda.empty_cache()
        if tried:
            results.append(max(tried, key=lambda r: r["combined"]))     # automatic best layer for this method
        else:
            results.append({"name": m, "method": m, "error": last_err})

    return {"view_rgb": resized, "box_view": (x1 * scale, y1 * scale, x2 * scale, y2 * scale),
            "base_score": base, "anchor_score": anchor_score, "results": results}

# ---------- rendering ----------
PALETTE = [(0, 255, 0), (255, 0, 255), (0, 255, 255), (255, 165, 0), (255, 255, 0), (0, 128, 255)]

def region_heat(heat16, box, margin=0.3):
    """Rescale a prediction's heat inside its own box region so each aircraft shows its own hotspot."""
    heat = heat16.astype(np.float32)
    H, W = heat.shape
    x1, y1, x2, y2 = box
    mg = margin * max(x2 - x1, y2 - y1)
    xa, ya = int(max(0, x1 - mg)), int(max(0, y1 - mg))
    xb, yb = int(min(W, x2 + mg)), int(min(H, y2 + mg))
    out = np.zeros_like(heat)
    if xb <= xa or yb <= ya:
        return out
    out[ya:yb, xa:xb] = norm01(heat[ya:yb, xa:xb])
    mask = np.zeros_like(heat)
    mask[ya:yb, xa:xb] = 1.0
    mask = cv2.GaussianBlur(mask, (0, 0), 0.08 * max(x2 - x1, y2 - y1) + 1)   # soft edges, no hard rectangle
    return out * mask

def combined_overlay(view_rgb, items, contour_thr=0.5):
    """items: list of (index, heat16, box_view). One image showing where the model looks for EVERY aircraft."""
    H, W = view_rgb.shape[:2]
    comb = np.zeros((H, W), np.float32)
    for _, heat16, box in items:
        comb = np.maximum(comb, region_heat(heat16, box))
    ov = np.ascontiguousarray(show_cam_on_image(view_rgb.astype(np.float32) / 255.0, comb, use_rgb=True))
    for k, (idx, heat16, box) in enumerate(items):
        color = PALETTE[k % len(PALETTE)]
        rh = region_heat(heat16, box)
        cnts, _ = cv2.findContours((rh >= contour_thr).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(ov, cnts, -1, color, 2)                       # this aircraft's hotspot outline
        x1, y1, x2, y2 = [int(v) for v in box]
        cv2.rectangle(ov, (x1, y1), (x2, y2), color, 2)
        cv2.putText(ov, f"#{idx}", (x1, max(18, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    return ov

def zoom_crop(img, box, margin=0.5, min_w=500):
    x1, y1, x2, y2 = box
    m = margin * max(x2 - x1, y2 - y1)
    h, w = img.shape[:2]
    crop = img[int(max(0, y1 - m)):int(min(h, y2 + m)), int(max(0, x1 - m)):int(min(w, x2 + m))]
    if crop.shape[1] < min_w:
        f = min_w / crop.shape[1]
        crop = cv2.resize(crop, None, fx=f, fy=f, interpolation=cv2.INTER_CUBIC)
    return crop

def ranked_ok(res):
    return sorted([r for r in res["results"] if "error" not in r], key=lambda r: r["combined"], reverse=True)

def render_image_explanations(explained, image_name):
    """explained: list of (detection_index, detection_dict, explain_result) for one image."""
    view_rgb = explained[0][2]["view_rgb"]
    box_by_idx = {i: res["box_view"] for i, _, res in explained}
    name_by_idx = {i: f"{d['Aircraft']} ({d['Confidence']:.0%})" for i, d, _ in explained}
    best_items, best_names, per_method, failures = [], [], {}, set()

    for i, d, res in explained:
        ok = ranked_ok(res)
        for r in res["results"]:
            if "error" in r:
                failures.add(f"{r['name']}: {r['error']}")
        for r in ok:
            per_method.setdefault(r["method"], {})[i] = r
        if ok:
            best_items.append((i, ok[0]["heat"], res["box_view"]))
            best_names.append(f"#{i} {d['Aircraft']}: {ok[0]['name']}")

    if not best_items:
        st.error("All explainability methods failed for this image.")
        for f in sorted(failures):
            st.caption(f)
        return

    # 1) ONE image: where the model looks for all aircraft (best method per aircraft)
    st.markdown("#### Where the model looks")
    st.image(combined_overlay(view_rgb, best_items), width=1000,
             caption="Colored box + outline = one aircraft's strongest-attention region. "
                     "Best method per aircraft: " + " | ".join(best_names))

    # 2) ZOOM on every aircraft
    st.markdown("#### Zoom on each aircraft")
    zoom_choice = st.selectbox("Zoom using", ["Best method per aircraft"] + list(per_method),
                               key=f"zoom_{image_name}")
    if zoom_choice == "Best method per aircraft":
        zoom_items = best_items
    else:
        zoom_items = [(i, r["heat"], box_by_idx[i]) for i, r in per_method[zoom_choice].items()]
    zoom_ov = combined_overlay(view_rgb, zoom_items)
    zcols = st.columns(min(3, len(zoom_items)))
    for k, (idx, _, box) in enumerate(zoom_items):
        with zcols[k % len(zcols)]:
            st.image(zoom_crop(zoom_ov, box), caption=f"#{idx} {name_by_idx[idx]}")

    # 3) Other explainability techniques: one image per method, all aircraft in it
    st.markdown("#### Other explainability techniques")
    rows = []
    for m, per in per_method.items():
        rows.append({"Method": m,
                     "Combined": np.mean([r["combined"] for r in per.values()]),   # used only for ordering
                     "items": [(i, r["heat"], box_by_idx[i]) for i, r in per.items()]})
    rows.sort(key=lambda r: r["Combined"], reverse=True)

    st.success(f"Best technique for this image: **{rows[0]['Method']}**")
    cols = st.columns(3)
    for k, row in enumerate(rows):
        with cols[k % 3]:
            st.image(combined_overlay(view_rgb, row["items"]), width=400, caption=row["Method"])

    for f in sorted(failures):
        st.caption(f"⚠️ {f}")

# =========================== PIPELINE ===========================
st.divider()
st.subheader(f"Detection Results ({len(uploaded_files)} Images)")
all_detections = []
for image_number, uploaded_file in enumerate(uploaded_files, start=1):
    st.markdown(f"## Image {image_number}: `{uploaded_file.name}`")
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)

    with st.spinner(f"Detecting aircraft in {uploaded_file.name}..."):
        results = model.predict(source=image, conf=confidence, imgsz=IMGSZ, device=DEVICE, verbose=False)
    result = results[0]

    detections = []
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            detections.append({"Image": uploaded_file.name, "ClassID": class_id, "Aircraft": model.names[class_id],
                               "Confidence": float(boxes.conf[i].item()), "X1": round(x1, 2), "Y1": round(y1, 2),
                               "X2": round(x2, 2), "Y2": round(y2, 2)})
    df = pd.DataFrame(detections)
    all_detections.extend(detections)

    annotated_rgb = cv2.cvtColor(result.plot(), cv2.COLOR_BGR2RGB)
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("### Original Image")
        st.image(image, width=800)
    with col2:
        st.markdown("### YOLOv8 Detection")
        st.image(annotated_rgb, width=800)

    # EXPLAINABILITY (ALL aircraft in the image)
    if run_xai and detections and selected_methods:
        st.markdown("### Explainability")
        order = list(range(len(detections)))
        explained = []
        prog = st.progress(0.0, text="Starting explainability...")
        for n, i in enumerate(order):
            d = detections[i]
            prog.progress(n / len(order), text=f"Explaining aircraft {n + 1}/{len(order)}: {d['Aircraft']} "
                                               f"(first run is slow, results are cached)")
            res = explain_detection(model, image_np, d["ClassID"], (d["X1"], d["Y1"], d["X2"], d["Y2"]),
                                    tuple(selected_methods))
            explained.append((i, d, res))
        prog.empty()
        render_image_explanations(explained, uploaded_file.name)

    # DETECTION SUMMARY
    st.markdown("### Detection Summary")
    if detections:
        avg_c, max_c, n_types = df["Confidence"].mean(), df["Confidence"].max(), df["Aircraft"].nunique()
    else:
        avg_c = max_c = 0.0; n_types = 0
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Detections", len(detections))
    m2.metric("Aircraft Types", n_types)
    m3.metric("Average Confidence", f"{avg_c:.2%}")
    m4.metric("Highest Confidence", f"{max_c:.2%}")

    if detections:
        st.markdown("### Detection Details")
        display_df = df[["Aircraft", "Confidence"]].copy()
        display_df.index = [f"#{i}" for i in range(len(display_df))]
        display_df["Confidence"] = display_df["Confidence"].map(lambda x: f"{x:.2%}")
        st.dataframe(display_df, use_container_width=True)
    else:
        st.info("No aircraft detected above the selected confidence threshold.")
    st.divider()
