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
from pytorch_grad_cam import EigenCAM, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from streamlit_image_coordinates import streamlit_image_coordinates

# PAGE CONFIG
st.set_page_config(page_title="Aircraft Detection Dashboard", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")

# PROJECT PATH
PROJECT_ROOT = Path("/home/admin1/Abishek-Intern/new_atr")
MODEL_PATH = PROJECT_ROOT /"models" / "yolov8s_832_79class_multiscale" / "weights" / "best.pt"
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
CAM_REGISTRY = {"EigenCAM": EigenCAM, "XGradCAM": XGradCAM}
OTHER_METHODS = ["SmoothGrad", "Occlusion"]
ALL_METHODS = list(CAM_REGISTRY) + OTHER_METHODS

EIGEN_LAYERS = [-2, -3, -4]        # tried automatically for EigenCAM
GRAD_LAYERS = [-8, -9, -10]        # tried automatically for gradient CAMs
SLOW_METHODS = set()
SLOW_LAYERS = [-9]
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

# Explainability always runs with the four fixed methods (no sidebar controls)
run_xai = True
selected_methods = ALL_METHODS

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

def smoothgrad_map(wrapped, tensor, target, n=16, sigma=0.08):
    acc = torch.zeros_like(tensor)
    for _ in range(n):
        acc += _input_grad(wrapped, tensor + sigma * torch.randn_like(tensor), target).abs()
    return acc.amax(1)[0].cpu().numpy()

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

def compute_heat(method, layer, wrapped, tensor, target, cls_id, idxs, base):
    if method in CAM_REGISTRY:
        return norm01(run_cam(method, layer, wrapped, tensor, target))
    if method == "SmoothGrad":
        return norm01(cv2.GaussianBlur(smoothgrad_map(wrapped, tensor, target), (0, 0), 6), 99.5)
    if method == "Occlusion":
        return norm01(cv2.GaussianBlur(occlusion_map(wrapped, tensor, cls_id, idxs, base), (0, 0), 12))
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


# ---------- selected-aircraft rendering ----------
def crop_with_margin(img, box, margin=0.45, min_width=420):
    x1, y1, x2, y2 = [float(v) for v in box]
    h, w = img.shape[:2]
    m = margin * max(x2 - x1, y2 - y1)
    xa, ya = max(0, int(x1 - m)), max(0, int(y1 - m))
    xb, yb = min(w, int(x2 + m)), min(h, int(y2 + m))
    crop = img[ya:yb, xa:xb]
    if crop.size == 0:
        return img
    if crop.shape[1] < min_width:
        factor = min_width / crop.shape[1]
        crop = cv2.resize(crop, None, fx=factor, fy=factor, interpolation=cv2.INTER_CUBIC)
    return crop


def render_selected_explainability(explain_result, detection, detection_id):
    view_rgb = explain_result["view_rgb"]
    box = explain_result["box_view"]
    valid = [r for r in explain_result["results"] if "error" not in r]
    failures = [r for r in explain_result["results"] if "error" in r]
    if not valid:
        st.error("All explainability methods failed for this aircraft.")
        for failure in failures:
            st.caption(f"{failure['name']}: {failure['error']}")
        return

    valid.sort(key=lambda r: r["combined"], reverse=True)
    st.success(f"Best technique for this aircraft: **{valid[0]['name']}**")
    st.caption("Explainability was computed only for the aircraft you selected. Scores are internal comparison metrics, not classification accuracy.")
    cols = st.columns(2)
    for i, item in enumerate(valid):
        heat = item["heat"].astype(np.float32)
        heat = norm01(heat)
        overlay = show_cam_on_image(view_rgb.astype(np.float32) / 255.0, heat, use_rgb=True)
        crop = crop_with_margin(overlay, box)
        with cols[i % 2]:
            st.image(crop, caption=f"{item['name']} — Localization: {item['loc']:.3f} | Faithfulness: {item['faith']:.3f}", use_container_width=True)
    for failure in failures:
        st.warning(f"{failure['name']} failed: {failure['error']}")


# =========================== PIPELINE ===========================
st.divider()
st.subheader(f"Detection Results ({len(uploaded_files)} Images)")

@st.cache_data(show_spinner=False, max_entries=32)
def cached_detect(image_bytes, conf, imgsz):
    """Cache deterministic detection output so Streamlit reruns don't rerun YOLO."""
    pil_image = Image.open(__import__('io').BytesIO(image_bytes)).convert("RGB")
    image_np = np.array(pil_image)
    prediction = model.predict(source=pil_image, conf=conf, imgsz=imgsz, device=DEVICE, verbose=False)[0]
    detections = []
    if prediction.boxes is not None and len(prediction.boxes) > 0:
        boxes = prediction.boxes
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            detections.append({
                "ClassID": class_id,
                "Aircraft": model.names[class_id],
                "Confidence": float(boxes.conf[i].item()),
                "X1": float(x1), "Y1": float(y1), "X2": float(x2), "Y2": float(y2)
            })
    annotated_rgb = cv2.cvtColor(prediction.plot(), cv2.COLOR_BGR2RGB)
    return image_np, annotated_rgb, detections

for image_number, uploaded_file in enumerate(uploaded_files, start=1):
    st.markdown(f"## Image {image_number}: `{uploaded_file.name}`")
    image_bytes = uploaded_file.getvalue()
    image_np, annotated_rgb, detections = cached_detect(image_bytes, confidence, IMGSZ)
    original_image = Image.fromarray(image_np)

    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("### 1. Original Image")
        st.image(original_image, use_container_width=True)
    with col2:
        st.markdown("### 2. YOLOv8 Detection — click a bounding box")
        if detections:
            # The coordinate component returns click coordinates in the displayed image.
            clicked = streamlit_image_coordinates(
                Image.fromarray(annotated_rgb),
                key=f"aircraft_click_{image_number}_{uploaded_file.name}"
            )
        else:
            st.image(annotated_rgb, use_container_width=True)
            clicked = None

    if detections:
        st.markdown("### 3. Detection list")
        display_df = pd.DataFrame([
            {"ID": f"#{i + 1}", "Aircraft": d["Aircraft"], "Confidence": f"{d['Confidence']:.2%}"}
            for i, d in enumerate(detections)
        ])
        st.dataframe(display_df, use_container_width=True, hide_index=True)

        # A click is mapped from the displayed image coordinates back to original pixels.
        clicked_id = None
        if clicked and clicked.get("x") is not None and clicked.get("y") is not None:
            disp_w = float(clicked.get("width") or annotated_rgb.shape[1])
            disp_h = float(clicked.get("height") or annotated_rgb.shape[0])
            px = clicked["x"] * image_np.shape[1] / disp_w
            py = clicked["y"] * image_np.shape[0] / disp_h
            hits = []
            for i, d in enumerate(detections):
                if d["X1"] <= px <= d["X2"] and d["Y1"] <= py <= d["Y2"]:
                    area = max(1.0, (d["X2"] - d["X1"]) * (d["Y2"] - d["Y1"]))
                    hits.append((area, i))
            if hits:
                # If boxes overlap, select the smallest box containing the click.
                clicked_id = min(hits)[1]
                st.session_state[f"selected_aircraft_{image_number}"] = clicked_id
                # Keep the dropdown synchronized with the aircraft clicked on the image.
                st.session_state[f"aircraft_select_{image_number}_{uploaded_file.name}"] = clicked_id
            else:
                st.info("Your click was outside all detected bounding boxes. Click inside an aircraft box.")

        selection_key = f"selected_aircraft_{image_number}"
        current_selection = st.session_state.get(selection_key, 0)
        if current_selection >= len(detections):
            current_selection = 0
        selected_id = st.selectbox(
            "Selected aircraft (you can also choose from this list)",
            options=list(range(len(detections))),
            index=current_selection,
            format_func=lambda i: f"#{i + 1} — {detections[i]['Aircraft']} ({detections[i]['Confidence']:.1%})",
            key=f"aircraft_select_{image_number}_{uploaded_file.name}"
        )
        st.session_state[selection_key] = selected_id
        selected = detections[selected_id]
        selected_box = (selected["X1"], selected["Y1"], selected["X2"], selected["Y2"])

        st.markdown("### 4. Zoom on Selected Aircraft")
        st.write(f"**Aircraft #{selected_id + 1}:** {selected['Aircraft']}  |  **Confidence:** {selected['Confidence']:.2%}")
        st.image(crop_with_margin(image_np, selected_box), caption="Cropped from the original image.", use_container_width=True)

        st.markdown("### 5. Explainability of Selected Aircraft")
        run_key = f"run_xai_{image_number}_{uploaded_file.name}_{selected_id}"
        if st.button(f"Run Explainability for Aircraft #{selected_id + 1}", key=run_key, type="primary"):
            with st.spinner(f"Running explainability for Aircraft #{selected_id + 1} ({selected['Aircraft']})…"):
                result = explain_detection(
                    model,
                    image_np,
                    selected["ClassID"],
                    selected_box,
                    tuple(ALL_METHODS)
                )
            render_selected_explainability(result, selected, selected_id + 1)
        else:
            st.info("Explainability has not run yet. Click the button above when you are ready.")

        st.markdown("### Detection Summary")
        summary_cols = st.columns(4)
        summary_cols[0].metric("Total Detections", len(detections))
        summary_cols[1].metric("Aircraft Types", len(set(d["Aircraft"] for d in detections)))
        summary_cols[2].metric("Average Confidence", f"{np.mean([d['Confidence'] for d in detections]):.2%}")
        summary_cols[3].metric("Highest Confidence", f"{max(d['Confidence'] for d in detections):.2%}")
    else:
        st.info("No aircraft detected above the selected confidence threshold.")
    st.divider()
