import base64
import io
import time
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
import torch
from PIL import Image
from pytorch_grad_cam import EigenCAM, XGradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from ultralytics import YOLO

# PAGE CONFIG
st.set_page_config(page_title="Aircraft Detection Dashboard", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")

# PROJECT PATH
PROJECT_ROOT = Path("/home/admin1/Abishek-Intern/new_atr")
MODEL_PATH = PROJECT_ROOT / "models"  / "best (2).pt"
#/ "yolov8s_832_79class_multiscale" / "weights"
# / "main" / "models" / "yolo_aircraft.pt"
# / "models" / "yolov8s_832_targeted_aug" / "weights" / "best.pt"

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
EIGEN_LAYERS = [-2, -3, -4]        
GRAD_LAYERS = [-8, -9, -10]        
SLOW_METHODS = set()
SLOW_LAYERS = [-9]
W_LOC = 0.5                       

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



COMPONENT_DIR = Path(__file__).parent / "aircraft_component"
COMPONENT_HTML = r"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  html, body { margin:0; padding:0; font-family:Arial,sans-serif; background:transparent; }
  #hint { font-size:13px; color:#777; margin-bottom:6px; }
  #stage { position:relative; width:100%; overflow:hidden; background:#151515; border-radius:8px; }
  #view { width:100%; height:100%; display:block; cursor:crosshair; }
  #popup { display:none; position:absolute; z-index:5; width:230px; padding:6px; background:#fff; color:#111;
           border:2px solid #f0c419; border-radius:8px; box-shadow:0 4px 18px #0008; pointer-events:none; }
  #popup-title { font-size:12px; font-weight:bold; margin-bottom:4px; }
  #zoom { width:216px; height:130px; display:block; background:#222; }
  #popup-foot { font-size:11px; color:#444; margin-top:3px; }
  #selected { margin-top:7px; padding:8px 10px; border-radius:6px; background:#f3f4f6; color:#111; font-weight:600; }
</style>
</head>
<body>
<div id="stage">
  <canvas id="view"></canvas>
  <div id="popup">
    <div id="popup-title"></div>
    <canvas id="zoom" width="216" height="130"></canvas>
    <div id="popup-foot">Click the box to select</div>
  </div>
</div>
<div id="selected">Selected aircraft: none</div>
<script>
  function send(type, extra) {
    window.parent.postMessage(Object.assign({isStreamlitMessage: true, type: type}, extra || {}), "*");
  }

  const stage = document.getElementById('stage');
  const canvas = document.getElementById('view');
  const ctx = canvas.getContext('2d');
  const popup = document.getElementById('popup');
  const zoomCanvas = document.getElementById('zoom');
  const zctx = zoomCanvas.getContext('2d');
  const title = document.getElementById('popup-title');
  const selectedBox = document.getElementById('selected');

  let img = null, imgSrc = "", dets = [], selected = -1, hovered = -1;
  let scale = 1, ox = 0, oy = 0, stageH = 560;

  function updateSelectedText() {
    selectedBox.textContent = (selected >= 0 && dets[selected])
      ? 'Selected aircraft: ' + dets[selected].label : 'Selected aircraft: none';
  }

  function fitCanvas() {
    if (!img) return;
    const rect = stage.getBoundingClientRect();
    const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.floor(rect.width * dpr));
    canvas.height = Math.max(1, Math.floor(rect.height * dpr));
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    scale = Math.min(rect.width / img.width, rect.height / img.height);
    ox = (rect.width - img.width * scale) / 2;
    oy = (rect.height - img.height * scale) / 2;
    draw();
  }

  function draw() {
    if (!img) return;
    const rect = stage.getBoundingClientRect();
    const cw = rect.width, ch = rect.height;
    ctx.clearRect(0, 0, cw, ch);
    ctx.fillStyle = '#151515'; ctx.fillRect(0, 0, cw, ch);
    ctx.drawImage(img, ox, oy, img.width * scale, img.height * scale);
    dets.forEach((d, i) => {
      const b = d.box;
      const x = ox + b[0] * scale, y = oy + b[1] * scale;
      const w = (b[2] - b[0]) * scale, h = (b[3] - b[1]) * scale;
      const active = (i === selected), hover = (i === hovered);
      const col = active ? '#ffd400' : (hover ? '#00e5ff' : '#00ff66');
      ctx.lineWidth = (active || hover) ? 4 : 2;
      ctx.strokeStyle = col;
      ctx.strokeRect(x, y, w, h);
      const fontSize = Math.max(11, Math.min(16, cw / 75));
      ctx.font = 'bold ' + fontSize + 'px Arial';
      const tw = ctx.measureText(d.label).width;
      const lh = fontSize + 8;
      const lx = Math.max(0, Math.min(x, cw - tw - 12));
      const ly = Math.max(lh, y);
      ctx.fillStyle = col;
      ctx.fillRect(lx, ly - lh, tw + 10, lh);
      ctx.fillStyle = '#111';
      ctx.fillText(d.label, lx + 5, ly - 5);
    });
  }

  function hitTest(px, py) {
    const x = (px - ox) / scale, y = (py - oy) / scale;
    const hits = dets.filter(d => x >= d.box[0] && x <= d.box[2] && y >= d.box[1] && y <= d.box[3]);
    if (!hits.length) return -1;
    hits.sort((a, b) => ((a.box[2]-a.box[0])*(a.box[3]-a.box[1])) - ((b.box[2]-b.box[0])*(b.box[3]-b.box[1])));
    return hits[0].id;   // smallest box under the cursor wins (handles overlaps)
  }

  function drawPopup(d, px, py) {
    const b = d.box, margin = Math.max(b[2]-b[0], b[3]-b[1]) * 0.35;
    const x1 = Math.max(0, b[0]-margin), y1 = Math.max(0, b[1]-margin);
    const x2 = Math.min(img.width, b[2]+margin), y2 = Math.min(img.height, b[3]+margin);
    const sw = x2 - x1, sh = y2 - y1;
    zctx.fillStyle = '#222'; zctx.fillRect(0, 0, zoomCanvas.width, zoomCanvas.height);
    if (sw > 0 && sh > 0) {
      const s = Math.min(zoomCanvas.width / sw, zoomCanvas.height / sh);
      const dw = sw * s, dh = sh * s;
      const offx = (zoomCanvas.width - dw) / 2, offy = (zoomCanvas.height - dh) / 2;
      zctx.drawImage(img, x1, y1, sw, sh, offx, offy, dw, dh);
      zctx.strokeStyle = '#ffd400'; zctx.lineWidth = 3;
      zctx.strokeRect((b[0]-x1)*s + offx, (b[1]-y1)*s + offy, (b[2]-b[0])*s, (b[3]-b[1])*s);
    }
    title.textContent = d.label;
    popup.style.display = 'block';
    popup.style.left = Math.min(Math.max(4, px + 14), stage.clientWidth - 242) + 'px';
    popup.style.top = Math.min(Math.max(4, py + 14), stage.clientHeight - 205) + 'px';
  }

  canvas.addEventListener('mousemove', e => {
    const r = canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    hovered = hitTest(x, y);
    draw();
    if (hovered >= 0) drawPopup(dets[hovered], x, y); else popup.style.display = 'none';
  });
  canvas.addEventListener('mouseleave', () => { hovered = -1; popup.style.display = 'none'; draw(); });
  canvas.addEventListener('click', e => {
    const r = canvas.getBoundingClientRect();
    const id = hitTest(e.clientX - r.left, e.clientY - r.top);
    if (id < 0) return;
    selected = id;
    updateSelectedText();
    draw();
    // a timestamp makes every click a "new" value, even re-clicking the same box
    send('streamlit:setComponentValue', {value: {id: id, t: Date.now()}, dataType: 'json'});
  });
  window.addEventListener('resize', fitCanvas);

  function onRender(args) {
    dets = args.dets || [];
    selected = (args.selected === null || args.selected === undefined) ? -1 : args.selected;
    stageH = args.height || 560;
    stage.style.height = stageH + 'px';
    updateSelectedText();
    if (args.image && args.image !== imgSrc) {
      imgSrc = args.image;
      const im = new Image();
      im.onload = () => { img = im; fitCanvas(); };
      im.src = imgSrc;
    } else {
      fitCanvas();
    }
    send('streamlit:setFrameHeight', {height: stageH + 90});
  }

  window.addEventListener('message', e => {
    if (e.data && e.data.type === 'streamlit:render') onRender(e.data.args);
  });
  send('streamlit:componentReady', {apiVersion: 1});
</script>
</body>
</html>
"""

def _ensure_component_files():
    COMPONENT_DIR.mkdir(exist_ok=True)
    index = COMPONENT_DIR / "index.html"
    if not index.exists() or index.read_text(encoding="utf-8") != COMPONENT_HTML:
        index.write_text(COMPONENT_HTML, encoding="utf-8")
_ensure_component_files()
_aircraft_component = components.declare_component("aircraft_selector", path=str(COMPONENT_DIR))

def render_interactive_detection(image_rgb, detections, selected_id, key, height=560):
    """Interactive detection canvas. Returns {"id": int, "t": ms} after a click, otherwise None."""
    buffer = io.BytesIO()
    Image.fromarray(image_rgb).save(buffer, format="JPEG", quality=90)
    image_src = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    payload = [{
        "id": i,
        "label": f"#{i + 1} {d['Aircraft']} {d['Confidence']:.0%}",
        "box": [d["X1"], d["Y1"], d["X2"], d["Y2"]],
    } for i, d in enumerate(detections)]
    return _aircraft_component(image=image_src, dets=payload,
                               selected=-1 if selected_id is None else int(selected_id),
                               height=height, key=key, default=None)


#XAI CORE
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

#CAM
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

#I/P Grad 
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

#Perturbation 
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

#Metrics 
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
        m = np.zeros(H * W, np.float32)
        m[order[:int(f * H * W)]] = 1
        m = torch.from_numpy(m.reshape(1, 1, H, W)).to(tensor.device)
        dels.append(tensor * (1 - m) + gray * m)
        inss.append(tensor * m + gray * (1 - m))
    sc = score_fn(wrapped(torch.cat(dels + inss, 0)), cls_id, idxs).float().cpu().numpy() / base
    sc = np.clip(sc, 0, 1)
    n = len(fracs)
    return float(np.clip(sc[n:].mean() - sc[:n].mean(), 0, 1))

#Det+metho
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


#Selected aircraft rendering 
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

def render_selected_explainability(explain_result):
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
    cols = st.columns(2)
    for i, item in enumerate(valid):
        heat = norm01(item["heat"].astype(np.float32))
        overlay = show_cam_on_image(view_rgb.astype(np.float32) / 255.0, heat, use_rgb=True)
        crop = crop_with_margin(overlay, box)
        with cols[i % 2]:
            st.image(crop, caption=f"{item['name']} — Localization: {item['loc']:.3f} | Faithfulness: {item['faith']:.3f}",
                     use_container_width=True)
    for failure in failures:
        st.warning(f"{failure['name']} failed: {failure['error']}")


#PIPELINE
st.divider()
st.subheader(f"Detection Results ({len(uploaded_files)} Images)")
@st.cache_data(show_spinner=False, max_entries=32)
def cached_detect(image_bytes, conf, imgsz):
    """Cache deterministic detection output so Streamlit reruns don't rerun YOLO."""
    pil_image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
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
                "X1": float(x1), "Y1": float(y1), "X2": float(x2), "Y2": float(y2),
            })
    return image_np, detections

for image_number, uploaded_file in enumerate(uploaded_files, start=1):
    st.markdown(f"## Image {image_number}: `{uploaded_file.name}`")
    image_bytes = uploaded_file.getvalue()
    image_np, detections = cached_detect(image_bytes, confidence, IMGSZ)
    original_image = Image.fromarray(image_np)
    uid = f"{image_number}_{uploaded_file.name}"
    selection_key = f"selected_aircraft_{uid}"
    dropdown_key = f"aircraft_select_{uid}"
    comp_key = f"aircraft_hover_{uid}"
    last_click_key = f"last_click_{uid}"
    current_selection = st.session_state.get(selection_key, 0)
    if not detections:
        current_selection = None
    elif current_selection is None or current_selection >= len(detections):
        current_selection = 0
    col1, col2 = st.columns(2, gap="large")
    with col1:
        st.markdown("### 1. Original Image")
        st.image(original_image, use_container_width=True)
    with col2:
        st.markdown("### 2. YOLOv8 Detection")
        if detections:
            click = render_interactive_detection(image_np, detections, current_selection, key=comp_key)
            # FIX: the component keeps its last value across reruns, so only react when the
            # click is NEW; otherwise it would override later dropdown changes.
            if isinstance(click, dict) and click != st.session_state.get(last_click_key):
                st.session_state[last_click_key] = click
                cid = int(click.get("id", -1))
                if 0 <= cid < len(detections):
                    current_selection = cid
                    st.session_state[selection_key] = cid
                    st.session_state[dropdown_key] = cid
        else:
            st.info("No aircraft detected.")
    if detections:
        if current_selection is None or current_selection >= len(detections):
            current_selection = 0
        if dropdown_key not in st.session_state or st.session_state[dropdown_key] >= len(detections):
            st.session_state[dropdown_key] = current_selection
        selected_id = st.selectbox(
            "Selected aircraft (click a box above or choose here)",
            options=list(range(len(detections))),
            format_func=lambda i: f"#{i + 1} — {detections[i]['Aircraft']} ({detections[i]['Confidence']:.1%})",
            key=dropdown_key,
        )
        st.session_state[selection_key] = selected_id
        selected = detections[selected_id]
        selected_box = (selected["X1"], selected["Y1"], selected["X2"], selected["Y2"])
        if selected_id != current_selection:
            st.markdown(f"**Current selection: #{selected_id + 1} — {selected['Aircraft']}**")
            st.image(
                crop_with_margin(image_np, selected_box, margin=0.35, min_width=320),
                caption="Quick zoom preview of the current dropdown selection.",
                use_container_width=True,
            )
        st.markdown("### 3. Detection list")
        display_df = pd.DataFrame([
            {"ID": f"#{i + 1}", "Aircraft": d["Aircraft"], "Confidence": f"{d['Confidence']:.2%}"}
            for i, d in enumerate(detections)
        ])
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        st.markdown("### 4. Selected Aircraft Details")
        st.write(f"**Aircraft #{selected_id + 1}:** {selected['Aircraft']}  |  **Confidence:** {selected['Confidence']:.2%}")
        st.image(crop_with_margin(image_np, selected_box), caption="Cropped from the original image.", use_container_width=True)
        st.markdown("### 5. Explainability of Selected Aircraft")
        with st.spinner(f"Running explainability for Aircraft #{selected_id + 1} ({selected['Aircraft']})…"):
            result = explain_detection(model, image_np, selected["ClassID"], selected_box, tuple(ALL_METHODS))
        render_selected_explainability(result)
        st.markdown("### Detection Summary")
        summary_cols = st.columns(4)
        summary_cols[0].metric("Total Detections", len(detections))
        summary_cols[1].metric("Aircraft Types", len(set(d["Aircraft"] for d in detections)))
        summary_cols[2].metric("Average Confidence", f"{np.mean([d['Confidence'] for d in detections]):.2%}")
        summary_cols[3].metric("Highest Confidence", f"{max(d['Confidence'] for d in detections):.2%}")
    else:
        st.info("No aircraft detected above the selected confidence threshold.")
    st.divider()
