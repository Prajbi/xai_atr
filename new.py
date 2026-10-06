from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM, GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

# ---------------------------------------------------------------- PAGE CONFIG
st.set_page_config(
    page_title="Aircraft Detection Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------- PROJECT PATH
PROJECT_ROOT = Path("/home/admin1/Abishek-Intern/new_atr")
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8s_832_targeted_aug" / "weights" / "best.pt"
IMGSZ = 832

# ---------------------------------------------------------------- GPU CHECK
if not torch.cuda.is_available():
    st.error("CUDA GPU is not available. This application is configured for GPU-only inference.")
    st.stop()
DEVICE = 0
CAM_DEVICE = torch.device("cuda:0")
GPU_NAME = torch.cuda.get_device_name(0)


# ---------------------------------------------------------------- LOAD MODEL
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found:\n\n{MODEL_PATH}")
        st.stop()
    return YOLO(str(MODEL_PATH))


model = load_model()

# ---------------------------------------------------------------- SIDEBAR
st.sidebar.title("Detection Settings")
st.sidebar.markdown("### Confidence Threshold")
confidence = st.sidebar.slider("Confidence", min_value=0.01, max_value=1.00, value=0.25, step=0.01)

st.sidebar.divider()
st.sidebar.markdown("### CAM Layer Selection")
cam_mode = st.sidebar.radio(
    "Mode",
    ["Auto (best layer per image)", "Manual"],
    index=0,
)
manual_layer_idx = -3
if cam_mode == "Manual":
    manual_layer_idx = st.sidebar.number_input("Layer index", value=-3, step=1)

st.sidebar.divider()
st.sidebar.markdown("### Model Information")
st.sidebar.info("GPU-only inference")
st.sidebar.write(f"**Model:** `{MODEL_PATH.name}`")
st.sidebar.write("**Architecture:** YOLOv8s")
st.sidebar.write("**Task:** Object Detection")
if hasattr(model, "names"):
    st.sidebar.write(f"**Number of Classes:** {len(model.names)}")
st.sidebar.write(f"**GPU:** {GPU_NAME}")

# ---------------------------------------------------------------- HEADER
st.title("Military Aircraft Detection")
st.markdown("Upload one or multiple aircraft images to detect and classify aircraft using the trained YOLOv8s model.")
st.divider()

# ---------------------------------------------------------------- IMAGE UPLOAD
st.subheader("Upload Aircraft Images")
uploaded_files = st.file_uploader(
    "Choose one or more images", type=["jpg", "jpeg", "png"], accept_multiple_files=True
)
if not uploaded_files:
    st.info("Upload one or more aircraft images to start detection.")
    st.stop()


# ---------------------------------------------------------------- CAM HELPERS
class YoloCamWrapper(torch.nn.Module):
    """Makes the YOLO model return a single tensor so pytorch-grad-cam can use it."""

    def __init__(self, yolo_model):
        super().__init__()
        self.model = yolo_model.model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, (tuple, list)):
            return output[0]
        return output


class YoloCamTarget:
    def __call__(self, model_output):
        return model_output.sum()


@st.cache_resource
def get_cam_wrapper(_yolo_model):
    wrapped = YoloCamWrapper(_yolo_model).to(CAM_DEVICE)
    wrapped.eval()
    return wrapped


def get_head_layer_ids(yolo_model):
    """Indices of the layers feeding the Detect head, e.g. [15, 18, 21] for YOLOv8."""
    return list(yolo_model.model.model[-1].f)


def prepare_cam_input(image_rgb, imgsz, device):
    orig_h, orig_w = image_rgb.shape[:2]
    scale = imgsz / max(orig_h, orig_w)
    new_h, new_w = int(round(orig_h * scale)), int(round(orig_w * scale))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    pad_top = (imgsz - new_h) // 2
    pad_left = (imgsz - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    rgb_float = canvas.astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb_float).permute(2, 0, 1).unsqueeze(0).float().to(device)
    return tensor, pad_top, pad_left, new_h, new_w


def crop_gray(gray, pad_top, pad_left, new_h, new_w, orig_w, orig_h):
    gray = gray[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    gray = cv2.resize(gray, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)


def eigencam_gray(wrapped, layer_idx, tensor):
    target_layers = [wrapped.model.model[layer_idx]]
    with EigenCAM(model=wrapped, target_layers=target_layers) as cam:
        return cam(input_tensor=tensor, targets=None, eigen_smooth=True)[0]


def gradcam_gray(wrapped, layer_idx, tensor):
    target_layers = [wrapped.model.model[layer_idx]]
    tensor = tensor.clone().detach().requires_grad_(True)
    with torch.enable_grad():
        with GradCAM(model=wrapped, target_layers=target_layers) as cam:
            return cam(input_tensor=tensor, targets=[YoloCamTarget()])[0]


def box_score(gray, boxes_xyxy):
    """How concentrated the heat is inside detected boxes.
    mean heat inside boxes / mean heat over the whole image (>1 means focused on the aircraft)."""
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=bool)
    for x1, y1, x2, y2 in boxes_xyxy:
        mask[max(0, int(y1)):min(h, int(y2)), max(0, int(x1)):min(w, int(x2))] = True
    if not mask.any():
        return 0.0
    return float(gray[mask].mean() / (gray.mean() + 1e-8))


def generate_cam(yolo_model, image_rgb, boxes_xyxy, method="eigencam", mode="auto",
                 manual_layer=-3, imgsz=IMGSZ):
    """Returns (overlay_rgb, info_text). mode is 'auto' or 'manual'."""
    wrapped = get_cam_wrapper(yolo_model)
    gray_fn = eigencam_gray if method == "eigencam" else gradcam_gray
    orig_h, orig_w = image_rgb.shape[:2]
    tensor, pad_top, pad_left, new_h, new_w = prepare_cam_input(image_rgb, imgsz, CAM_DEVICE)

    def run(idx):
        g = gray_fn(wrapped, idx, tensor)
        return crop_gray(g, pad_top, pad_left, new_h, new_w, orig_w, orig_h)

    if mode == "manual":
        gray = run(int(manual_layer))
        info = f"manual layer {int(manual_layer)}"
    else:
        layer_ids = get_head_layer_ids(yolo_model)
        grays = {idx: run(idx) for idx in layer_ids}
        if boxes_xyxy:
            scores = {idx: box_score(g, boxes_xyxy) for idx, g in grays.items()}
            best = max(scores, key=scores.get)
            gray = grays[best]
            info = f"auto layer {best} (" + ", ".join(f"L{k}={v:.2f}" for k, v in scores.items()) + ")"
        else:
            gray = np.mean(list(grays.values()), axis=0)
            gray = (gray - gray.min()) / (gray.max() - gray.min() + 1e-8)
            info = f"no detections: averaged layers {layer_ids}"

    overlay = show_cam_on_image(image_rgb.astype(np.float32) / 255.0, gray, use_rgb=True)
    return overlay, info


# ---------------------------------------------------------------- PROCESS IMAGES
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
    boxes_xyxy = []
    if result.boxes is not None and len(result.boxes) > 0:
        boxes = result.boxes
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            confidence_score = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            boxes_xyxy.append([x1, y1, x2, y2])
            detections.append({
                "Image": uploaded_file.name,
                "ClassID": class_id,
                "Aircraft": model.names[class_id],
                "Confidence": confidence_score,
                "X1": round(x1, 2), "Y1": round(y1, 2),
                "X2": round(x2, 2), "Y2": round(y2, 2),
            })
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

    # ------------------------------------------------ EIGENCAM + GRADCAM
    mode = "manual" if cam_mode == "Manual" else "auto"
    cam_col1, cam_col2 = st.columns(2, gap="large")

    with cam_col1:
        with st.spinner(f"Generating EigenCAM for {uploaded_file.name}..."):
            try:
                overlay, info = generate_cam(model, image_np, boxes_xyxy, "eigencam", mode, manual_layer_idx)
                st.markdown("### EigenCAM Overlay")
                st.image(overlay, width=800)
                st.caption(info)
            except Exception as e:
                st.error(f"EigenCAM generation failed:\n\n{e}")
                st.info("The YOLOv8 detection result above is still valid.")

    with cam_col2:
        with st.spinner(f"Generating GradCAM for {uploaded_file.name}..."):
            try:
                overlay, info = generate_cam(model, image_np, boxes_xyxy, "gradcam", mode, manual_layer_idx)
                st.markdown("### GradCAM Overlay")
                st.image(overlay, width=800)
                st.caption(info)
            except Exception as e:
                st.error(f"GradCAM generation failed:\n\n{e}")
                st.info("The YOLOv8 detection result above is still valid.")

    # ------------------------------------------------ DETECTION SUMMARY
    st.markdown("### Detection Summary")
    total_detections = len(detections)
    if total_detections > 0:
        average_confidence = df["Confidence"].mean()
        highest_confidence = df["Confidence"].max()
        unique_aircraft = df["Aircraft"].nunique()
    else:
        average_confidence = highest_confidence = 0.0
        unique_aircraft = 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Detections", total_detections)
    m2.metric("Aircraft Types", unique_aircraft)
    m3.metric("Average Confidence", f"{average_confidence:.2%}")
    m4.metric("Highest Confidence", f"{highest_confidence:.2%}")

    # ------------------------------------------------ DETECTION DETAILS
    if total_detections > 0:
        st.markdown("### Detection Details")
        display_df = df[["Aircraft", "Confidence"]].copy()
        display_df["Confidence"] = display_df["Confidence"].map(lambda x: f"{x:.2%}")
        st.dataframe(display_df, use_container_width=True, hide_index=False)
    else:
        st.info("No aircraft detected above the selected confidence threshold.")

    st.divider()
