import io
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO
#PAGE CONFIG
st.set_page_config(page_title="Aircraft Detection Dashboard", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")

#PROJECT PATH
PROJECT_ROOT = Path("/home/admin1/Abishek-Intern/new_atr")
MODEL_PATH = PROJECT_ROOT / "new" / "models" / "yolo_aircraft.pt"

#GPU CHECK 
if not torch.cuda.is_available():
    st.error("CUDA GPU is not available. This application is configured for GPU-only inference.")
    st.stop()
DEVICE = 0
GPU_NAME = torch.cuda.get_device_name(0)

#LOAD MODEL
@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        st.error(f"Model file not found:\n\n{MODEL_PATH}")
        st.stop()
    return YOLO(str(MODEL_PATH))
model = load_model()

#SIDEBAR 
st.sidebar.title("Detection Settings")
st.sidebar.markdown("### Confidence Threshold")
confidence = st.sidebar.slider("Confidence",min_value=0.01,max_value=1.00,value=0.25,step=0.01)
st.sidebar.divider()
st.sidebar.markdown("### Model Information")
st.sidebar.info("GPU-only inference")
st.sidebar.write(f"**Model:** `{MODEL_PATH.name}`")
st.sidebar.write("**Architecture:** YOLOv8s")
st.sidebar.write("**Task:** Object Detection")
if hasattr(model, "names"):
    st.sidebar.write(f"**Number of Classes:** {len(model.names)}")
st.sidebar.write("**Input Size:** 832 × 832")
st.sidebar.write("**Device:** CUDA:0")
st.sidebar.write(f"**GPU:** {GPU_NAME}")

#HEADER 
st.title("Military Aircraft Detection")
st.markdown("""Upload one or multiple aircraft images to detect and classify aircraft using the trained YOLOv8s model.
An **EigenCAM heatmap** is generated for each image to visualize the regions that strongly activate the model.""")

st.divider()

#IMG UPLOAD
st.subheader("Upload Aircraft Images")
uploaded_files = st.file_uploader("Choose one or more images", type=["jpg","jpeg","png"], accept_multiple_files=True)
if not uploaded_files:
    st.info("Upload one or more aircraft images to start detection.")
    st.stop()

#EIGENCAM FUNC
def generate_eigencam(yolo_model, image_rgb, imgsz=832):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for EigenCAM.")
    device = torch.device("cuda:0")
    pytorch_model = yolo_model.model
    pytorch_model.to(device)
    pytorch_model.eval()
    activation = []
    def forward_hook(module, inputs, output):
        if isinstance(output, (tuple, list)):
            output = output[0]
        if torch.is_tensor(output) and output.ndim == 4:
            activation.append(output.detach())
    target_layer = None
    for layer in reversed(list(pytorch_model.model[:-1])):
        if isinstance(layer, torch.nn.Module):
            if hasattr(layer, "conv") or isinstance(layer, torch.nn.Conv2d):
                target_layer = layer
                break
    if target_layer is None:
        raise RuntimeError("Could not find a suitable convolutional feature layer for EigenCAM.")
    hook = target_layer.register_forward_hook(forward_hook)
    try:
        resized = cv2.resize(image_rgb, (imgsz, imgsz), interpolation=cv2.INTER_LINEAR)
        tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
        tensor = tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            pytorch_model(tensor)
        if not activation:
            raise RuntimeError("No spatial feature activation was captured.")
        features = activation[-1]
        if features.ndim != 4:
            raise RuntimeError(f"Unexpected activation shape: {features.shape}")
        features = features[0].float()
        channels, height, width = features.shape
        features = features.reshape(channels, height * width)
        features = features.transpose(0, 1)
        features = features - features.mean(dim=0, keepdim=True)
        try:
            _, _, vh = torch.linalg.svd(features, full_matrices=False)
            principal_component = vh[0]
        except RuntimeError:
            covariance = features.T @ features
            _, eigenvectors = torch.linalg.eigh(covariance)
            principal_component = eigenvectors[:, -1]
        cam = features @ principal_component
        cam = cam.reshape(height, width)
        cam = cam.detach().cpu().numpy()
        cam = np.abs(cam)
        cam -= cam.min()
        maximum = cam.max()
        if maximum > 0:
            cam /= maximum
        cam = cv2.resize(cam, (image_rgb.shape[1], image_rgb.shape[0]), interpolation=cv2.INTER_LINEAR)
        heatmap = np.uint8(cam * 255)
        heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        image_float = image_rgb.astype(np.float32)
        heatmap_float = heatmap.astype(np.float32)
        overlay = 0.55 * image_float + 0.45 * heatmap_float
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return heatmap, overlay
    finally:
        hook.remove()

#PROCESS IMG
st.divider()
st.subheader(f"Detection Results ({len(uploaded_files)} Images)")
all_detections = []
for image_number, uploaded_file in enumerate(uploaded_files, start=1):
    st.markdown(f"## Image {image_number}: `{uploaded_file.name}`")
    image = Image.open(uploaded_file).convert("RGB")
    image_np = np.array(image)
    with st.spinner(f"Detecting aircraft in {uploaded_file.name}..."): #YOLO INTER
        results = model.predict(source=image, conf=confidence,imgsz=832,device=DEVICE,verbose=False)
    result = results[0]
    detections = []  #  PROCESS DET
    if result.boxes is not None and len(result.boxes)>0:
        boxes = result.boxes
        for i in range(len(boxes)):
            class_id = int(boxes.cls[i].item())
            confidence_score = float(boxes.conf[i].item())
            x1, y1, x2, y2 = boxes.xyxy[i].tolist()
            class_name = model.names[class_id]
            detections.append({"Image":uploaded_file.name,"Class  ID":class_id,"Aircraft":class_name,"Confidence":confidence_score,"X1":round(x1,2),
            "Y1":round(y1,2),"X2":round(x2,2),"Y2":round(y2,2)})
    df = pd.DataFrame(detections)
    all_detections.extend(detections)
    annotated = result.plot() #CREATE IMG
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    col1, col2 = st.columns(2, gap="large") #OP
    with col1:
        st.markdown("### Original Image")
        st.image(image, use_container_width=True)
    with col2:
        st.markdown("### YOLOv8 Detection")
        st.image(annotated_rgb, use_container_width=True)

#EIGENCAM
with st.spinner(f"Generating EigenCAM for {uploaded_file.name}..."):
    try:
        heatmap, overlay = generate_eigencam(model,image_np,imgsz=832)
        cam_col1, cam_col2 = st.columns(2,gap="large")
        with cam_col1:
            st.markdown("### EigenCAM Heatmap")
            st.image(heatmap,use_container_width=True)
        with cam_col2:
            st.markdown("### EigenCAM Overlay")
            st.image(overlay,use_container_width=True)
    except Exception as e:
        st.error(f"EigenCAM generation failed:\n\n{e}")
        st.info("The YOLOv8 detection result above is still valid.")
# DET SUM
st.markdown("### Detection Summary")
total_detections = len(detections)
if total_detections > 0:
    average_confidence = df["Confidence"].mean()
    highest_confidence = df["Confidence"].max()
    unique_aircraft = df["Aircraft"].nunique()
else:
    average_confidence = 0.0
    highest_confidence = 0.0
    unique_aircraft = 0
metric1, metric2, metric3, metric4 = st.columns(4)
with metric1:
    st.metric("Total Detections",total_detections) 
with metric2:
    st.metric("Aircraft Types",unique_aircraft)
with metric3:
    st.metric("Average Confidence",f"{average_confidence:.2%}")
with metric4:
    st.metric("Highest Confidence",f"{highest_confidence:.2%}")
    
  
# DETECTION DET
if total_detections > 0:
    st.markdown("### Detection Details")
    display_df = df[["Aircraft", "Confidence"]].copy()
    display_df["Confidence"] = (display_df["Confidence"].map(lambda x: f"{x:.2%}"))
    st.dataframe(display_df,use_container_width=True,hide_index=False)
else:
    st.info("No aircraft detected above the selected confidence threshold.")
st.divider()
