import io
from pathlib import Path
import cv2
import numpy as np
import pandas as pd
import streamlit as st
import torch
from PIL import Image
from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
#PAGE CONFIG
st.set_page_config(page_title="Aircraft Detection Dashboard", page_icon="✈️", layout="wide", initial_sidebar_state="expanded")

#PROJECT PATH
PROJECT_ROOT = Path("/home/admin1/Abishek-Intern/new_atr")
MODEL_PATH = PROJECT_ROOT / "models" / "yolov8s_832_finetuned" / "weights" / "best.pt"

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
st.sidebar.markdown("### EigenCAM Settings")
num_layers = len(model.model.model)
cam_layer_idx = st.sidebar.slider(
    "Target Layer Index",
    min_value=-num_layers,
    max_value=-1,
    value=-4,
    step=1,

)
eigen_smooth = st.sidebar.checkbox("Eigen Smooth", value=True, help="Applies extra smoothing to reduce noise in the CAM.")
st.sidebar.divider()
st.sidebar.markdown("### Model Information")
st.sidebar.info("GPU-only inference")
st.sidebar.write(f"**Model:** `{MODEL_PATH.name}`")
st.sidebar.write("**Architecture:** YOLOv8s")
st.sidebar.write("**Task:** Object Detection")
if hasattr(model, "names"):
    st.sidebar.write(f"**Number of Classes:** {len(model.names)}")
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
class YoloEigenCamWrapper(torch.nn.Module):
    def __init__(self, yolo_model):
        super().__init__()
        self.model = yolo_model.model

    def forward(self, x):
        output = self.model(x)
        if isinstance(output, (tuple, list)):
            return output[0]
        return output

def generate_eigencam(yolo_model, image_rgb, imgsz=832, target_layer_idx=-4, eigen_smooth=True):
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required for EigenCAM.")
    device = torch.device("cuda:0")
    wrapped = YoloEigenCamWrapper(yolo_model)
    wrapped = wrapped.to(device)
    wrapped.eval()
    target_layer = wrapped.model.model[target_layer_idx]
    target_layers = [target_layer]
    orig_h, orig_w = image_rgb.shape[:2]
    scale = imgsz / max(orig_h, orig_w)
    new_h, new_w = int(round(orig_h * scale)), int(round(orig_w * scale))
    resized = cv2.resize(image_rgb, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    pad_top = (imgsz - new_h) // 2
    pad_left = (imgsz - new_w) // 2
    canvas[pad_top:pad_top + new_h, pad_left:pad_left + new_w] = resized
    rgb_float = canvas.astype(np.float32) / 255.0
    tensor = torch.from_numpy(rgb_float).permute(2, 0, 1).unsqueeze(0)
    tensor = tensor.float().to(device)
    with EigenCAM(model=wrapped, target_layers=target_layers) as cam:
        grayscale_cam = cam(input_tensor=tensor, targets=None, eigen_smooth=eigen_smooth)
    grayscale_cam = grayscale_cam[0]
    visualization = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    visualization = visualization[pad_top:pad_top + new_h, pad_left:pad_left + new_w]
    visualization = cv2.resize(visualization, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR)
    return visualization

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
            detections.append({"Image":uploaded_file.name,"ClassID":class_id,"Aircraft":class_name,"Confidence":confidence_score,"X1":round(x1,2),
            "Y1":round(y1,2),"X2":round(x2,2),"Y2":round(y2,2)})
    df = pd.DataFrame(detections)
    all_detections.extend(detections)
    annotated = result.plot() #CREATE IMG
    annotated_rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
    col1, col2 = st.columns(2, gap="large") #OP
    with col1:
        st.markdown("### Original Image")
        st.image(image,width=800)
    with col2:
        st.markdown("### YOLOv8 Detection")
        st.image(annotated_rgb,width=800)

    #EIGENCAM
    with st.spinner(f"Generating EigenCAM for {uploaded_file.name}..."):
        try:
            overlay = generate_eigencam(model,image_np,imgsz=832,target_layer_idx=cam_layer_idx,eigen_smooth=eigen_smooth)
            st.markdown("### EigenCAM Overlay")
            st.image(overlay,width=800)
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
