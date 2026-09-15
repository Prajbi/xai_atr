import streamlit as st
from PIL import Image
import pandas as pd
import base64
import os
import datetime
from utils_8 import AircraftClassifier, ShipClassifier

# --- 1. SET PAGE CONFIG FIRST ---
st.set_page_config(page_title="ATR", page_icon="✈️", layout="wide")

# --- 2. SESSION STATE INITIALIZATION ---
if 'mode' not in st.session_state:
    st.session_state.mode = None
if 'saving' not in st.session_state:
    st.session_state.saving = False
if 'results' not in st.session_state:
    st.session_state.results = []
if 'last_uploaded_filenames' not in st.session_state:
    st.session_state.last_uploaded_filenames = []

# --- Helper functions ---
def get_base64_image(img_path):
    try:
        with open(img_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except FileNotFoundError:
        return ""

logo_left_base64 = get_base64_image("images/cabs_logo.png").replace("\n", "")
logo_right_base64 = get_base64_image("images/ailab_logo.png").replace("\n", "")

@st.cache_resource
def load_aircraft_model():
    return AircraftClassifier()

@st.cache_resource
def load_ship_model():
    return ShipClassifier()

aircraft_model = load_aircraft_model()
ship_model = load_ship_model()

def apply_dynamic_styles(active_mode):
    try:
        with open("style_8.css") as f:
            css_content = f.read()
        BLUE, GREY, WHITE, BLACK = "#37585c", "#E0E0E0", "white", "black"        
        if active_mode == "Aircraft":
            b1_bg, b1_txt, b2_bg, b2_txt = BLUE, WHITE, GREY, BLACK
        elif active_mode == "Ship":
            b1_bg, b1_txt, b2_bg, b2_txt = GREY, BLACK, BLUE, WHITE
        else:
            b1_bg, b1_txt, b2_bg, b2_txt = GREY, BLACK, GREY, BLACK
        
        save_bg = BLUE if st.session_state.saving else GREY
        dynamic_style = f"""
            <style>
            {css_content}
            .block-container {{ padding-top: 0rem !important; padding-bottom: 0rem !important; }}
            header {{ visibility: hidden; }}
            .stTable {{ font-size: 11px !important; }}
            div[data-testid="stColumn"]:nth-of-type(1) div.stButton button {{ background-color: {b1_bg} !important; color: {b1_txt} !important; }}
            div[data-testid="stColumn"]:nth-of-type(2) div.stButton button {{ background-color: {b2_bg} !important; color: {b2_txt} !important; }}
            div.stButton > button:has(div p:contains("Save All")) {{ background-color: {save_bg} !important; color: white !important; }}
            </style>
        """
        st.markdown(dynamic_style, unsafe_allow_html=True)
    except FileNotFoundError:
        pass

def save_results_logic(mode):
    """Logic separated from the button to be used in a callback"""
    if not st.session_state.results:
        return
    
    st.session_state.saving = True
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    folder_name = f"results_{mode}_{timestamp}"
    os.makedirs(folder_name, exist_ok=True)
    all_detections = []
    for item in st.session_state.results:
        fname = item['filename']
        img = item['image']
        dets = item['detections']
        base_name = os.path.splitext(fname)[0]
        img.save(f"{folder_name}/{base_name}_result.jpg")
        if dets and dets[0][0] != "No Object Detected":
            for obj, conf in dets:
                all_detections.append({"Filename": fname, "Object": obj, "Confidence": conf})
        else:
            all_detections.append({"Filename": fname, "Object": "None", "Confidence": 0})
    
    pd.DataFrame(all_detections).to_csv(f"{folder_name}/detections_summary.csv", index=False)
    st.session_state.save_folder = folder_name

# --- 3. APPLY STYLES IMMEDIATELY ---
apply_dynamic_styles(st.session_state.mode)

# --- UI Layout ---
html_template = """<div style="display: flex; align-items: center; justify-content: center; background-color: #1e2c4f; padding: 8px 30px; border-radius: 15px; color: white; margin-bottom: 20px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); gap: 10px;">
    <div style="flex: 1; display: flex; justify-content: flex-start;"><img src="data:image/png;base64,{left}" style="width: 130px; height: auto;"></div>
    <div style="flex: 2; text-align: center;"><h1 style="margin: 0; color: white; font-family: 'Segoe UI', sans-serif; font-size: 3.5rem; font-weight: 700;">AI-based Automatic Target Recognition</h1></div>
    <div style="flex: 1; display: flex; justify-content: flex-end;"><img src="data:image/png;base64,{right}" style="width: 140px; height: auto;"></div>
</div>"""
st.markdown(html_template.format(left=logo_left_base64, right=logo_right_base64), unsafe_allow_html=True)

btn_col1, btn_col2 = st.columns(2)
with btn_col1:
    if st.button("✈️ Aircraft Classification", use_container_width=True):
        st.session_state.mode = "Aircraft"
        st.session_state.saving = False
        st.session_state.results = []
        st.session_state.last_uploaded_filenames = []
        st.rerun()
with btn_col2:
    if st.button("🚢 Ship Classification", use_container_width=True):
        st.session_state.mode = "Ship"
        st.session_state.saving = False
        st.session_state.results = []
        st.session_state.last_uploaded_filenames = []
        st.rerun()

if st.session_state.mode is None:
    st.info("Please select a category above to upload images.")
else:
    current_mode = st.session_state.mode
    active_model = aircraft_model if current_mode == "Aircraft" else ship_model
    uploaded_files = st.file_uploader(f"Upload {current_mode} Image(s)", type=["jpg", "png", "jpeg"], accept_multiple_files=True)

    current_filenames = [f.name for f in uploaded_files] if uploaded_files else []
    if uploaded_files and current_filenames != st.session_state.last_uploaded_filenames:
        st.session_state.results = []
        st.session_state.saving = False
        files_to_process = uploaded_files[::-1]
        with st.status(f"🚀 Processing {len(files_to_process)} images...", expanded=True) as status:
            for idx, uploaded_file in enumerate(files_to_process):
                status.write(f"🔍 Analyzing image {idx+1}/{len(files_to_process)}: `{uploaded_file.name}`...")
                original_image = Image.open(uploaded_file).convert('RGB')
                annotated_image = active_model.get_annotated_image(original_image)
                detections = active_model.predict(original_image)
                st.session_state.results.append({
                    "filename": uploaded_file.name,
                    "image": annotated_image,
                    "original": original_image,
                    "detections": detections
                })
            status.update(label="✅ Processing Complete!", state="complete", expanded=False)
        st.toast(f"Successfully processed {len(files_to_process)} images!", icon="✅")
        st.session_state.last_uploaded_filenames = current_filenames

    if st.session_state.results:
        info_col, save_col = st.columns([4, 1])
        with info_col:
            st.info(f"📂 Processed {len(st.session_state.results)} file(s).")
        with save_col:
            st.markdown('<div style="margin-top: 10px;"></div>', unsafe_allow_html=True)
            # FIX: Use on_click callback to handle state change and processing in one go
            if st.button("💾 Save All", use_container_width=True, on_click=save_results_logic, args=(current_mode,)):
                pass 
        
        # Display success message if folder was just saved
        if 'save_folder' in st.session_state:
            st.success(f"Saved to: `{st.session_state.save_folder}`")
            # Clear save_folder from state so the message doesn't persist forever
            del st.session_state.save_folder

        st.markdown("---")
        for i in range(0, len(st.session_state.results), 2):
            row_cols = st.columns(4)
            for img_idx in range(2):
                res_idx = i + img_idx
                if res_idx < len(st.session_state.results):
                    item = st.session_state.results[res_idx]
                    with row_cols[img_idx * 2]:
                        st.markdown(f"**📄 {item['filename']}**")
                        st.image(item['original'], use_container_width=True)
                    with row_cols[(img_idx * 2) + 1]:
                        st.markdown("**OUTPUT IMAGE**")
                        st.image(item['image'], use_container_width=True)
                        if item['detections'] and item['detections'][0][0] != "No Object Detected":
                            df = pd.DataFrame(item['detections'], columns=["Obj", "Conf"])
                            df["Conf"] = df["Conf"].apply(lambda x: f"{x:.1%}")
                            st.table(df)
                        else:
                            st.caption("No objects detected.")
            st.markdown("---")
