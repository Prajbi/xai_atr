import os
import cv2
import numpy as np
import torch

from pathlib import Path
from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from pytorch_grad_cam import EigenCAM


# ============================================================
# PATHS
# ============================================================

MODEL_PATH = "/home/admin1/Abishek-Intern/new_atr/runs/best.pt"

SOURCE_DIR = (
    "/home/admin1/Abishek-Intern/new_atr/"
    "data/balanced_800/images/test"
)

OUTPUT_DIR = (
    "/home/admin1/Abishek-Intern/new_atr/"
    "runs/eigencam_yolo26n_balanced800"
)

IMG_SIZE = 832
CONF = 0.05

# YOLO26n:
# layer 16 = high-resolution P3 feature map
TARGET_LAYER_INDEX = 16

# Number of images
MAX_IMAGES = 10


# ============================================================
# YOLO WRAPPER
# ============================================================

class YOLOWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        output = self.model(x)

        if isinstance(output, tuple):
            return output[0]

        return output


# ============================================================
# REMOVE LETTERBOX
# ============================================================

def remove_letterbox(cam, original_shape, new_shape):

    orig_h, orig_w = original_shape[:2]

    new_h, new_w = new_shape

    scale = min(
        new_w / orig_w,
        new_h / orig_h
    )

    resized_w = int(round(orig_w * scale))
    resized_h = int(round(orig_h * scale))

    pad_x = (new_w - resized_w) // 2
    pad_y = (new_h - resized_h) // 2

    cam = cam[
        pad_y:pad_y + resized_h,
        pad_x:pad_x + resized_w
    ]

    cam = cv2.resize(
        cam,
        (orig_w, orig_h),
        interpolation=cv2.INTER_LINEAR
    )

    return cam


# ============================================================
# MAIN
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print("Loading model...")
yolo = YOLO(MODEL_PATH)

net = yolo.model.to(device)
net.eval()

# Important for gradients
for p in net.parameters():
    p.requires_grad_(True)

wrapper = YOLOWrapper(net).to(device)

# ============================================================
# TARGET LAYER
# ============================================================

target_layer = net.model[TARGET_LAYER_INDEX]

print("\nModel:", MODEL_PATH)
print("Target layer:", TARGET_LAYER_INDEX)
print("Target layer type:", type(target_layer).__name__)
print("Output directory:", OUTPUT_DIR)

# ============================================================
# IMAGE LIST
# ============================================================

image_extensions = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

images = sorted(
    [
        p for p in Path(SOURCE_DIR).iterdir()
        if p.suffix.lower() in image_extensions
    ]
)

images = images[:MAX_IMAGES]

print("\nImages to process:", len(images))


# ============================================================
# PROCESS IMAGES
# ============================================================

for idx, image_path in enumerate(images, 1):

    print(
        f"\n[{idx}/{len(images)}] "
        f"{image_path.name}"
    )

    image = cv2.imread(str(image_path))

    if image is None:
        print("Could not read image")
        continue

    original = image.copy()

    orig_h, orig_w = image.shape[:2]

    # --------------------------------------------------------
    # LETTERBOX
    # --------------------------------------------------------

    letterbox = LetterBox(
        new_shape=(IMG_SIZE, IMG_SIZE),
        auto=False,
        stride=32,
        center=True,
        padding_value=114
    )

    padded = letterbox({"img": image})["img"]

    # BGR -> RGB
    padded_rgb = cv2.cvtColor(
        padded,
        cv2.COLOR_BGR2RGB
    )

    tensor = (
        torch.from_numpy(padded_rgb)
        .permute(2, 0, 1)
        .float()
        / 255.0
    )

    tensor = tensor.unsqueeze(0).to(device)

    tensor.requires_grad_(True)

    # --------------------------------------------------------
    # GET NORMAL YOLO DETECTIONS
    # --------------------------------------------------------

    with torch.no_grad():

        results = yolo.predict(
            source=image,
            imgsz=IMG_SIZE,
            conf=CONF,
            device=device,
            verbose=False
        )

    result = results[0]

    if result.boxes is not None and len(result.boxes) > 0:

        confidences = result.boxes.conf.cpu().numpy()
        classes = result.boxes.cls.cpu().numpy().astype(int)

        best_idx = np.argmax(confidences)

        best_conf = float(confidences[best_idx])
        best_cls = int(classes[best_idx])

        class_name = yolo.names[best_cls]

        print(
            f"Detection: {class_name} "
            f"({best_conf:.3f})"
        )

    else:

        print("No detection found")

    # --------------------------------------------------------
    # EIGENCAM
    # --------------------------------------------------------

    try:

        # Clear previous gradients
        net.zero_grad(set_to_none=True)

        cam = EigenCAM(
            model=wrapper,
            target_layers=[target_layer]
        )

        grayscale_cam = cam(
            input_tensor=tensor
        )[0]

        # ----------------------------------------------------
        # REMOVE LETTERBOX
        # ----------------------------------------------------

        grayscale_cam = remove_letterbox(
            grayscale_cam,
            (orig_h, orig_w),
            (IMG_SIZE, IMG_SIZE)
        )

        # Normalize
        grayscale_cam = np.maximum(
            grayscale_cam,
            0
        )

        if grayscale_cam.max() > 0:
            grayscale_cam = (
                grayscale_cam /
                grayscale_cam.max()
            )

        # ----------------------------------------------------
        # HEATMAP
        # ----------------------------------------------------

        heatmap = cv2.applyColorMap(
            np.uint8(255 * grayscale_cam),
            cv2.COLORMAP_JET
        )

        heatmap = cv2.cvtColor(
            heatmap,
            cv2.COLOR_BGR2RGB
        )

        original_rgb = cv2.cvtColor(
            original,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # OVERLAY
        # ----------------------------------------------------

        overlay = (
            0.55 * original_rgb +
            0.45 * heatmap
        )

        overlay = np.uint8(
            np.clip(overlay, 0, 255)
        )

        # ----------------------------------------------------
        # SAVE
        # ----------------------------------------------------

        output_path = (
            Path(OUTPUT_DIR) /
            f"{image_path.stem}_eigencam.jpg"
        )

        cv2.imwrite(
            str(output_path),
            cv2.cvtColor(
                overlay,
                cv2.COLOR_RGB2BGR
            )
        )

        print(
            "Saved:",
            output_path
        )

    except Exception as e:

        print(
            "EigenCAM failed:",
            repr(e)
        )


print("\n===================================")
print("EigenCAM completed")
print("===================================")
print("Output:", OUTPUT_DIR)
