import os
import cv2
import torch
import numpy as np

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================
# CONFIG
# ============================================================

WEIGHTS = (
    "/home/admin1/Abishek-Intern/new_atr/"
    "runs/yolov8s_832_balanced800/weights/best.pt"
)

IMAGE_DIR = (
    "/home/admin1/Abishek-Intern/new_atr/"
    "data/newdata/test/images"
)

OUTPUT_DIR = (
    "/home/admin1/Abishek-Intern/new_atr/"
    "data/newdata/test/eigencam_minus5"
)

IMG_SIZE = 832
DEVICE = "cuda:0"
CONF = 0.10

MAX_IMAGES = 10


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

yolo = YOLO(WEIGHTS)

model = yolo.model.to(DEVICE)
model.eval()

print("Model loaded.")
print("Classes:", model.nc)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# GET FIRST 10 IMAGES
# ============================================================

image_files = sorted([
    f for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    )
])

image_files = image_files[:MAX_IMAGES]

print()
print("Images to process:", len(image_files))

for i, filename in enumerate(image_files, 1):
    print(f"{i:02d}. {filename}")


# ============================================================
# EIGENCAM TARGET LAYER
# ============================================================

layer_index = -5

layer = model.model[layer_index]

print()
print("=" * 75)
print("EigenCAM target layer:", layer_index)
print("=" * 75)

print(layer)


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


wrapped_model = YOLOWrapper(model)


# ============================================================
# PROCESS IMAGES
# ============================================================

for image_number, filename in enumerate(
    image_files,
    1
):

    print()
    print("=" * 75)
    print(
        f"Processing {image_number}/{len(image_files)}: "
        f"{filename}"
    )
    print("=" * 75)


    # --------------------------------------------------------
    # IMAGE PATH
    # --------------------------------------------------------

    image_path = os.path.join(
        IMAGE_DIR,
        filename
    )

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "WARNING: Could not read image. Skipping."
        )

        continue


    original = image.copy()

    h, w = image.shape[:2]

    print(
        f"Original image size: {w} x {h}"
    )


    # ========================================================
    # YOLO PREDICTION
    # ========================================================

    print("Running YOLO...")

    with torch.no_grad():

        result = yolo.predict(
            source=image,
            imgsz=IMG_SIZE,
            conf=CONF,
            device=DEVICE,
            verbose=False,
            rect=False,
        )[0]


    print(
        "Detections:",
        len(result.boxes)
    )


    # --------------------------------------------------------
    # PRINT DETECTIONS
    # --------------------------------------------------------

    for i in range(
        len(result.boxes)
    ):

        cls = int(
            result.boxes.cls[i]
            .cpu()
            .item()
        )

        conf = float(
            result.boxes.conf[i]
            .cpu()
            .item()
        )

        box = (
            result.boxes.xyxy[i]
            .cpu()
            .numpy()
            .astype(int)
        )

        print(
            f"  {i}: "
            f"{yolo.names[cls]} "
            f"{conf:.4f} "
            f"{box.tolist()}"
        )


    # ========================================================
    # PREPROCESS
    # ========================================================

    letterbox = LetterBox(
        new_shape=(IMG_SIZE, IMG_SIZE),
        auto=False,
        stride=32,
        center=True,
        padding_value=114,
    )

    padded = letterbox(
        {"img": image}
    )["img"]


    padded = cv2.cvtColor(
        padded,
        cv2.COLOR_BGR2RGB
    )


    padded = (
        padded.astype(
            np.float32
        ) / 255.0
    )


    input_tensor = torch.from_numpy(
        padded.transpose(2, 0, 1)
    ).unsqueeze(0).to(DEVICE)


    # ========================================================
    # EIGENCAM
    # ========================================================

    print(
        "Running EigenCAM on layer -5..."
    )

    try:

        cam = EigenCAM(
            model=wrapped_model,
            target_layers=[layer],
        )

        grayscale_cam = cam(
            input_tensor=input_tensor
        )[0]

    except Exception as e:

        print(
            "EigenCAM FAILED:"
        )

        print(
            repr(e)
        )

        continue


    # ========================================================
    # CAM STATISTICS
    # ========================================================

    print(
        "CAM shape:",
        grayscale_cam.shape
    )

    print(
        "CAM min:",
        float(
            grayscale_cam.min()
        )
    )

    print(
        "CAM max:",
        float(
            grayscale_cam.max()
        )
    )


    # ========================================================
    # REMOVE LETTERBOX
    # ========================================================

    grayscale_cam = cv2.resize(
        grayscale_cam,
        (
            IMG_SIZE,
            IMG_SIZE
        ),
        interpolation=cv2.INTER_LINEAR
    )


    scale = min(
        IMG_SIZE / w,
        IMG_SIZE / h
    )


    nw = int(
        round(w * scale)
    )

    nh = int(
        round(h * scale)
    )


    px = (
        IMG_SIZE - nw
    ) // 2

    py = (
        IMG_SIZE - nh
    ) // 2


    grayscale_cam = grayscale_cam[
        py:py + nh,
        px:px + nw
    ]


    grayscale_cam = cv2.resize(
        grayscale_cam,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )


    # ========================================================
    # OVERLAY
    # ========================================================

    rgb_original = cv2.cvtColor(
        original,
        cv2.COLOR_BGR2RGB
    )


    rgb_float = (
        rgb_original.astype(
            np.float32
        ) / 255.0
    )


    visualization = show_cam_on_image(
        rgb_float,
        grayscale_cam,
        use_rgb=True
    )


    visualization = cv2.cvtColor(
        visualization,
        cv2.COLOR_RGB2BGR
    )


    # ========================================================
    # DRAW ALL DETECTIONS
    # ========================================================

    for i in range(
        len(result.boxes)
    ):

        cls = int(
            result.boxes.cls[i]
            .cpu()
            .item()
        )

        conf = float(
            result.boxes.conf[i]
            .cpu()
            .item()
        )

        box = (
            result.boxes.xyxy[i]
            .cpu()
            .numpy()
            .astype(int)
        )


        x1, y1, x2, y2 = box


        cv2.rectangle(
            visualization,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            3
        )


        cv2.putText(
            visualization,
            f"{yolo.names[cls]} {conf:.2f}",
            (
                x1,
                max(
                    25,
                    y1 - 10
                )
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # SAVE
    # ========================================================

    stem = os.path.splitext(
        filename
    )[0]


    output_path = os.path.join(
        OUTPUT_DIR,
        f"{stem}_eigencam_minus5.jpg"
    )


    cv2.imwrite(
        output_path,
        visualization
    )


    print(
        "Saved:",
        output_path
    )


# ============================================================
# COMPLETE
# ============================================================

print()
print("=" * 75)
print("ALL 10 EIGENCAM -5 IMAGES COMPLETE")
print("=" * 75)

print(
    "Output directory:",
    OUTPUT_DIR
)
