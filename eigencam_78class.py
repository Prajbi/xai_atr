import os
import cv2
import numpy as np
import torch
import torch.nn as nn

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================
# CONFIG
# ============================================================

WEIGHTS = (
    "/home/admin1/Abishek-Intern/new_atr/new/models/yolo_aircraft.pt"
)

SOURCE = (
    "/home/admin1/Abishek-Intern/new_atr/"
    "data/newdata/test/images"
)

OUTPUT = (
    "/home/admin1/Abishek-Intern/new_atr/new/eigencam"
)

IMG_SIZE = 832
CONF = 0.5
DEVICE = "cuda:0"


# ============================================================
# YOLO OUTPUT WRAPPER
# ============================================================

class YOLOOutputWrapper(nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):

        output = self.model(x)

        if isinstance(output, tuple):
            output = output[0]

        if isinstance(output, (list, tuple)):
            output = output[0]

        return output


# ============================================================
# SETUP
# ============================================================

os.makedirs(OUTPUT, exist_ok=True)

print("Loading model...")

model = YOLO(WEIGHTS)

print("Number of classes:", len(model.names))

yolo_model = model.model
yolo_model.eval()
yolo_model.to(DEVICE)

target_layers = [
    yolo_model.model[-4]
]

print("Target layer:")
print(target_layers[0])


cam_model = YOLOOutputWrapper(yolo_model)
cam_model.eval()
cam_model.to(DEVICE)


# ============================================================
# IMAGE LIST
# ============================================================

image_files = []

for filename in sorted(os.listdir(SOURCE)):

    if filename.lower().endswith(
        (".jpg", ".jpeg", ".png", ".bmp", ".webp")
    ):
        image_files.append(
            os.path.join(SOURCE, filename)
        )

print("Images found:", len(image_files))


# ============================================================
# PROCESS
# ============================================================

# TEST ONLY 5 IMAGES
test_images = image_files[:100]


for index, image_path in enumerate(test_images, 1):

    print(
        f"\n[{index}/{len(test_images)}] "
        f"{os.path.basename(image_path)}"
    )

    # --------------------------------------------------------
    # Read original image
    # --------------------------------------------------------

    image = cv2.imread(image_path)

    if image is None:
        print("  ERROR: Could not read image")
        continue

    original_h, original_w = image.shape[:2]


    # --------------------------------------------------------
    # Normal YOLO prediction
    # --------------------------------------------------------

    results = model.predict(
        source=image,
        imgsz=IMG_SIZE,
        conf=CONF,
        device=DEVICE,
        rect=False,
        verbose=False
    )

    result = results[0]


    # --------------------------------------------------------
    # LetterBox
    # --------------------------------------------------------

    letterbox = LetterBox(
        new_shape=(IMG_SIZE, IMG_SIZE),
        auto=False,
        stride=32,
        center=True,
        padding_value=114
    )

    padded = letterbox(
        {"img": image}
    )["img"]


    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        padded,
        cv2.COLOR_BGR2RGB
    )


    # --------------------------------------------------------
    # Normalize
    # --------------------------------------------------------

    rgb_float = (
        rgb.astype(np.float32) / 255.0
    )


    # --------------------------------------------------------
    # Tensor
    # --------------------------------------------------------

    input_tensor = torch.from_numpy(
        rgb_float.transpose(2, 0, 1)
    ).unsqueeze(0).float().to(DEVICE)


    # --------------------------------------------------------
    # EigenCAM
    # --------------------------------------------------------

    try:

        cam = EigenCAM(
            model=cam_model,
            target_layers=target_layers
        )

        grayscale_cam = cam(
            input_tensor=input_tensor
        )[0]

    except Exception as e:

        print("  EigenCAM failed:")
        print(" ", repr(e))
        continue


    # --------------------------------------------------------
    # CAM is currently 832x832
    # --------------------------------------------------------

    grayscale_cam = cv2.resize(
        grayscale_cam,
        (IMG_SIZE, IMG_SIZE)
    )


    # ========================================================
    # IMPORTANT:
    # Remove LetterBox padding
    # ========================================================

    # Calculate scaling used by LetterBox

    scale = min(
        IMG_SIZE / original_w,
        IMG_SIZE / original_h
    )

    resized_w = int(round(original_w * scale))
    resized_h = int(round(original_h * scale))


    # Padding on each side

    pad_x = (IMG_SIZE - resized_w) // 2
    pad_y = (IMG_SIZE - resized_h) // 2


    # Crop CAM to actual image region

    grayscale_cam = grayscale_cam[
        pad_y:pad_y + resized_h,
        pad_x:pad_x + resized_w
    ]


    # Resize CAM to original image size

    grayscale_cam = cv2.resize(
        grayscale_cam,
        (original_w, original_h)
    )


    # --------------------------------------------------------
    # Original image for visualization
    # --------------------------------------------------------

    original_rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    original_float = (
        original_rgb.astype(np.float32) / 255.0
    )


    # --------------------------------------------------------
    # CAM overlay on ORIGINAL image
    # --------------------------------------------------------

    visualization = show_cam_on_image(
        original_float,
        grayscale_cam,
        use_rgb=True
    )


    visualization = cv2.cvtColor(
        visualization,
        cv2.COLOR_RGB2BGR
    )


    # ========================================================
    # Draw YOLO boxes
    # ========================================================

    if result.boxes is not None:

        for box in result.boxes:

            xyxy = (
                box.xyxy[0]
                .detach()
                .cpu()
                .numpy()
                .astype(int)
            )

            cls_id = int(
                box.cls[0]
                .detach()
                .cpu()
                .item()
            )

            confidence = float(
                box.conf[0]
                .detach()
                .cpu()
                .item()
            )

            x1, y1, x2, y2 = xyxy

            class_name = model.names[cls_id]

            label = (
                f"{class_name} "
                f"{confidence:.2f}"
            )


            # Bounding box

            cv2.rectangle(
                visualization,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                2
            )


            # Label

            cv2.putText(
                visualization,
                label,
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )


    # ========================================================
    # Save
    # ========================================================

    filename = os.path.basename(
        image_path
    )

    output_path = os.path.join(
        OUTPUT,
        filename
    )

    cv2.imwrite(
        output_path,
        visualization
    )

    print("  Saved:", output_path)


print("\n========================================")
print("EigenCAM test complete")
print("Output:")
print(OUTPUT)
print("========================================")
