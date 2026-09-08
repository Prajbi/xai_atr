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

IMAGE_PATH = (
    "/home/admin1/Abishek-Intern/new_atr/"
    "data/newdata/test/images/"
    "0079ecb92d4a99dc405dae5bde86d8a9_jpg.rf."
    "RYaoT97eQIP6lL2NSuiQ.jpg"
)

OUTPUT_DIR = (
    "/home/admin1/Abishek-Intern/new_atr/"
    "data/newdata/test/eigencam_layers"
)

IMG_SIZE = 832
DEVICE = "cuda:0"
CONF = 0.10


# ============================================================
# LOAD
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
# IMAGE
# ============================================================

image = cv2.imread(
    IMAGE_PATH
)

if image is None:
    raise RuntimeError(
        "Cannot read image."
    )

original = image.copy()

h, w = image.shape[:2]


# ============================================================
# YOLO PREDICTION
# ============================================================

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


# ============================================================
# PREPROCESS
# ============================================================

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


# ============================================================
# WRAPPER
# ============================================================

class YOLOWrapper(torch.nn.Module):

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):

        output = self.model(x)

        if isinstance(
            output,
            tuple
        ):
            return output[0]

        return output


wrapped_model = YOLOWrapper(
    model
)


# ============================================================
# TEST LAYERS
# ============================================================

#
# We already tested -4.
#
# Now test progressively earlier layers.
#
# The exact spatial resolution is printed automatically.
#

layer_indices = [
    -6,
    -5,
    -4,
]


# ============================================================
# HELPER: GET LAYER OUTPUT SHAPE
# ============================================================

def get_layer_shape(
    layer
):

    captured = {}

    def hook(
        module,
        inputs,
        output
    ):

        captured["output"] = output

    hdl = layer.register_forward_hook(
        hook
    )

    with torch.no_grad():

        model(
            input_tensor
        )

    hdl.remove()

    output = captured.get(
        "output"
    )

    if torch.is_tensor(
        output
    ):

        return tuple(
            output.shape
        )

    if isinstance(
        output,
        (list, tuple)
    ):

        shapes = []

        for x in output:

            if torch.is_tensor(x):
                shapes.append(
                    tuple(x.shape)
                )

        return shapes

    return str(
        type(output)
    )


# ============================================================
# LETTERBOX CROP FUNCTION
# ============================================================

def remove_letterbox(
    cam
):

    cam = cv2.resize(
        cam,
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

    cam = cam[
        py:py + nh,
        px:px + nw
    ]

    cam = cv2.resize(
        cam,
        (w, h),
        interpolation=cv2.INTER_LINEAR
    )

    return cam


# ============================================================
# RUN EACH LAYER
# ============================================================

for layer_index in layer_indices:

    print()
    print("=" * 75)

    print(
        "Testing layer:",
        layer_index
    )

    layer = model.model[
        layer_index
    ]

    print(
        layer
    )

    # --------------------------------------------------------
    # Get output shape
    # --------------------------------------------------------

    try:

        shape = get_layer_shape(
            layer
        )

        print(
            "Layer output shape:",
            shape
        )

    except Exception as e:

        print(
            "Could not determine shape:",
            e
        )

        continue


    # --------------------------------------------------------
    # EigenCAM
    # --------------------------------------------------------

    print(
        "Running EigenCAM..."
    )

    try:

        cam = EigenCAM(
            model=wrapped_model,
            target_layers=[
                layer
            ],
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


    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    print(
        "CAM shape:",
        grayscale_cam.shape
    )

    print(
        "CAM min:",
        float(grayscale_cam.min())
    )

    print(
        "CAM max:",
        float(grayscale_cam.max())
    )


    # --------------------------------------------------------
    # Remove letterbox
    # --------------------------------------------------------

    grayscale_cam = (
        remove_letterbox(
            grayscale_cam
        )
    )


    # --------------------------------------------------------
    # Overlay
    # --------------------------------------------------------

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


    # --------------------------------------------------------
    # Draw detections
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


    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    output_path = os.path.join(
        OUTPUT_DIR,
        f"layer_{layer_index}_eigencam.jpg"
    )

    cv2.imwrite(
        output_path,
        visualization
    )

    print(
        "Saved:",
        output_path
    )


print()
print("=" * 75)
print("ALL LAYER TESTS COMPLETE")
print("=" * 75)

print(
    "Output directory:",
    OUTPUT_DIR
)
