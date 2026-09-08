import os
import cv2
import torch
import numpy as np

from ultralytics import YOLO
from ultralytics.data.augment import LetterBox


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
    "data/newdata/test/gradcam_proper"
)

IMG_SIZE = 832
CONF = 0.10
DEVICE = "cuda:0"

MAX_IMAGES = 1


# ============================================================
# LOAD
# ============================================================

print("Loading model...")

yolo = YOLO(WEIGHTS)

net = yolo.model
net.to(DEVICE)
net.eval()

for p in net.parameters():
    p.requires_grad_(True)

print("Model loaded.")
print("Classes:", net.nc)

detect_head = net.model[-1]

print("Detection head:", type(detect_head))
print("Detection scales:", detect_head.nl)


# ============================================================
# LETTERBOX
# ============================================================

letterbox = LetterBox(
    new_shape=(IMG_SIZE, IMG_SIZE),
    auto=False,
    stride=32,
    center=True,
    padding_value=114
)


# ============================================================
# FIND IMAGES
# ============================================================

extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

image_files = sorted(
    os.path.join(IMAGE_DIR, f)
    for f in os.listdir(IMAGE_DIR)
    if f.lower().endswith(extensions)
)

print("Images found:", len(image_files))

if MAX_IMAGES is not None:
    image_files = image_files[:MAX_IMAGES]

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# PROCESS
# ============================================================

for image_number, image_path in enumerate(
    image_files,
    start=1
):

    print()
    print("=" * 75)
    print(
        f"[{image_number}/{len(image_files)}] "
        f"{os.path.basename(image_path)}"
    )
    print("=" * 75)

    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    image = cv2.imread(image_path)

    if image is None:
        print("ERROR: Cannot read image.")
        continue

    original = image.copy()

    h0, w0 = image.shape[:2]


    # --------------------------------------------------------
    # NORMAL YOLO PREDICTION
    # --------------------------------------------------------

    print("Running YOLO...")

    with torch.no_grad():

        result = yolo.predict(
            source=image,
            imgsz=IMG_SIZE,
            conf=CONF,
            device=DEVICE,
            verbose=False,
            rect=False
        )[0]

    n = len(result.boxes)

    print("Detections:", n)

    if n == 0:

        print(
            "No detections. "
            "This version needs a predicted detection "
            "for the first test."
        )

        continue


    # --------------------------------------------------------
    # PRINT ALL DETECTIONS
    # --------------------------------------------------------

    for i in range(n):

        cls = int(
            result.boxes.cls[i]
            .detach()
            .cpu()
            .item()
        )

        conf = float(
            result.boxes.conf[i]
            .detach()
            .cpu()
            .item()
        )

        box = (
            result.boxes.xyxy[i]
            .detach()
            .cpu()
            .numpy()
        )

        print(
            f"  {i}: "
            f"{yolo.names[cls]} "
            f"{conf:.3f} "
            f"{box.astype(int).tolist()}"
        )


    # --------------------------------------------------------
    # SELECT STRONGEST DETECTION
    # --------------------------------------------------------

    confidences = (
        result.boxes.conf
        .detach()
        .cpu()
        .numpy()
    )

    det_index = int(
        np.argmax(confidences)
    )

    target_class = int(
        result.boxes.cls[det_index]
        .detach()
        .cpu()
        .item()
    )

    target_name = yolo.names[
        target_class
    ]

    target_conf = float(
        result.boxes.conf[det_index]
        .detach()
        .cpu()
        .item()
    )

    target_box = (
        result.boxes.xyxy[det_index]
        .detach()
        .cpu()
        .numpy()
    )

    print()
    print("TARGET")
    print("Class:", target_name)
    print("Class ID:", target_class)
    print("Confidence:", target_conf)
    print("Box:", target_box.astype(int).tolist())


    # ========================================================
    # PREPROCESS
    # ========================================================

    padded = letterbox(
        {"img": image}
    )["img"]

    padded = cv2.cvtColor(
        padded,
        cv2.COLOR_BGR2RGB
    )

    padded = padded.transpose(
        2, 0, 1
    )

    padded = np.ascontiguousarray(
        padded
    )

    tensor = torch.from_numpy(
        padded
    ).float()

    tensor /= 255.0

    tensor = tensor.unsqueeze(0)
    tensor = tensor.to(DEVICE)
    tensor.requires_grad_(True)


    # ========================================================
    # CAPTURE THE THREE FEATURE MAPS
    # ========================================================

    captured_feats = [None, None, None]

    handles = []

    # The Detect head receives the three feature maps.
    #
    # We hook the last convolutional feature immediately
    # before the detection branches.
    #
    # These correspond to:
    #
    #   P3 = 104x104
    #   P4 = 52x52
    #   P5 = 26x26
    #
    # We capture them directly from the Detect head input.

    def detect_input_hook(module, inputs):

        nonlocal_dummy = None

        if len(inputs) == 0:
            return

        x = inputs[0]

        if isinstance(x, (list, tuple)):

            for i in range(
                min(3, len(x))
            ):

                if torch.is_tensor(x[i]):

                    captured_feats[i] = x[i]

                    if x[i].requires_grad:
                        x[i].retain_grad()


    handles.append(
        detect_head.register_forward_pre_hook(
            detect_input_hook
        )
    )


    # ========================================================
    # RAW FORWARD
    # ========================================================

    net.zero_grad(
        set_to_none=True
    )

    print()
    print(
        "Running raw YOLO forward..."
    )

    old_net_state = net.training
    old_head_state = detect_head.training

    net.eval()
    detect_head.train()

    try:

        with torch.inference_mode(False):

            with torch.enable_grad():

                raw = net(tensor)

    finally:

        detect_head.train(
            old_head_state
        )

        net.train(
            old_net_state
        )


    # ========================================================
    # REMOVE HOOKS
    # ========================================================

    for handle in handles:
        handle.remove()


    # ========================================================
    # CHECK FEATURES
    # ========================================================

    print()
    print("Captured feature maps:")

    for i, feat in enumerate(
        captured_feats
    ):

        if feat is None:

            print(
                f"  P{i+3}: NONE"
            )

        else:

            print(
                f"  P{i+3}:",
                tuple(feat.shape),
                "grad:",
                feat.requires_grad
            )


    # ========================================================
    # RAW SCORE INFORMATION
    # ========================================================

    scores = raw["scores"]

    print()
    print(
        "Raw scores:",
        tuple(scores.shape)
    )


    # ========================================================
    # FIND THE RAW LOCATION THAT BEST MATCHES THE PREDICTED
    # CLASS
    # ========================================================

    #
    # IMPORTANT:
    #
    # Instead of assuming the box center is the detection
    # location, we search all 14,196 locations for the
    # strongest response of the detected class.
    #

    class_scores = scores[
        0,
        target_class
    ]

    raw_location = int(
        torch.argmax(
            class_scores
        ).item()
    )

    raw_score = float(
        class_scores[
            raw_location
        ]
        .detach()
        .cpu()
        .item()
    )

    print()
    print(
        "Strongest raw response for",
        target_name
    )

    print(
        "Location:",
        raw_location
    )

    print(
        "Raw score:",
        raw_score
    )


    # ========================================================
    # DETERMINE SCALE
    # ========================================================

    sizes = [
        104 * 104,
        52 * 52,
        26 * 26
    ]

    if raw_location < sizes[0]:

        scale_id = 0
        local_location = raw_location

    elif raw_location < sizes[0] + sizes[1]:

        scale_id = 1

        local_location = (
            raw_location - sizes[0]
        )

    else:

        scale_id = 2

        local_location = (
            raw_location
            - sizes[0]
            - sizes[1]
        )


    feature = captured_feats[
        scale_id
    ]

    if feature is None:

        print(
            "ERROR: Appropriate feature map missing."
        )

        continue


    # ========================================================
    # SPATIAL LOCATION
    # ========================================================

    fh = feature.shape[2]
    fw = feature.shape[3]

    gy = (
        local_location // fw
    )

    gx = (
        local_location % fw
    )

    print()
    print("Selected scale:", scale_id)
    print(
        "Feature size:",
        fh,
        "x",
        fw
    )

    print(
        "Grid location:",
        gx,
        gy
    )


    # ========================================================
    # TARGET CLASS SCORE
    # ========================================================

    target = scores[
        0,
        target_class,
        raw_location
    ]

    print()
    print(
        "Backpropagating target:",
        float(
            target.detach().cpu()
        )
    )


    # ========================================================
    # BACKWARD
    # ========================================================

    net.zero_grad(
        set_to_none=True
    )

    target.backward()

    print(
        "Backward complete."
    )


    # ========================================================
    # GET CORRECT SCALE GRADIENT
    # ========================================================

    feature_grad = feature.grad

    if feature_grad is None:

        print(
            "ERROR: No gradient on selected feature map."
        )

        continue

    print(
        "Feature gradient:",
        tuple(feature_grad.shape)
    )


    # ========================================================
    # GRAD-CAM
    # ========================================================

    activation = feature[0]
    gradient = feature_grad[0]

    weights = gradient.mean(
        dim=(1, 2),
        keepdim=True
    )

    cam = (
        weights * activation
    ).sum(
        dim=0
    )

    cam = torch.relu(
        cam
    )

    cam = cam.detach().cpu().numpy()


    # ========================================================
    # NORMALIZE
    # ========================================================

    mn = cam.min()
    mx = cam.max()

    print(
        "CAM min/max:",
        float(mn),
        float(mx)
    )

    if mx - mn < 1e-10:

        print(
            "WARNING: CAM is zero."
        )

        continue

    cam = (
        cam - mn
    ) / (
        mx - mn
    )


    # ========================================================
    # RESIZE TO 832
    # ========================================================

    cam = cv2.resize(
        cam,
        (IMG_SIZE, IMG_SIZE),
        interpolation=cv2.INTER_LINEAR
    )


    # ========================================================
    # REMOVE LETTERBOX
    # ========================================================

    scale = min(
        IMG_SIZE / w0,
        IMG_SIZE / h0
    )

    nw = int(
        round(w0 * scale)
    )

    nh = int(
        round(h0 * scale)
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
        (w0, h0),
        interpolation=cv2.INTER_LINEAR
    )


    # ========================================================
    # HEATMAP
    # ========================================================

    heatmap = cv2.applyColorMap(
        np.uint8(
            cam * 255
        ),
        cv2.COLORMAP_JET
    )


    # ========================================================
    # OVERLAY
    # ========================================================

    overlay = cv2.addWeighted(
        original,
        0.55,
        heatmap,
        0.45,
        0
    )


    # ========================================================
    # DRAW ALL DETECTIONS
    # ========================================================

    for i in range(n):

        box = (
            result.boxes.xyxy[i]
            .detach()
            .cpu()
            .numpy()
            .astype(int)
        )

        cls = int(
            result.boxes.cls[i]
            .detach()
            .cpu()
            .item()
        )

        conf = float(
            result.boxes.conf[i]
            .detach()
            .cpu()
            .item()
        )

        x1, y1, x2, y2 = box

        cv2.rectangle(
            overlay,
            (x1, y1),
            (x2, y2),
            (0, 255, 0),
            2
        )

        cv2.putText(
            overlay,
            f"{yolo.names[cls]} {conf:.2f}",
            (
                x1,
                max(20, y1 - 8)
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )


    # ========================================================
    # SAVE
    # ========================================================

    stem = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    output_path = os.path.join(
        OUTPUT_DIR,
        stem
        + "_"
        + target_name
        + "_proper_cam.jpg"
    )

    cv2.imwrite(
        output_path,
        overlay
    )

    print()
    print(
        "Saved:",
        output_path
    )


print()
print("=" * 75)
print("DONE")
print("=" * 75)

print(
    "Output directory:",
    OUTPUT_DIR
)
