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
    "data/newdata/test/gradcam_v2"
)

IMG_SIZE = 832
CONF = 0.10
DEVICE = "cuda:0"

# FIRST TEST ONLY
MAX_IMAGES = 1

# IoU used for matching raw predictions to YOLO detections
MIN_MATCH_IOU = 0.30


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

yolo = YOLO(WEIGHTS)

net = yolo.model.to(DEVICE)
net.eval()

for p in net.parameters():
    p.requires_grad_(True)

detect_head = net.model[-1]

print("Model loaded.")
print("Classes:", net.nc)
print("Detection head:", type(detect_head))
print("Scales:", detect_head.nl)


# ============================================================
# LETTERBOX
# ============================================================

letterbox = LetterBox(
    new_shape=(IMG_SIZE, IMG_SIZE),
    auto=False,
    stride=32,
    center=True,
    padding_value=114,
)


# ============================================================
# IOU
# ============================================================

def box_iou_one_to_many(box, boxes):
    """
    box:   [4]
    boxes: [N,4]

    xyxy
    """

    if boxes.numel() == 0:
        return torch.empty(
            0,
            device=boxes.device
        )

    x1 = torch.maximum(
        boxes[:, 0],
        box[0]
    )

    y1 = torch.maximum(
        boxes[:, 1],
        box[1]
    )

    x2 = torch.minimum(
        boxes[:, 2],
        box[2]
    )

    y2 = torch.minimum(
        boxes[:, 3],
        box[3]
    )

    inter_w = (
        x2 - x1
    ).clamp(min=0)

    inter_h = (
        y2 - y1
    ).clamp(min=0)

    inter = (
        inter_w * inter_h
    )

    area_a = (
        box[2] - box[0]
    ).clamp(min=0) * (
        box[3] - box[1]
    ).clamp(min=0)

    area_b = (
        boxes[:, 2] - boxes[:, 0]
    ).clamp(min=0) * (
        boxes[:, 3] - boxes[:, 1]
    ).clamp(min=0)

    union = (
        area_a +
        area_b -
        inter
    ).clamp(min=1e-7)

    return inter / union


# ============================================================
# DECODE RAW BOXES
# ============================================================

def decode_raw_boxes(head, raw_boxes):
    """
    Decode YOLO raw DFL box predictions.

    IMPORTANT:
    Box decoding is only used for matching the raw prediction
    to the final YOLO detection. We do NOT need gradients
    through this operation.

    Therefore the entire decoding operation is performed
    under torch.no_grad().
    """

    print()
    print("Decoding raw boxes...")

    print(
        "raw_boxes:",
        tuple(raw_boxes.shape)
    )

    # --------------------------------------------------------
    # Everything here is ONLY for spatial matching.
    # No gradient is required.
    # --------------------------------------------------------

    with torch.no_grad():

        # 64 channels -> 4 distance values
        dfl_boxes = head.dfl(
            raw_boxes
        )

        print(
            "After DFL:",
            tuple(dfl_boxes.shape)
        )

        # ----------------------------------------------------
        # IMPORTANT:
        #
        # anchors and strides were generated during normal
        # inference and may therefore be "inference tensors".
        #
        # clone() creates normal tensors that can safely
        # participate in subsequent operations.
        # ----------------------------------------------------

        anchors = (
            head.anchors
            .detach()
            .clone()
            .to(raw_boxes.device)
        )

        strides = (
            head.strides
            .detach()
            .clone()
            .to(raw_boxes.device)
        )

        print(
            "Anchors:",
            tuple(anchors.shape)
        )

        print(
            "Strides:",
            tuple(strides.shape)
        )

        if anchors.ndim == 2:
            anchors = anchors.unsqueeze(0)

        if strides.ndim == 1:
            strides = strides.view(
                1,
                1,
                -1
            )

        elif strides.ndim == 2:
            strides = strides.unsqueeze(0)

        # ----------------------------------------------------
        # Decode DFL distances into xyxy boxes
        # ----------------------------------------------------

        decoded = head.decode_bboxes(
            dfl_boxes,
            anchors
        )

        decoded = (
            decoded *
            strides
        )

        # Explicitly detach because these boxes are only used
        # for matching.
        decoded = decoded.detach()

    print(
        "Decoded:",
        tuple(decoded.shape)
    )

    return decoded


# ============================================================
# MAP FLAT LOCATION TO FEATURE SCALE
# ============================================================

def location_to_scale(location):

    p3 = 104 * 104
    p4 = 52 * 52
    p5 = 26 * 26

    if location < p3:

        return (
            0,
            location
        )

    elif location < p3 + p4:

        return (
            1,
            location - p3
        )

    else:

        return (
            2,
            location - p3 - p4
        )


# ============================================================
# FIND RESPONSIBLE RAW PREDICTION
# ============================================================

def find_matching_location(
    raw_scores,
    decoded_boxes,
    class_id,
    target_box,
):

    """
    Find the raw YOLO location that corresponds
    to a final YOLO detection.

    We don't use the image center.

    We don't simply use the strongest class score.

    Instead:

        1. select this class
        2. rank candidates by class confidence
        3. calculate IoU against final detection box
        4. choose the best spatial match
    """

    class_logits = raw_scores[
        0,
        class_id
    ]

    class_conf = (
        class_logits
        .sigmoid()
    )

    boxes = decoded_boxes[
        0
    ].transpose(
        0,
        1
    )

    target_box = torch.tensor(
        target_box,
        dtype=torch.float32,
        device=boxes.device
    )

    # --------------------------------------------------------
    # Ignore extremely weak class responses.
    # --------------------------------------------------------

    valid = (
        class_conf >= 0.001
    )

    valid_indices = torch.where(
        valid
    )[0]

    if len(valid_indices) == 0:

        raise RuntimeError(
            "No valid raw class predictions."
        )

    candidate_scores = class_conf[
        valid_indices
    ]

    # --------------------------------------------------------
    # Keep strongest candidates.
    #
    # This avoids calculating IoU against every location
    # unnecessarily while still giving us a large search.
    # --------------------------------------------------------

    k = min(
        2000,
        len(valid_indices)
    )

    top_conf, top_order = torch.topk(
        candidate_scores,
        k=k
    )

    candidate_indices = (
        valid_indices[
            top_order
        ]
    )

    candidate_boxes = boxes[
        candidate_indices
    ]

    # --------------------------------------------------------
    # Calculate IoU with actual final YOLO box.
    # --------------------------------------------------------

    ious = box_iou_one_to_many(
        target_box,
        candidate_boxes
    )

    # --------------------------------------------------------
    # Primary criterion:
    #
    # highest IoU.
    #
    # Secondary criterion:
    # class confidence.
    #
    # IoU is weighted much more heavily because we want
    # the location corresponding to THIS detection.
    # --------------------------------------------------------

    metric = (
        ious +
        0.05 * top_conf
    )

    best = int(
        torch.argmax(
            metric
        ).item()
    )

    location = int(
        candidate_indices[best]
        .item()
    )

    matched_iou = float(
        ious[best]
        .detach()
        .cpu()
        .item()
    )

    matched_conf = float(
        top_conf[best]
        .detach()
        .cpu()
        .item()
    )

    matched_box = (
        candidate_boxes[best]
        .detach()
        .cpu()
        .numpy()
    )

    return (
        location,
        matched_iou,
        matched_conf,
        matched_box
    )


# ============================================================
# IMAGE LIST
# ============================================================

extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
)

image_files = sorted(
    os.path.join(
        IMAGE_DIR,
        f
    )
    for f in os.listdir(
        IMAGE_DIR
    )
    if f.lower().endswith(
        extensions
    )
)

print(
    "Images found:",
    len(image_files)
)

if MAX_IMAGES is not None:
    image_files = image_files[
        :MAX_IMAGES
    ]

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# MAIN LOOP
# ============================================================

for image_num, image_path in enumerate(
    image_files,
    1
):

    print()
    print("=" * 80)

    print(
        f"[{image_num}/{len(image_files)}]",
        os.path.basename(image_path)
    )

    print("=" * 80)

    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "Could not read image."
        )

        continue

    original = image.copy()

    h0, w0 = image.shape[:2]


    # ========================================================
    # NORMAL YOLO PREDICTION
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

    n = len(
        result.boxes
    )

    print(
        "Detections:",
        n
    )

    if n == 0:

        print(
            "No detections."
        )

        continue


    # --------------------------------------------------------
    # Print detections
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
            f"  DET {i}: "
            f"{yolo.names[cls]} "
            f"{conf:.4f} "
            f"{box.astype(int).tolist()}"
        )


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
        2,
        0,
        1
    )

    padded = np.ascontiguousarray(
        padded
    )

    tensor = torch.from_numpy(
        padded
    ).float()

    tensor /= 255.0

    tensor = tensor.unsqueeze(
        0
    ).to(DEVICE)

    tensor.requires_grad_(True)


    # ========================================================
    # CAPTURE DETECT HEAD INPUT FEATURES
    # ========================================================

    captured_feats = [
        None,
        None,
        None
    ]

    def capture_features(
        module,
        inputs
    ):

        if not inputs:
            return

        x = inputs[0]

        if not isinstance(
            x,
            (list, tuple)
        ):
            return

        for i in range(
            min(3, len(x))
        ):

            if torch.is_tensor(
                x[i]
            ):

                captured_feats[i] = x[i]

                if x[i].requires_grad:
                    x[i].retain_grad()


    hook = (
        detect_head
        .register_forward_pre_hook(
            capture_features
        )
    )


    # ========================================================
    # RAW FORWARD
    # ========================================================

    net.zero_grad(
        set_to_none=True
    )

    old_net_training = net.training
    old_head_training = (
        detect_head.training
    )

    net.eval()
    detect_head.train()

    try:

        with torch.inference_mode(False):

            with torch.enable_grad():

                raw = net(
                    tensor
                )

    finally:

        detect_head.train(
            old_head_training
        )

        net.train(
            old_net_training
        )

        hook.remove()


    raw_scores = raw[
        "scores"
    ]

    raw_boxes = raw[
        "boxes"
    ]

    print()
    print(
        "Raw scores:",
        tuple(raw_scores.shape)
    )

    print(
        "Raw boxes:",
        tuple(raw_boxes.shape)
    )


    # ========================================================
    # DECODE RAW BOXES
    # ========================================================

    decoded_boxes = (
        decode_raw_boxes(
            detect_head,
            raw_boxes
        )
    )


    # ========================================================
    # PROCESS EVERY DETECTION
    # ========================================================

    for det_index in range(n):

        print()
        print("-" * 70)

        target_class = int(
            result.boxes.cls[
                det_index
            ]
            .detach()
            .cpu()
            .item()
        )

        target_name = yolo.names[
            target_class
        ]

        target_conf = float(
            result.boxes.conf[
                det_index
            ]
            .detach()
            .cpu()
            .item()
        )

        target_box = (
            result.boxes.xyxy[
                det_index
            ]
            .detach()
            .cpu()
            .numpy()
        )


        print(
            f"Detection {det_index}: "
            f"{target_name} "
            f"{target_conf:.4f}"
        )

        print(
            "Final box:",
            target_box.astype(
                int
            ).tolist()
        )


        # ====================================================
        # MATCH FINAL BOX TO RAW LOCATION
        # ====================================================

        (
            raw_location,
            matched_iou,
            matched_conf,
            matched_box
        ) = find_matching_location(
            raw_scores,
            decoded_boxes,
            target_class,
            target_box
        )


        print()
        print(
            "MATCHED RAW LOCATION:",
            raw_location
        )

        print(
            "Raw class confidence:",
            f"{matched_conf:.6f}"
        )

        print(
            "IoU with final YOLO box:",
            f"{matched_iou:.6f}"
        )

        print(
            "Decoded raw box:",
            matched_box.astype(
                int
            ).tolist()
        )


        # ====================================================
        # SCALE
        # ====================================================

        scale_id, local_location = (
            location_to_scale(
                raw_location
            )
        )

        feature = captured_feats[
            scale_id
        ]

        if feature is None:

            print(
                "ERROR: feature missing."
            )

            continue


        fh = feature.shape[2]
        fw = feature.shape[3]

        gy = (
            local_location // fw
        )

        gx = (
            local_location % fw
        )

        print(
            "Scale:",
            scale_id
        )

        print(
            "Feature:",
            tuple(feature.shape)
        )

        print(
            "Grid:",
            gx,
            gy
        )


        # ====================================================
        # TARGET RAW CLASS LOGIT
        # ====================================================

        target = raw_scores[
            0,
            target_class,
            raw_location
        ]

        print(
            "Raw class logit:",
            float(
                target
                .detach()
                .cpu()
                .item()
            )
        )

        print(
            "Sigmoid:",
            float(
                target.sigmoid()
                .detach()
                .cpu()
                .item()
            )
        )


        # ====================================================
        # BACKPROP
        # ====================================================

        net.zero_grad(
            set_to_none=True
        )

        target.backward(
            retain_graph=(
                det_index < n - 1
            )
        )


        feature_grad = (
            feature.grad
        )

        if feature_grad is None:

            print(
                "ERROR: no feature gradient."
            )

            continue


        # ====================================================
        # GRAD-CAM
        # ====================================================

        activation = feature[
            0
        ]

        gradient = feature_grad[
            0
        ]

        weights = gradient.mean(
            dim=(1, 2),
            keepdim=True
        )

        cam = (
            weights *
            activation
        ).sum(
            dim=0
        )

        cam = torch.relu(
            cam
        )

        cam = (
            cam
            .detach()
            .cpu()
            .numpy()
        )


        # ====================================================
        # NORMALIZE
        # ====================================================

        cam_min = cam.min()
        cam_max = cam.max()

        print(
            "CAM:",
            float(cam_min),
            "->",
            float(cam_max)
        )

        if (
            cam_max - cam_min
            < 1e-12
        ):

            print(
                "WARNING: CAM is zero."
            )

            continue

        cam = (
            cam - cam_min
        ) / (
            cam_max - cam_min
        )


        # ====================================================
        # UPSCALE TO 832
        # ====================================================

        cam = cv2.resize(
            cam,
            (
                IMG_SIZE,
                IMG_SIZE
            ),
            interpolation=cv2.INTER_LINEAR
        )


        # ====================================================
        # REMOVE LETTERBOX
        # ====================================================

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


        # ====================================================
        # HEATMAP
        # ====================================================

        heatmap = cv2.applyColorMap(
            np.uint8(
                cam * 255
            ),
            cv2.COLORMAP_JET
        )


        # ====================================================
        # OVERLAY
        # ====================================================

        overlay = cv2.addWeighted(
            original,
            0.60,
            heatmap,
            0.40,
            0
        )


        # ====================================================
        # DRAW ALL YOLO BOXES
        # ====================================================

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

            thickness = (
                3
                if i == det_index
                else 1
            )

            cv2.rectangle(
                overlay,
                (x1, y1),
                (x2, y2),
                (0, 255, 0),
                thickness
            )

            cv2.putText(
                overlay,
                f"{yolo.names[cls]} {conf:.2f}",
                (
                    x1,
                    max(
                        20,
                        y1 - 8
                    )
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )


        # ====================================================
        # SAVE
        # ====================================================

        stem = os.path.splitext(
            os.path.basename(
                image_path
            )
        )[0]

        output_name = (
            f"{stem}"
            f"_det{det_index}"
            f"_{target_name}"
            f"_{target_conf:.3f}"
            f"_cam.jpg"
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            output_name
        )

        cv2.imwrite(
            output_path,
            overlay
        )

        print()
        print(
            "SAVED:",
            output_path
        )


print()
print("=" * 80)
print("FINISHED")
print("=" * 80)
print(
    "Output:",
    OUTPUT_DIR
)
