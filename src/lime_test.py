import os
import cv2
import numpy as np
import torch

from ultralytics import YOLO
from lime.lime_image import LimeImageExplainer
from skimage.segmentation import slic


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

# Process first 50 images
MAX_IMAGES = 50

OUTPUT_DIR = (
    "/home/admin1/Abishek-Intern/new_atr/"
    "data/newdata/test/lime_positive"
)

DEVICE = "cuda:0"

IMG_SIZE = 832

# Detection threshold for perturbed images
PRED_CONF = 0.001

# Detection threshold for original image
TARGET_CONF = 0.10

# LIME parameters
NUM_SAMPLES = 500
BATCH_SIZE = 8

# SLIC parameters
NUM_SEGMENTS = 60
COMPACTNESS = 10
SIGMA = 1


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("LOADING YOLO MODEL")
print("=" * 70)

yolo = YOLO(WEIGHTS)

print("Model loaded.")
print("Classes:", yolo.model.nc)
print()


# ============================================================
# GET IMAGE LIST
# ============================================================

image_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

image_paths = [
    os.path.join(IMAGE_DIR, f)
    for f in sorted(os.listdir(IMAGE_DIR))
    if f.lower().endswith(image_extensions)
]

if len(image_paths) == 0:
    raise RuntimeError(
        "No images found in:\n" + IMAGE_DIR
    )


# Take only first MAX_IMAGES
image_paths = image_paths[:MAX_IMAGES]

print("=" * 70)
print("IMAGE LIST")
print("=" * 70)

print("Images found:", len(image_paths))
print("Images to process:", len(image_paths))
print()


# ============================================================
# IOU
# ============================================================

def calculate_iou(box_a, box_b):

    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    x1 = max(ax1, bx1)
    y1 = max(ay1, by1)

    x2 = min(ax2, bx2)
    y2 = min(ay2, by2)

    iw = max(0, x2 - x1)
    ih = max(0, y2 - y1)

    intersection = iw * ih

    area_a = (
        max(0, ax2 - ax1)
        *
        max(0, ay2 - ay1)
    )

    area_b = (
        max(0, bx2 - bx1)
        *
        max(0, by2 - by1)
    )

    union = (
        area_a
        +
        area_b
        -
        intersection
    )

    if union <= 0:
        return 0.0

    return intersection / union


# ============================================================
# GLOBAL TARGET VARIABLES
# ============================================================

target_class = None
target_box = None


# ============================================================
# TARGET SCORE
# ============================================================

def get_target_score(result):

    """
    Find the same target class whose bounding box
    overlaps the ORIGINAL target box.

    Target score:

        confidence × IoU
    """

    best_score = 0.0
    best_conf = 0.0
    best_iou = 0.0

    if target_class is None or target_box is None:
        return (
            best_score,
            best_conf,
            best_iou
        )

    for i in range(len(result.boxes)):

        cls = int(
            result.boxes.cls[i]
            .cpu()
            .item()
        )

        # Must be the same aircraft class
        if cls != target_class:
            continue

        conf = float(
            result.boxes.conf[i]
            .cpu()
            .item()
        )

        box = (
            result.boxes.xyxy[i]
            .cpu()
            .numpy()
        )

        iou = calculate_iou(
            target_box,
            box
        )

        score = conf * iou

        if score > best_score:

            best_score = score
            best_conf = conf
            best_iou = iou

    return (
        best_score,
        best_conf,
        best_iou
    )


# ============================================================
# PROCESS EACH IMAGE
# ============================================================

successful = 0
skipped = 0
failed = 0


for image_number, image_path in enumerate(
    image_paths,
    start=1
):

    print()
    print()
    print("#" * 80)
    print(
        f"IMAGE {image_number}/{len(image_paths)}"
    )
    print("#" * 80)

    print(
        "File:",
        os.path.basename(image_path)
    )


    # ========================================================
    # LOAD IMAGE
    # ========================================================

    image_bgr = cv2.imread(
        image_path
    )

    if image_bgr is None:

        print("ERROR: Could not read image.")
        failed += 1
        continue


    image_rgb = cv2.cvtColor(
        image_bgr,
        cv2.COLOR_BGR2RGB
    )

    h, w = image_rgb.shape[:2]

    print(
        "Image size:",
        w,
        "x",
        h
    )


    # ========================================================
    # ORIGINAL YOLO PREDICTION
    # ========================================================

    print(
        "Running original YOLO..."
    )

    try:

        with torch.no_grad():

            original_result = yolo.predict(
                source=image_bgr,
                imgsz=IMG_SIZE,
                conf=TARGET_CONF,
                device=DEVICE,
                verbose=False,
                rect=False,
            )[0]

    except Exception as e:

        print(
            "YOLO ERROR:",
            str(e)
        )

        failed += 1
        continue


    num_detections = len(
        original_result.boxes
    )

    print(
        "Detections:",
        num_detections
    )


    # ========================================================
    # NO DETECTIONS
    # ========================================================

    if num_detections == 0:

        print(
            "SKIPPED: No aircraft detected."
        )

        skipped += 1
        continue


    # ========================================================
    # PRINT DETECTIONS
    # ========================================================

    print()
    print("Detections:")

    for i in range(
        num_detections
    ):

        cls = int(
            original_result.boxes.cls[i]
            .cpu()
            .item()
        )

        conf = float(
            original_result.boxes.conf[i]
            .cpu()
            .item()
        )

        box = (
            original_result.boxes.xyxy[i]
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
    # SELECT STRONGEST DETECTION
    # ========================================================

    target_index = int(
        torch.argmax(
            original_result.boxes.conf
        ).item()
    )

    target_class = int(
        original_result.boxes.cls[
            target_index
        ].item()
    )

    target_name = yolo.names[
        target_class
    ]

    target_conf = float(
        original_result.boxes.conf[
            target_index
        ].item()
    )

    target_box = (
        original_result.boxes.xyxy[
            target_index
        ]
        .cpu()
        .numpy()
    )


    print()
    print("-" * 70)
    print("LIME TARGET")
    print("-" * 70)

    print(
        "Class:",
        target_name
    )

    print(
        "Class ID:",
        target_class
    )

    print(
        "Confidence:",
        f"{target_conf:.4f}"
    )

    print(
        "Box:",
        target_box.astype(int).tolist()
    )


    # ========================================================
    # ORIGINAL SCORE
    # ========================================================

    (
        original_score,
        original_conf,
        original_iou
    ) = get_target_score(
        original_result
    )

    print(
        "Original target score:",
        f"{original_score:.6f}"
    )

    print(
        "Original confidence:",
        f"{original_conf:.6f}"
    )

    print(
        "Original IoU:",
        f"{original_iou:.6f}"
    )


    # ========================================================
    # SLIC SEGMENTATION
    # ========================================================

    print()
    print(
        "Creating SLIC superpixels..."
    )

    def segmentation_fn(image):

        image = np.asarray(
            image
        )

        return slic(
            image,
            n_segments=NUM_SEGMENTS,
            compactness=COMPACTNESS,
            sigma=SIGMA,
            start_label=0,
            channel_axis=-1
        )


    segments_preview = segmentation_fn(
        image_rgb
    )

    actual_segments = len(
        np.unique(
            segments_preview
        )
    )

    print(
        "Requested segments:",
        NUM_SEGMENTS
    )

    print(
        "Actual segments:",
        actual_segments
    )


    # ========================================================
    # LIME CLASSIFIER
    # ========================================================

    call_count = 0

    def classifier_fn(images):

        nonlocal_dummy = None

        global call_count

        call_count += 1

        print(
            f"  YOLO perturbation batch "
            f"{call_count}"
        )


        # ----------------------------------------------------
        # RGB -> BGR
        # ----------------------------------------------------

        bgr_images = []

        for img in images:

            img = np.asarray(
                img
            )

            if img.dtype != np.uint8:

                img = np.clip(
                    img,
                    0,
                    255
                ).astype(
                    np.uint8
                )

            bgr = cv2.cvtColor(
                img,
                cv2.COLOR_RGB2BGR
            )

            bgr_images.append(
                bgr
            )


        # ----------------------------------------------------
        # YOLO
        # ----------------------------------------------------

        with torch.no_grad():

            results = yolo.predict(
                source=bgr_images,
                imgsz=IMG_SIZE,
                conf=PRED_CONF,
                device=DEVICE,
                verbose=False,
                rect=False
            )


        # ----------------------------------------------------
        # TARGET SCORE
        # ----------------------------------------------------

        output = []

        for result in results:

            (
                score,
                conf,
                iou
            ) = get_target_score(
                result
            )

            score = float(
                np.clip(
                    score,
                    0.0,
                    1.0
                )
            )

            # Column 0 = non-target
            # Column 1 = target

            output.append(
                [
                    1.0 - score,
                    score
                ]
            )

        return np.asarray(
            output,
            dtype=np.float32
        )


    # ========================================================
    # CREATE LIME EXPLAINER
    # ========================================================

    print()
    print(
        "Starting LIME..."
    )

    print(
        "Samples:",
        NUM_SAMPLES
    )

    print(
        "SLIC segments:",
        actual_segments
    )


    explainer = LimeImageExplainer(
        random_state=42
    )


    # ========================================================
    # RUN LIME
    # ========================================================

    try:

        explanation = (
            explainer.explain_instance(
                image_rgb,
                classifier_fn,
                labels=(1,),
                hide_color=0,
                num_samples=NUM_SAMPLES,
                batch_size=BATCH_SIZE,
                segmentation_fn=segmentation_fn
            )
        )

    except Exception as e:

        print()
        print(
            "LIME ERROR:",
            str(e)
        )

        failed += 1
        continue


    # ========================================================
    # GET LIME WEIGHTS
    # ========================================================

    weights = dict(
        explanation.local_exp[1]
    )

    segments = (
        explanation.segments
    )


    # ========================================================
    # POSITIVE-ONLY MAP
    # ========================================================

    positive_map = np.zeros(
        segments.shape,
        dtype=np.float32
    )

    for segment_id, weight in (
        weights.items()
    ):

        if weight > 0:

            positive_map[
                segments == segment_id
            ] = weight


    # ========================================================
    # NORMALIZE
    # ========================================================

    max_value = positive_map.max()

    if max_value > 1e-12:

        positive_map = (
            positive_map /
            max_value
        )

    else:

        positive_map = np.zeros_like(
            positive_map
        )


    # ========================================================
    # PRINT POSITIVE FEATURES
    # ========================================================

    positive_weights = sorted(
        [
            (
                segment_id,
                weight
            )
            for segment_id, weight
            in weights.items()
            if weight > 0
        ],
        key=lambda x: x[1],
        reverse=True
    )


    print()
    print(
        "Positive LIME segments:",
        len(positive_weights)
    )

    print(
        "Top positive segments:"
    )

    for segment_id, weight in (
        positive_weights[:10]
    ):

        print(
            f"  Segment {segment_id:3d}: "
            f"{weight:+.6f}"
        )


    # ========================================================
    # HEATMAP
    # ========================================================

    heatmap = cv2.applyColorMap(
        np.uint8(
            positive_map * 255
        ),
        cv2.COLORMAP_JET
    )


    # ========================================================
    # OVERLAY
    # ========================================================

    overlay = cv2.addWeighted(
        image_bgr,
        0.60,
        heatmap,
        0.40,
        0
    )


    # ========================================================
    # DRAW TARGET BOX
    # ========================================================

    x1, y1, x2, y2 = (
        target_box.astype(int)
    )

    cv2.rectangle(
        overlay,
        (x1, y1),
        (x2, y2),
        (0, 255, 0),
        3
    )

    cv2.putText(
        overlay,
        f"{target_name} {target_conf:.2f}",
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
    # SAVE POSITIVE LIME OUTPUT
    # ========================================================

    stem = os.path.splitext(
        os.path.basename(
            image_path
        )
    )[0]

    output_path = os.path.join(
        OUTPUT_DIR,
        stem
        + "_"
        + target_name
        + "_lime_positive.jpg"
    )


    cv2.imwrite(
        output_path,
        overlay
    )


    print()
    print(
        "SAVED:"
    )

    print(
        output_path
    )

    successful += 1


# ============================================================
# FINAL SUMMARY
# ============================================================

print()
print()
print("=" * 80)
print("LIME 50-IMAGE TEST COMPLETE")
print("=" * 80)

print(
    "Images requested:",
    len(image_paths)
)

print(
    "Successfully processed:",
    successful
)

print(
    "Skipped (no detection):",
    skipped
)

print(
    "Failed:",
    failed
)

print()
print(
    "Output directory:"
)

print(
    OUTPUT_DIR
)

print()
print("DONE")
