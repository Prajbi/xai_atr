from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path("/home/admin1/Abishek-Intern/new_atr")

WEIGHTS = (
    PROJECT_ROOT
    / "runs"
    / "yolov8s_832_350plus"
    / "weights"
    / "best.pt"
)

TEST_DIR = (
    PROJECT_ROOT
    / "data"
    / "filtered_350"
    / "images"
    / "test"
)

OUT_DIR = (
    PROJECT_ROOT
    / "outputs"
    / "gradcam"
)

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 832

CONF = 0.05

DEVICE = (
    "cuda:0"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# YOLO WRAPPER
# ============================================================

class YoloGradCAMWrapper(torch.nn.Module):

    def __init__(self, yolo_model):

        super().__init__()

        self.model = yolo_model.model

    def forward(self, x):

        # DO NOT use torch.no_grad().
        # GradCAM needs gradients.

        output = self.model(x)

        if isinstance(output, (tuple, list)):

            output = output[0]

        return output


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path):

    img = cv2.imread(
        str(path)
    )

    if img is None:

        raise FileNotFoundError(
            f"Could not read image:\n{path}"
        )

    img = cv2.cvtColor(
        img,
        cv2.COLOR_BGR2RGB
    )

    img = cv2.resize(
        img,
        (IMAGE_SIZE, IMAGE_SIZE)
    )

    rgb_float = (
        img.astype(np.float32)
        / 255.0
    )

    tensor = torch.from_numpy(
        rgb_float
    )

    tensor = tensor.permute(
        2,
        0,
        1
    )

    tensor = tensor.unsqueeze(0)

    tensor = tensor.float()

    return rgb_float, tensor


# ============================================================
# YOLO DETECTION TARGET
# ============================================================

class YoloDetectionTarget:

    """
    Differentiable target for YOLOv8 detection output.

    YOLOv8 output for this model is:

        [25, 14196]

    where:

        4   = bounding-box channels
        21  = class channels

    We use the strongest class activation as the scalar
    GradCAM target.
    """

    def __call__(self, model_output):

        # ----------------------------------------------------
        # Handle tuple/list
        # ----------------------------------------------------

        if isinstance(
            model_output,
            (tuple, list)
        ):

            model_output = model_output[0]


        # ----------------------------------------------------
        # Print shape for debugging
        # ----------------------------------------------------

        if model_output.ndim != 2:

            raise RuntimeError(
                "Unexpected YOLO output shape: "
                f"{model_output.shape}"
            )


        # ----------------------------------------------------
        # YOLOv8:
        #
        # [25, 14196]
        #
        # First 4 rows = box coordinates
        # Remaining 21 rows = class predictions
        # ----------------------------------------------------

        class_scores = model_output[4:, :]


        # ----------------------------------------------------
        # Select strongest class activation
        # ----------------------------------------------------

        target = class_scores.max()


        # ----------------------------------------------------
        # Make sure it is scalar
        # ----------------------------------------------------

        target = target.squeeze()


        return target


# ============================================================
# DRAW DETECTIONS
# ============================================================

def draw_detections(
    image,
    result
):

    annotated = image.copy()

    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):

        return annotated


    for box in result.boxes:

        cls_id = int(
            box.cls.item()
        )

        confidence = float(
            box.conf.item()
        )

        name = result.names[
            cls_id
        ]

        x1, y1, x2, y2 = (
            box.xyxy[0]
            .cpu()
            .numpy()
            .astype(int)
        )


        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            annotated,
            (x1, y1),
            (x2, y2),
            (255, 255, 255),
            2
        )


        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        label = (
            f"{name} "
            f"{confidence:.2f}"
        )

        cv2.putText(
            annotated,
            label,
            (x1, max(20, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 255),
            2,
            cv2.LINE_AA
        )


    return annotated


# ============================================================
# RUN GRADCAM
# ============================================================

def run_gradcam(
    image_path,
    yolo
):

    # --------------------------------------------------------
    # YOLO DETECTION
    # --------------------------------------------------------

    results = yolo.predict(
        source=str(image_path),
        imgsz=IMAGE_SIZE,
        conf=CONF,
        device=DEVICE,
        verbose=False
    )

    result = results[0]


    # --------------------------------------------------------
    # PRINT DETECTIONS
    # --------------------------------------------------------

    print(
        "\nDetections:"
    )

    if (
        result.boxes is None
        or len(result.boxes) == 0
    ):

        print(
            "  No detections."
        )

    else:

        for i, box in enumerate(
            result.boxes
        ):

            cls_id = int(
                box.cls.item()
            )

            confidence = float(
                box.conf.item()
            )

            name = result.names[
                cls_id
            ]

            print(
                f"  {i + 1}. "
                f"{name} "
                f"confidence={confidence:.4f}"
            )


    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    rgb_float, tensor = load_image(
        image_path
    )

    tensor = tensor.to(
        DEVICE
    )


    # --------------------------------------------------------
    # WRAPPER
    # --------------------------------------------------------

    wrapped = YoloGradCAMWrapper(
        yolo
    )

    wrapped = wrapped.to(
        DEVICE
    )

    wrapped.eval()


    # --------------------------------------------------------
    # TARGET LAYER
    # --------------------------------------------------------

    target_layer_idx = -4

    target_layer = (
        wrapped.model.model[
            target_layer_idx
        ]
    )

    target_layers = [
        target_layer
    ]

    print(
        "\nTarget layer:"
    )

    print(
        f"  model.model[{target_layer_idx}]"
    )

    print(
        f"  {type(target_layer).__name__}"
    )


    # --------------------------------------------------------
    # GRADCAM TARGET
    # --------------------------------------------------------

    targets = [
        YoloDetectionTarget()
    ]


    # --------------------------------------------------------
    # GRADCAM
    # --------------------------------------------------------

    print(
        "\nGenerating GradCAM..."
    )

    with GradCAM(
        model=wrapped,
        target_layers=target_layers
    ) as cam:

        grayscale_cam = cam(
            input_tensor=tensor,
            targets=targets
        )


    grayscale_cam = (
        grayscale_cam[0]
    )


    # --------------------------------------------------------
    # CAM OVERLAY
    # --------------------------------------------------------

    cam_image = show_cam_on_image(
        rgb_float,
        grayscale_cam,
        use_rgb=True
    )


    # --------------------------------------------------------
    # DRAW YOLO DETECTIONS
    # --------------------------------------------------------

    annotated = draw_detections(
        cam_image,
        result
    )


    return (
        annotated,
        grayscale_cam,
        result
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "           YOLO GRADCAM"
    )

    print(
        "========================================"
    )

    print(
        f"\nWeights:\n{WEIGHTS}"
    )

    print(
        f"\nTest directory:\n{TEST_DIR}"
    )

    print(
        f"\nDevice:\n{DEVICE}"
    )

    print(
        f"\nImage size:\n{IMAGE_SIZE}"
    )

    print(
        f"\nConfidence:\n{CONF}"
    )


    # ========================================================
    # CHECK PATHS
    # ========================================================

    if not WEIGHTS.exists():

        raise FileNotFoundError(
            f"\nWeights not found:\n{WEIGHTS}"
        )


    if not TEST_DIR.exists():

        raise FileNotFoundError(
            f"\nTest directory not found:\n{TEST_DIR}"
        )


    # ========================================================
    # LOAD MODEL
    # ========================================================

    print(
        "\nLoading YOLO model..."
    )

    yolo = YOLO(
        WEIGHTS.as_posix()
    )

    print(
        "Model loaded."
    )


    # ========================================================
    # SAMPLE IMAGES
    # ========================================================

    sample_images = sorted(
        TEST_DIR.glob("*.jpg")
    )[:20]


    if not sample_images:

        raise FileNotFoundError(
            f"\nNo JPG images found in:\n{TEST_DIR}"
        )


    print(
        f"\nFound {len(sample_images)} "
        f"sample images."
    )


    # ========================================================
    # PROCESS
    # ========================================================

    successful = 0

    for i, img_path in enumerate(
        sample_images,
        start=1
    ):

        print(
            "\n----------------------------------------"
        )

        print(
            f"[{i}/{len(sample_images)}] "
            f"{img_path.name}"
        )


        try:

            (
                visualization,
                cam_map,
                result
            ) = run_gradcam(
                img_path,
                yolo
            )


            # ------------------------------------------------
            # SAVE VISUALIZATION
            # ------------------------------------------------

            output_path = (
                OUT_DIR
                / f"{img_path.stem}_gradcam.png"
            )

            cv2.imwrite(
                str(output_path),
                cv2.cvtColor(
                    visualization,
                    cv2.COLOR_RGB2BGR
                )
            )


            # ------------------------------------------------
            # SAVE RAW CAM
            # ------------------------------------------------

            cam_path = (
                OUT_DIR
                / f"{img_path.stem}_cam.npy"
            )

            np.save(
                str(cam_path),
                cam_map
            )


            print(
                "\nSaved:"
            )

            print(
                f"  {output_path}"
            )

            print(
                f"  {cam_path}"
            )


            successful += 1


        except Exception as e:

            print(
                "\nERROR:"
            )

            print(
                f"  {type(e).__name__}: {e}"
            )


    # ========================================================
    # SUMMARY
    # ========================================================

    print(
        "\n========================================"
    )

    print(
        "          GRADCAM COMPLETE"
    )

    print(
        "========================================"
    )

    print(
        f"\nSuccessful: "
        f"{successful}/{len(sample_images)}"
    )

    print(
        f"\nOutput directory:\n{OUT_DIR}"
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()
