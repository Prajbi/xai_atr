from pathlib import Path

import cv2
import numpy as np
import torch

from ultralytics import YOLO
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path("/home/admin1/Abishek-Intern/new_atr")

WEIGHTS = (
    PROJECT_ROOT
    / "new"
    / "models"
    / "yolo_aircraft.pt"
)

OUT_DIR = PROJECT_ROOT / "outputs" / "eigencam_new"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# SETTINGS
# ============================================================

IMAGE_SIZE = 832
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


# ============================================================
# YOLO WRAPPER
# ============================================================

class YoloEigenCamWrapper(torch.nn.Module):

    def __init__(self, yolo_model):
        super().__init__()

        # Underlying Ultralytics PyTorch model
        self.model = yolo_model.model

    def forward(self, x):

        output = self.model(x)

        # Ultralytics models can return tuples
        # depending on mode/version.
        if isinstance(output, (tuple, list)):
            return output[0]

        return output


# ============================================================
# IMAGE LOADING
# ============================================================

def load_image(path, size=IMAGE_SIZE):

    img = cv2.imread(str(path))

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
        (size, size)
    )

    rgb_float = (
        img.astype(np.float32) / 255.0
    )

    tensor = torch.from_numpy(
        rgb_float
    ).permute(2, 0, 1).unsqueeze(0)

    tensor = tensor.float()

    return rgb_float, tensor


# ============================================================
# EIGENCAM
# ============================================================

def run_eigencam(
    image_path,
    yolo_model,
    target_layer_idx=-4
):

    wrapped = YoloEigenCamWrapper(
        yolo_model
    )

    wrapped = wrapped.to(DEVICE)
    wrapped.eval()


    # --------------------------------------------------------
    # TARGET LAYER
    # --------------------------------------------------------

    target_layer = (
        wrapped.model.model[target_layer_idx]
    )

    target_layers = [target_layer]


    print(
        f"\nTarget layer: "
        f"model.model[{target_layer_idx}]"
    )

    print(
        f"Layer type: "
        f"{type(target_layer).__name__}"
    )


    # --------------------------------------------------------
    # IMAGE
    # --------------------------------------------------------

    rgb_float, tensor = load_image(
        image_path,
        IMAGE_SIZE
    )

    tensor = tensor.to(DEVICE)


    # --------------------------------------------------------
    # EIGENCAM
    # --------------------------------------------------------

    # IMPORTANT:
    # No task="od".
    #
    # EigenCAM's current constructor is:
    #
    # EigenCAM(
    #     model=model,
    #     target_layers=target_layers
    # )

    with EigenCAM(
        model=wrapped,
        target_layers=target_layers
    ) as cam:

        grayscale_cam = cam(
            input_tensor=tensor,
            targets=None
        )

    grayscale_cam = grayscale_cam[0]


    # --------------------------------------------------------
    # OVERLAY
    # --------------------------------------------------------

    visualization = show_cam_on_image(
        rgb_float,
        grayscale_cam,
        use_rgb=True
    )

    return visualization, grayscale_cam


# ============================================================
# MAIN
# ============================================================

def main():

    print("\n========================================")
    print("             YOLO EIGENCAM")
    print("========================================")

    print(f"\nWeights:")
    print(WEIGHTS)

    print(f"\nDevice:")
    print(DEVICE)

    print(f"\nImage size:")
    print(IMAGE_SIZE)


    # --------------------------------------------------------
    # CHECK MODEL
    # --------------------------------------------------------

    if not WEIGHTS.exists():

        raise FileNotFoundError(
            f"\nModel not found:\n{WEIGHTS}"
        )


    # --------------------------------------------------------
    # LOAD MODEL
    # --------------------------------------------------------

    print("\nLoading YOLO model...")

    yolo = YOLO(
        WEIGHTS.as_posix()
    )

    print("Model loaded.")


    # --------------------------------------------------------
    # TEST IMAGES
    # --------------------------------------------------------

    test_dir = (
        PROJECT_ROOT
        / "data"
        / "filtered_350"
        / "images"
        / "test"
    )

    sample_images = sorted(
        test_dir.glob("*.jpg")
    )[:20]


    if not sample_images:

        raise FileNotFoundError(
            f"No JPG images found in:\n{test_dir}"
        )


    print(
        f"\nFound {len(sample_images)} "
        f"sample images."
    )


    # --------------------------------------------------------
    # PROCESS
    # --------------------------------------------------------

    for i, img_path in enumerate(
        sample_images,
        start=1
    ):

        print(
            f"\n[{i}/{len(sample_images)}] "
            f"{img_path.name}"
        )

        try:

            vis, cam_map = run_eigencam(
                img_path,
                yolo
            )


            # ------------------------------------------------
            # SAVE VISUALIZATION
            # ------------------------------------------------

            out_path = (
                OUT_DIR
                / f"{img_path.stem}_eigencam.png"
            )

            cv2.imwrite(
                str(out_path),
                cv2.cvtColor(
                    vis,
                    cv2.COLOR_RGB2BGR
                )
            )


            # ------------------------------------------------
            # SAVE RAW CAM
            # ------------------------------------------------

            npy_path = (
                OUT_DIR
                / f"{img_path.stem}_cam.npy"
            )

            np.save(
                str(npy_path),
                cam_map
            )


            print(
                f"Saved:\n"
                f"  {out_path}\n"
                f"  {npy_path}"
            )


        except Exception as e:

            print(
                f"ERROR processing "
                f"{img_path.name}:"
            )

            print(e)


    print("\n========================================")
    print("           EIGENCAM COMPLETE")
    print("========================================")

    print(
        f"\nOutput directory:\n{OUT_DIR}"
    )


# ============================================================
# ENTRY
# ============================================================

if __name__ == "__main__":
    main()
