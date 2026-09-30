import os
import glob
import cv2
import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from ultralytics import YOLO

# ───────────────────────── Config ─────────────────────────
IMAGE_DIR = "/home/admin1/Abishek-Intern/new_atr/data/newdata/test/images"
OUTPUT_DIR = "/home/admin1/Abishek-Intern/new_atr/data/newdata/test/images/gradcam_new2"
MODEL_WEIGHTS_PATH = "/home/admin1/Abishek-Intern/new_atr/new/models/yolo_aircraft.pt"
IMG_SIZE = 640          # match the size the model was trained/exported at
CLASS_IDX = None        # None = use the best class score across all anchors; or set an int
ALPHA = 0.5

os.makedirs(OUTPUT_DIR, exist_ok=True)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ───────────────────────── Load model ─────────────────────────
yolo = YOLO(MODEL_WEIGHTS_PATH)
yolo.to(device)
pure_pytorch_model = yolo.model
pure_pytorch_model.eval()

# model.model is the nn.Sequential of ultralytics layers; -4 is usually the
# last conv block before the detect head, similar in spirit to conv5_block3
# in ResNet. Fall back to the last Conv2d module if that layer isn't
# convolutional (index can vary between YOLO versions/sizes).
def get_gradcam_layer(seq_model, preferred_idx=-4):
    try:
        candidate = seq_model.model[preferred_idx]
        has_conv = any(isinstance(m, torch.nn.Conv2d) for m in candidate.modules())
        if has_conv:
            print(f"Grad-CAM target layer: model[{preferred_idx}] ({candidate.__class__.__name__})")
            return candidate
    except (IndexError, AttributeError):
        pass

    last_conv = None
    last_conv_name = None
    for name, module in seq_model.named_modules():
        if isinstance(module, torch.nn.Conv2d):
            last_conv = module
            last_conv_name = name
    if last_conv is None:
        raise ValueError("No suitable Conv2d layer found.")
    print(f"Grad-CAM fallback layer: {last_conv_name}")
    return last_conv


target_layer = get_gradcam_layer(pure_pytorch_model)


# ───────────────────────── Grad-CAM ─────────────────────────
class YOLOGradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        def forward_hook(module, inp, output):
            self.activations = output

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0]

        self.hooks.append(self.target_layer.register_forward_hook(forward_hook))
        self.hooks.append(self.target_layer.register_full_backward_hook(backward_hook))

    def generate_heatmap(self, input_tensor, class_idx=None):
        self.model.zero_grad()
        predictions = self.model(input_tensor)

        # Ultralytics DetectionModel in eval() mode returns either a single
        # tensor or a tuple (inference_out, raw_feature_maps) depending on
        # version — always take the inference tensor.
        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0]

        # YOLOv8-style raw output: (batch, 4 + num_classes, num_anchors)
        # (no separate objectness channel, unlike YOLOv5).
        class_scores = predictions[:, 4:, :]
        if class_idx is not None:
            score = class_scores[:, class_idx, :].max()
        else:
            score = class_scores.max()

        score.backward(retain_graph=True)

        if self.gradients is None or self.activations is None:
            raise RuntimeError(
                "Hooks did not fire — check that target_layer is actually "
                "on the forward path used by this model."
            )

        gradients = self.gradients.detach().cpu().numpy()[0]     # (C, H, W)
        activations = self.activations.detach().cpu().numpy()[0]  # (C, H, W)

        weights = np.mean(gradients, axis=(1, 2))                 # (C,)
        cam = np.zeros(activations.shape[1:], dtype=np.float32)   # (H, W)
        for i, w in enumerate(weights):
            cam += w * activations[i, :, :]

        # ReLU — only positive activations matter
        cam = np.maximum(cam, 0)

        # ── Key fix 1: Gaussian smooth to reduce edge spike artifacts
        cam = gaussian_filter(cam, sigma=1)

        # Normalize
        if cam.max() != 0:
            cam = cam / cam.max()

        # ── Key fix 2: Threshold — suppress weak activations below 20%
        cam[cam < 0.2] = 0

        # Re-normalize after threshold
        if cam.max() != 0:
            cam = cam / cam.max()

        return cam

    def remove_hooks(self):
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


cam_extractor = YOLOGradCAM(pure_pytorch_model, target_layer)


# ───────────────────────── Preprocessing ─────────────────────────
def preprocess(img_bgr, size=IMG_SIZE):
    img_resized = cv2.resize(img_bgr, (size, size))
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_norm = img_rgb.astype(np.float32) / 255.0
    img_chw = img_norm.transpose(2, 0, 1)            # HWC -> CHW
    img_chw = np.ascontiguousarray(img_chw)
    tensor = torch.from_numpy(img_chw).unsqueeze(0).to(device)
    tensor.requires_grad_(True)
    return tensor, img_resized


# ───────────────────────── Overlay / plot (mirrors Keras reference) ─────────────────────────
def overlay_heatmap(original_img_bgr, heatmap, output_path, alpha=ALPHA):
    img_bgr = cv2.resize(original_img_bgr, (IMG_SIZE, IMG_SIZE))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    heatmap_resized = cv2.resize(heatmap, (IMG_SIZE, IMG_SIZE))

    # ── Key fix 3: mask out pure-black background/letterbox pixels
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    _, fg_mask = cv2.threshold(gray, 10, 1, cv2.THRESH_BINARY)
    fg_mask = fg_mask.astype(np.float32)

    heatmap_masked = heatmap_resized * fg_mask
    if heatmap_masked.max() != 0:
        heatmap_masked = heatmap_masked / heatmap_masked.max()

    heatmap_uint8 = np.uint8(255 * heatmap_masked)
    heatmap_colored = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    img_float = img_rgb.astype(np.float32)
    heatmap_float = heatmap_colored.astype(np.float32)
    overlay = (img_float * (1 - alpha) + heatmap_float * alpha).astype(np.uint8)

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    fig.patch.set_facecolor('#0d0d0d')

    panels = [
        (img_rgb, None, 'Original Image'),
        (heatmap_masked, 'jet', 'Grad-CAM Heatmap'),
        (overlay, None, 'Overlay'),
    ]
    for ax, (data, cmap, title) in zip(axes, panels):
        ax.imshow(data, cmap=cmap)
        ax.set_title(title, color='white', fontsize=11, pad=6)
        ax.axis('off')

    sm = plt.cm.ScalarMappable(cmap='jet', norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes[1], fraction=0.046, pad=0.04)
    cbar.set_label('Attention', color='white', fontsize=9)
    cbar.ax.yaxis.set_tick_params(color='white')
    plt.setp(cbar.ax.yaxis.get_ticklabels(), color='white')

    plt.tight_layout(pad=0.5)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches='tight', facecolor='#0d0d0d')
    plt.close()
    return output_path


# ───────────────────────── Batch process ─────────────────────────
def main():
    image_extensions = ('*.jpg', '*.jpeg', '*.png')
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob.glob(os.path.join(IMAGE_DIR, ext)))
    print(f"Found {len(image_paths)} images")

    for img_path in image_paths:
        try:
            img_bgr = cv2.imread(img_path)
            if img_bgr is None:
                print(f"Skipping unreadable image: {img_path}")
                continue

            input_tensor, resized_bgr = preprocess(img_bgr)
            heatmap = cam_extractor.generate_heatmap(input_tensor, class_idx=CLASS_IDX)

            img_name = os.path.basename(img_path)
            save_path = os.path.join(OUTPUT_DIR, f"gradcam_{os.path.splitext(img_name)[0]}.png")
            overlay_heatmap(resized_bgr, heatmap, save_path)
            print(f"Saved: {save_path}")

        except Exception as e:
            print(f"Error processing {img_path}: {str(e)}")

    cam_extractor.remove_hooks()
    print("Done")


if __name__ == "__main__":
    main()
