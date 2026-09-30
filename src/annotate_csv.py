import cv2
import csv
import numpy as np
from pathlib import Path

ROOT_DIR = Path("/home/admin1/Abishek-Intern/new_atr/aircrafts")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

MOVE_STEP = 2
RESIZE_STEP = 2
EDGE_TOLERANCE = 10
MIN_BOX_SIZE = 10

# ============================================================
# AUTO-DETECTION (plug your model in here)
# ============================================================

AUTO_DETECT_ENABLED = True

MODEL_PATH = "/home/admin1/Abishek-Intern/new_atr/new/models/yolo_aircraft.pt"
CONFIDENCE_THRESHOLD = 0.25


def load_model():
    from ultralytics import YOLO
    return YOLO(MODEL_PATH)


def auto_detect(image):
    results = MODEL.predict(image, verbose=False)[0]

    boxes = []

    for b in results.boxes:

        if float(b.conf[0]) < CONFIDENCE_THRESHOLD:
            continue

        x1, y1, x2, y2 = b.xyxy[0].tolist()

        boxes.append(
            [int(x1), int(y1), int(x2), int(y2)]
        )

    return boxes


MODEL = load_model() if AUTO_DETECT_ENABLED else None

images = []
for class_dir in sorted(ROOT_DIR.iterdir()):
    if not class_dir.is_dir():
        continue
    class_name = class_dir.name
    for image_path in sorted(class_dir.iterdir()):
        if image_path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append((image_path, class_name))

if not images:
    print("No images found.")
    exit()

pending_images = []
completed_count = 0
for image_path, class_name in images:
    csv_path = image_path.with_suffix(".csv")
    if csv_path.exists():
        completed_count += 1
    else:
        pending_images.append((image_path, class_name))

if not pending_images:
    print("\nAll images are already annotated!")
    exit()

current_index = 0
boxes = []
selected_box = -1
image = None
display = None
image_width = 0
image_height = 0
drawing_new = False
dragging_box = False
resizing_box = False
start_x = 0
start_y = 0
drag_offset_x = 0
drag_offset_y = 0
resize_mode = None

# Fixed on-screen canvas size — every image is scaled/letterboxed to fit
# this, so the window never changes size between images.
DISPLAY_WIDTH = 1000
DISPLAY_HEIGHT = 700

display_scale = 1.0
offset_x = 0
offset_y = 0


def image_to_canvas(px, py):
    return (
        int(px * display_scale) + offset_x,
        int(py * display_scale) + offset_y,
    )


def canvas_to_image(cx, cy):
    return (
        (cx - offset_x) / display_scale,
        (cy - offset_y) / display_scale,
    )


def clamp_box(box):
    x1, y1, x2, y2 = box
    x1 = max(0, min(int(x1), image_width - 1))
    y1 = max(0, min(int(y1), image_height - 1))
    x2 = max(0, min(int(x2), image_width - 1))
    y2 = max(0, min(int(y2), image_height - 1))
    return [x1, y1, x2, y2]


def redraw():
    global display

    new_w = max(1, int(image_width * display_scale))
    new_h = max(1, int(image_height * display_scale))

    resized = cv2.resize(image, (new_w, new_h))

    canvas = np.full((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), 40, dtype=np.uint8)
    canvas[offset_y:offset_y + new_h, offset_x:offset_x + new_w] = resized

    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = box
        cx1, cy1 = image_to_canvas(x1, y1)
        cx2, cy2 = image_to_canvas(x2, y2)
        thickness = 3 if i == selected_box else 2
        cv2.rectangle(canvas, (cx1, cy1), (cx2, cy2), (0, 255, 0), thickness)
        if i == selected_box:
            handle = 6
            corners = [(cx1, cy1), (cx2, cy1), (cx1, cy2), (cx2, cy2)]
            for cx, cy in corners:
                cv2.rectangle(canvas, (cx - handle, cy - handle), (cx + handle, cy + handle), (0, 255, 255), -1)

    display = canvas
    cv2.imshow("Aircraft Annotation", display)


def find_box(x, y):
    for i in range(len(boxes) - 1, -1, -1):
        x1, y1, x2, y2 = boxes[i]
        if x1 <= x <= x2 and y1 <= y <= y2:
            return i
    return -1


def get_resize_mode(x, y, box):
    x1, y1, x2, y2 = box
    near_left = abs(x - x1) <= EDGE_TOLERANCE
    near_right = abs(x - x2) <= EDGE_TOLERANCE
    near_top = abs(y - y1) <= EDGE_TOLERANCE
    near_bottom = abs(y - y2) <= EDGE_TOLERANCE
    if near_left and near_top:
        return "top_left"
    if near_right and near_top:
        return "top_right"
    if near_left and near_bottom:
        return "bottom_left"
    if near_right and near_bottom:
        return "bottom_right"
    if near_left:
        return "left"
    if near_right:
        return "right"
    if near_top:
        return "top"
    if near_bottom:
        return "bottom"
    return None


def mouse_callback(event, x, y, flags, param):
    global drawing_new, dragging_box, resizing_box
    global start_x, start_y, drag_offset_x, drag_offset_y
    global resize_mode, selected_box

    # x, y arrive in on-screen canvas coordinates; convert to original
    # image pixel coordinates before doing any box math.
    img_x, img_y = canvas_to_image(x, y)

    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_box = find_box(img_x, img_y)
        if clicked_box != -1:
            selected_box = clicked_box
            box = boxes[selected_box]
            mode = get_resize_mode(img_x, img_y, box)
            if mode is not None:
                resizing_box = True
                resize_mode = mode
                start_x = img_x
                start_y = img_y
            else:
                dragging_box = True
                x1, y1, x2, y2 = box
                drag_offset_x = img_x - x1
                drag_offset_y = img_y - y1
            redraw()
        else:
            selected_box = -1
            drawing_new = True
            start_x = img_x
            start_y = img_y

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing_new:
            temp = display.copy()
            sx, sy = image_to_canvas(start_x, start_y)
            cv2.rectangle(temp, (sx, sy), (x, y), (0, 255, 0), 2)
            cv2.imshow("Aircraft Annotation", temp)
        elif dragging_box:
            x1, y1, x2, y2 = boxes[selected_box]
            width = x2 - x1
            height = y2 - y1
            new_x1 = img_x - drag_offset_x
            new_y1 = img_y - drag_offset_y
            new_x2 = new_x1 + width
            new_y2 = new_y1 + height
            boxes[selected_box] = clamp_box([new_x1, new_y1, new_x2, new_y2])
            redraw()
        elif resizing_box:
            x1, y1, x2, y2 = boxes[selected_box]
            if resize_mode == "left":
                x1 = min(img_x, x2 - MIN_BOX_SIZE)
            elif resize_mode == "right":
                x2 = max(img_x, x1 + MIN_BOX_SIZE)
            elif resize_mode == "top":
                y1 = min(img_y, y2 - MIN_BOX_SIZE)
            elif resize_mode == "bottom":
                y2 = max(img_y, y1 + MIN_BOX_SIZE)
            elif resize_mode == "top_left":
                x1 = min(img_x, x2 - MIN_BOX_SIZE)
                y1 = min(img_y, y2 - MIN_BOX_SIZE)
            elif resize_mode == "top_right":
                x2 = max(img_x, x1 + MIN_BOX_SIZE)
                y1 = min(img_y, y2 - MIN_BOX_SIZE)
            elif resize_mode == "bottom_left":
                x1 = min(img_x, x2 - MIN_BOX_SIZE)
                y2 = max(img_y, y1 + MIN_BOX_SIZE)
            elif resize_mode == "bottom_right":
                x2 = max(img_x, x1 + MIN_BOX_SIZE)
                y2 = max(img_y, y1 + MIN_BOX_SIZE)
            boxes[selected_box] = clamp_box([x1, y1, x2, y2])
            redraw()

    elif event == cv2.EVENT_LBUTTONUP:
        if drawing_new:
            drawing_new = False
            x1 = min(start_x, img_x)
            y1 = min(start_y, img_y)
            x2 = max(start_x, img_x)
            y2 = max(start_y, img_y)
            if abs(x2 - x1) >= MIN_BOX_SIZE and abs(y2 - y1) >= MIN_BOX_SIZE:
                boxes.append(clamp_box([x1, y1, x2, y2]))
                selected_box = len(boxes) - 1
            redraw()
        dragging_box = False
        resizing_box = False
        resize_mode = None


def save_annotation(image_path, class_name, image_width, image_height, boxes):
    csv_path = image_path.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["filename", "width", "height", "class", "xmin", "ymin", "xmax", "ymax"])
        filename = image_path.stem
        for x1, y1, x2, y2 in boxes:
            writer.writerow([filename, image_width, image_height, class_name, x1, y1, x2, y2])
    print(f"\nSaved: {csv_path}")


cv2.namedWindow("Aircraft Annotation", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Aircraft Annotation", DISPLAY_WIDTH, DISPLAY_HEIGHT)
cv2.setMouseCallback("Aircraft Annotation", mouse_callback)

while True:
    image_path, class_name = pending_images[current_index]
    image = cv2.imread(str(image_path))
    if image is None:
        current_index += 1
        if current_index >= len(pending_images):
            break
        continue
    image_height, image_width = image.shape[:2]

    scale_w = DISPLAY_WIDTH / image_width
    scale_h = DISPLAY_HEIGHT / image_height
    display_scale = min(scale_w, scale_h)

    scaled_w = int(image_width * display_scale)
    scaled_h = int(image_height * display_scale)

    offset_x = (DISPLAY_WIDTH - scaled_w) // 2
    offset_y = (DISPLAY_HEIGHT - scaled_h) // 2

    boxes = []

    if AUTO_DETECT_ENABLED:
        try:
            raw_boxes = auto_detect(image)
        except Exception as e:
            print(f"auto_detect failed on {image_path.name}: {e}")
            raw_boxes = []
        boxes = [clamp_box(b) for b in raw_boxes]

    selected_box = 0 if boxes else -1
    drawing_new = False
    dragging_box = False
    resizing_box = False

    print(f"\n[{current_index + 1}/{len(pending_images)}] {image_path.name}  "
          f"({len(boxes)} auto box(es) — press 's' to accept, or edit then 's')")

    redraw()

    while True:
        key = cv2.waitKey(0) & 0xFF
        key_char = chr(key).lower() if 32 <= key < 127 else ""

        if key_char == "s":
            if not boxes:
                continue
            save_annotation(image_path, class_name, image_width, image_height, boxes)
            break
        elif key_char == "r":
            boxes = []
            selected_box = -1
            redraw()
        elif key_char == "n":
            break

        elif key_char == "d":
            csv_path = image_path.with_suffix(".csv")
            try:
                image_path.unlink()
                if csv_path.exists():
                    csv_path.unlink()
                print(f"Deleted: {image_path.name}")
            except Exception as e:
                print(f"Could not delete {image_path.name}: {e}")
            break
        elif key_char == "q":
            cv2.destroyAllWindows()
            exit()
        elif key == 8:
            if selected_box >= 0:
                boxes.pop(selected_box)
                selected_box = -1
                redraw()
        elif key == 81:
            if selected_box >= 0:
                x1, y1, x2, y2 = boxes[selected_box]
                boxes[selected_box] = clamp_box([x1 - MOVE_STEP, y1, x2 - MOVE_STEP, y2])
                redraw()
        elif key == 82:
            if selected_box >= 0:
                x1, y1, x2, y2 = boxes[selected_box]
                boxes[selected_box] = clamp_box([x1, y1 - MOVE_STEP, x2, y2 - MOVE_STEP])
                redraw()
        elif key == 83:
            if selected_box >= 0:
                x1, y1, x2, y2 = boxes[selected_box]
                boxes[selected_box] = clamp_box([x1 + MOVE_STEP, y1, x2 + MOVE_STEP, y2])
                redraw()
        elif key == 84:
            if selected_box >= 0:
                x1, y1, x2, y2 = boxes[selected_box]
                boxes[selected_box] = clamp_box([x1, y1 + MOVE_STEP, x2, y2 + MOVE_STEP])
                redraw()
        elif key_char in ("+", "="):
            if selected_box >= 0:
                x1, y1, x2, y2 = boxes[selected_box]
                boxes[selected_box] = clamp_box([x1 - RESIZE_STEP, y1 - RESIZE_STEP, x2 + RESIZE_STEP, y2 + RESIZE_STEP])
                redraw()
        elif key_char in ("-", "_"):
            if selected_box >= 0:
                x1, y1, x2, y2 = boxes[selected_box]
                if x2 - x1 > MIN_BOX_SIZE and y2 - y1 > MIN_BOX_SIZE:
                    boxes[selected_box] = clamp_box([x1 + RESIZE_STEP, y1 + RESIZE_STEP, x2 - RESIZE_STEP, y2 - RESIZE_STEP])
                    redraw()
        elif key_char.isdigit() and key_char != "0":
            box_number = int(key_char) - 1
            if box_number < len(boxes):
                selected_box = box_number
                redraw()
        else:
            print(f"(unrecognized key code: {key} — click the image window first, then try again)")

    current_index += 1
    if current_index >= len(pending_images):
        break

cv2.destroyAllWindows()
