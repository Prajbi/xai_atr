import cv2
import csv
from pathlib import Path
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

# Folder structure:
# aircrafts/
#   J35/
#     image1.jpg
#     image1.csv          <- existing whole-aircraft annotation
#     image1_parts.csv    <- generated nose/wings/tail annotation
#   Su30/
#     ...

ROOT_DIR = Path("/home/admin1/Abishek-Intern/new_atr/data/aircrafts/J35")

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

DISPLAY_WIDTH = 1100
DISPLAY_HEIGHT = 750
MIN_BOX_SIZE = 5

# Colors are in BGR format.
WHOLE_BOX_COLOR = (255, 180, 0)  # blue/cyan
PART_COLORS = {
    "nose":  (0, 255, 255),       # yellow
    "wings": (0, 255, 0),         # green
    "tail":  (255, 0, 255),       # magenta
}

PART_KEYS = {
    ord("1"): "nose",
    ord("2"): "wings",
    ord("3"): "tail",
}

WINDOW_NAME = "Aircraft Part Annotation"

# ============================================================
# FIND IMAGES AND RESUME
# ============================================================

if not ROOT_DIR.exists():
    raise SystemExit(f"ROOT_DIR does not exist: {ROOT_DIR}")

images = []

# Supports either:
# 1) ROOT_DIR points directly to one aircraft class folder, e.g. .../aircrafts/J35
# 2) ROOT_DIR points to the parent folder containing class folders, e.g. .../aircrafts
if any(p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS for p in ROOT_DIR.iterdir()):
    class_name = ROOT_DIR.name
    for image_path in sorted(ROOT_DIR.iterdir()):
        if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
            images.append((image_path, class_name))
else:
    for class_dir in sorted(ROOT_DIR.iterdir()):
        if not class_dir.is_dir():
            continue
        for image_path in sorted(class_dir.iterdir()):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                images.append((image_path, class_dir.name))

pending = []
completed = 0

for image_path, class_name in images:
    parts_csv = image_path.with_name(image_path.stem + "_parts.csv")
    if parts_csv.exists():
        completed += 1
    else:
        pending.append((image_path, class_name))

print("=" * 65)
print("AIRCRAFT PART ANNOTATION")
print("=" * 65)
print(f"Root folder       : {ROOT_DIR}")
print(f"Total images      : {len(images)}")
print(f"Already completed : {completed}")
print(f"Remaining         : {len(pending)}")
print("=" * 65)

if not pending:
    raise SystemExit("No pending images. All images already have *_parts.csv files.")

# ============================================================
# GLOBAL STATE
# ============================================================

image = None
canvas = None
image_h = image_w = 0
scale = 1.0
offset_x = offset_y = 0

current_part = "nose"
part_boxes = {"nose": [], "wings": [], "tail": []}  # multiple boxes per part
box_history = []       # global drawing order, used by Undo
whole_boxes = []         # existing complete-aircraft boxes

drawing = False
start_x = start_y = 0
mouse_x = mouse_y = 0

# ============================================================
# COORDINATE CONVERSION
# ============================================================

def canvas_to_image(x, y):
    ix = (x - offset_x) / scale
    iy = (y - offset_y) / scale
    return int(round(ix)), int(round(iy))


def image_to_canvas(x, y):
    return int(round(x * scale + offset_x)), int(round(y * scale + offset_y))


def clamp_box(box):
    x1, y1, x2, y2 = box
    x1 = max(0, min(x1, image_w - 1))
    y1 = max(0, min(y1, image_h - 1))
    x2 = max(0, min(x2, image_w - 1))
    y2 = max(0, min(y2, image_h - 1))
    return [int(x1), int(y1), int(x2), int(y2)]


def inside_image(x, y):
    return (
        offset_x <= x < offset_x + int(image_w * scale)
        and offset_y <= y < offset_y + int(image_h * scale)
    )

# ============================================================
# LOAD EXISTING WHOLE-AIRCRAFT BOXES
# ============================================================

def load_whole_aircraft_boxes(image_path):
    csv_path = image_path.with_suffix(".csv")
    result = []

    if not csv_path.exists():
        print(f"WARNING: Existing whole-aircraft CSV not found: {csv_path}")
        return result

    try:
        with open(csv_path, "r", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                result.append({
                    "class": row.get("class", ""),
                    "box": [
                        int(float(row["xmin"])),
                        int(float(row["ymin"])),
                        int(float(row["xmax"])),
                        int(float(row["ymax"])),
                    ],
                })
    except Exception as exc:
        print(f"Could not read {csv_path}: {exc}")

    return result

# ============================================================
# DRAW DISPLAY
# ============================================================

def redraw():
    global canvas

    resized_w = max(1, int(image_w * scale))
    resized_h = max(1, int(image_h * scale))
    resized = cv2.resize(image, (resized_w, resized_h))

    canvas = np.full((DISPLAY_HEIGHT, DISPLAY_WIDTH, 3), 35, dtype=np.uint8)
    canvas[offset_y:offset_y + resized_h, offset_x:offset_x + resized_w] = resized

    # Existing whole-aircraft boxes: cyan/blue.
    for item in whole_boxes:
        x1, y1, x2, y2 = item["box"]
        p1 = image_to_canvas(x1, y1)
        p2 = image_to_canvas(x2, y2)
        cv2.rectangle(canvas, p1, p2, WHOLE_BOX_COLOR, 2)
        label = f"WHOLE AIRCRAFT: {item['class']}"
        cv2.putText(canvas, label, (p1[0], max(20, p1[1] - 8)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, WHOLE_BOX_COLOR, 2)

    # Nose, wings and tail boxes.
    for part, boxes_for_part in part_boxes.items():
        color = PART_COLORS[part]
        for box_index, box in enumerate(boxes_for_part, start=1):
            x1, y1, x2, y2 = box
            p1 = image_to_canvas(x1, y1)
            p2 = image_to_canvas(x2, y2)
            cv2.rectangle(canvas, p1, p2, color, 3)
            label = f"{part.upper()} {box_index}"
            cv2.putText(canvas, label, (p1[0], max(20, p1[1] - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    # Preview the box while the mouse is being dragged.
    if drawing:
        p1 = image_to_canvas(start_x, start_y)
        p2 = (mouse_x, mouse_y)
        cv2.rectangle(canvas, p1, p2, PART_COLORS[current_part], 2)
        cv2.putText(canvas, f"DRAWING: {current_part.upper()}",
                    (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    PART_COLORS[current_part], 2)

    # Instruction panel.
    panel_y = 65
    instructions = [
        f"CURRENT PART: {current_part.upper()}  |  1 Nose   2 Wings   3 Tail",
        "Drag mouse: draw/replace current part box",
        "S: Save and next image    K: Skip    U: Undo last box",
        "R: Clear all part boxes   Q: Quit",
        "You can draw MULTIPLE boxes for the selected part.",
    ]
    for line in instructions:
        cv2.putText(canvas, line, (15, panel_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1)
        panel_y += 25

    cv2.imshow(WINDOW_NAME, canvas)

# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(event, x, y, flags, param):
    global drawing, start_x, start_y, mouse_x, mouse_y

    mouse_x, mouse_y = x, y

    if event == cv2.EVENT_LBUTTONDOWN:
        if inside_image(x, y):
            start_x, start_y = canvas_to_image(x, y)
            start_x = max(0, min(start_x, image_w - 1))
            start_y = max(0, min(start_y, image_h - 1))
            drawing = True
            redraw()

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        redraw()

    elif event == cv2.EVENT_LBUTTONUP and drawing:
        drawing = False

        if not inside_image(x, y):
            redraw()
            return

        end_x, end_y = canvas_to_image(x, y)
        x1, x2 = sorted((start_x, end_x))
        y1, y2 = sorted((start_y, end_y))

        if (x2 - x1) >= MIN_BOX_SIZE and (y2 - y1) >= MIN_BOX_SIZE:
            # Add another box for the selected part. This supports
            # partially visible or separated left/right wing sections.
            new_box = clamp_box([x1, y1, x2, y2])
            part_boxes[current_part].append(new_box)
            box_history.append(current_part)
            print(
                f"Added {current_part} box "
                f"#{len(part_boxes[current_part])}: {new_box}"
            )
        else:
            print("Box too small; ignored.")

        redraw()

# ============================================================
# SAVE PART ANNOTATIONS
# ============================================================

def save_part_annotations(image_path, class_name):
    output_csv = image_path.with_name(image_path.stem + "_parts.csv")

    fieldnames = [
        "filename", "width", "height", "aircraft_class",
        "part", "xmin", "ymin", "xmax", "ymax"
    ]

    with open(output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for part in ("nose", "wings", "tail"):
            for box_index, box in enumerate(part_boxes[part], start=1):
                x1, y1, x2, y2 = box
                writer.writerow({
                    "filename": image_path.stem,
                    "width": image_w,
                    "height": image_h,
                    "aircraft_class": class_name,
                    "part": part,
                    "xmin": x1,
                    "ymin": y1,
                    "xmax": x2,
                    "ymax": y2,
                })

    print(f"SAVED: {output_csv}")
    annotated_parts = [
        f"{part} ({len(part_boxes[part])} box(es))"
        for part in ("nose", "wings", "tail")
        if part_boxes[part]
    ]
    print(f"Parts annotated: {', '.join(annotated_parts) if annotated_parts else 'None'}")

# ============================================================
# MAIN LOOP
# ============================================================

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WINDOW_NAME, DISPLAY_WIDTH, DISPLAY_HEIGHT)
cv2.setMouseCallback(WINDOW_NAME, mouse_callback)

processed = 0

for index, (image_path, class_name) in enumerate(pending, start=1):
    image = cv2.imread(str(image_path))

    if image is None:
        print(f"Could not read image: {image_path}")
        continue

    image_h, image_w = image.shape[:2]
    scale = min(DISPLAY_WIDTH / image_w, DISPLAY_HEIGHT / image_h)
    scaled_w = int(image_w * scale)
    scaled_h = int(image_h * scale)
    offset_x = (DISPLAY_WIDTH - scaled_w) // 2
    offset_y = (DISPLAY_HEIGHT - scaled_h) // 2

    whole_boxes = load_whole_aircraft_boxes(image_path)
    part_boxes = {"nose": [], "wings": [], "tail": []}
    box_history = []
    current_part = "nose"
    drawing = False

    print("\n" + "=" * 65)
    print(f"[{index}/{len(pending)}] {image_path.name}")
    print(f"Aircraft class: {class_name}")
    print(f"Existing whole-aircraft boxes: {len(whole_boxes)}")
    print("Select 1/2/3, draw only visible parts, then press S to save.")
    print("=" * 65)

    redraw()

    while True:
        key = cv2.waitKey(0) & 0xFF

        if key in PART_KEYS:
            current_part = PART_KEYS[key]
            print(f"Selected part: {current_part.upper()}")
            redraw()

        elif key == ord("s"):
            save_part_annotations(image_path, class_name)
            processed += 1
            break

        elif key == ord("k"):
            print(f"Skipped without saving: {image_path.name}")
            break

        elif key == ord("u"):
            # Undo the most recently added box across all three parts.
            # Each box receives a global order when drawn.
            if box_history:
                last_part = box_history.pop()
                if part_boxes[last_part]:
                    removed = part_boxes[last_part].pop()
                    print(f"Undid {last_part} box: {removed}")
            else:
                print("No part box to undo.")
            redraw()

        elif key == ord("r"):
            for part in part_boxes:
                part_boxes[part].clear()
            box_history.clear()
            print("Cleared all part boxes.")
            redraw()

        elif key == ord("q"):
            cv2.destroyAllWindows()
            print("\nAnnotation stopped. Previously saved CSV files are safe.")
            raise SystemExit

cv2.destroyAllWindows()

print("\n" + "=" * 65)
print("PART ANNOTATION SESSION FINISHED")
print(f"Images saved in this session: {processed}")
print(f"Total images already completed: {completed + processed}")
print("=" * 65)
