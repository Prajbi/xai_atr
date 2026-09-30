import cv2
import csv
import numpy as np
from pathlib import Path

# ============================================================
# CONFIG
# ============================================================

ROOT_DIR = Path("/home/admin1/Abishek-Intern/new_atr/aircrafts")

IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
}

MOVE_STEP = 2
RESIZE_STEP = 2
EDGE_TOLERANCE = 10
MIN_BOX_SIZE = 10

# Fixed on-screen window size. Every image is scaled to fit inside
# this and letterboxed, so the window never resizes between images.
DISPLAY_WIDTH = 1000
DISPLAY_HEIGHT = 700

# ============================================================
# FIND IMAGES
# ============================================================

images = []

for class_dir in sorted(ROOT_DIR.iterdir()):

    if not class_dir.is_dir():
        continue

    class_name = class_dir.name

    for image_path in sorted(class_dir.iterdir()):

        if image_path.suffix.lower() in IMAGE_EXTENSIONS:

            images.append(
                (image_path, class_name)
            )

if not images:
    print("No images found.")
    exit()

print(f"\nFound {len(images)} images.")
print(f"Root: {ROOT_DIR}\n")

# ============================================================
# RESUME
# ============================================================

pending_images = []
completed_count = 0

for image_path, class_name in images:

    csv_path = image_path.with_suffix(".csv")

    if csv_path.exists():

        completed_count += 1

    else:

        pending_images.append(
            (image_path, class_name)
        )

print("=" * 60)
print("RESUME INFORMATION")
print("=" * 60)

print(f"Total images     : {len(images)}")
print(f"Already annotated: {completed_count}")
print(f"Remaining        : {len(pending_images)}")

print("=" * 60)

if not pending_images:

    print("\nAll images are already annotated!")
    exit()

# ============================================================
# GLOBAL VARIABLES
# ============================================================

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

# Scale/offset used to fit the current image into the fixed window.
display_scale = 1.0
offset_x = 0
offset_y = 0

# ============================================================
# CANVAS <-> IMAGE COORDINATE HELPERS
# ============================================================

def image_to_canvas(px, py):

    return (
        int(px * display_scale) + offset_x,
        int(py * display_scale) + offset_y
    )


def canvas_to_image(cx, cy):

    return (
        (cx - offset_x) / display_scale,
        (cy - offset_y) / display_scale
    )

# ============================================================
# CLAMP BOX
# ============================================================

def clamp_box(box):

    x1, y1, x2, y2 = box

    x1 = max(
        0,
        min(int(x1), image_width - 1)
    )

    y1 = max(
        0,
        min(int(y1), image_height - 1)
    )

    x2 = max(
        0,
        min(int(x2), image_width - 1)
    )

    y2 = max(
        0,
        min(int(y2), image_height - 1)
    )

    return [
        x1,
        y1,
        x2,
        y2
    ]

# ============================================================
# REDRAW
# ============================================================

def redraw():

    global display

    scaled_w = max(1, int(image_width * display_scale))
    scaled_h = max(1, int(image_height * display_scale))

    resized = cv2.resize(image, (scaled_w, scaled_h))

    canvas = np.full(
        (DISPLAY_HEIGHT, DISPLAY_WIDTH, 3),
        40,
        dtype=np.uint8
    )

    canvas[
        offset_y:offset_y + scaled_h,
        offset_x:offset_x + scaled_w
    ] = resized

    for i, box in enumerate(boxes):

        x1, y1, x2, y2 = box

        cx1, cy1 = image_to_canvas(x1, y1)
        cx2, cy2 = image_to_canvas(x2, y2)

        thickness = 3 if i == selected_box else 2

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        cv2.rectangle(
            canvas,
            (cx1, cy1),
            (cx2, cy2),
            (0, 255, 0),
            thickness
        )

        # ----------------------------------------------------
        # Resize handles for selected box
        # ----------------------------------------------------

        if i == selected_box:

            handle = 6

            corners = [
                (cx1, cy1),
                (cx2, cy1),
                (cx1, cy2),
                (cx2, cy2)
            ]

            for cx, cy in corners:

                cv2.rectangle(
                    canvas,
                    (
                        cx - handle,
                        cy - handle
                    ),
                    (
                        cx + handle,
                        cy + handle
                    ),
                    (0, 255, 255),
                    -1
                )

    display = canvas

    cv2.imshow(
        "Aircraft Annotation",
        display
    )

# ============================================================
# FIND BOX
# ============================================================

def find_box(x, y):

    for i in range(
        len(boxes) - 1,
        -1,
        -1
    ):

        x1, y1, x2, y2 = boxes[i]

        if (
            x1 <= x <= x2
            and
            y1 <= y <= y2
        ):

            return i

    return -1

# ============================================================
# DETERMINE RESIZE EDGE
# ============================================================

def get_resize_mode(x, y, box):

    x1, y1, x2, y2 = box

    near_left = (
        abs(x - x1)
        <= EDGE_TOLERANCE
    )

    near_right = (
        abs(x - x2)
        <= EDGE_TOLERANCE
    )

    near_top = (
        abs(y - y1)
        <= EDGE_TOLERANCE
    )

    near_bottom = (
        abs(y - y2)
        <= EDGE_TOLERANCE
    )

    # --------------------------------------------------------
    # Corners
    # --------------------------------------------------------

    if near_left and near_top:
        return "top_left"

    if near_right and near_top:
        return "top_right"

    if near_left and near_bottom:
        return "bottom_left"

    if near_right and near_bottom:
        return "bottom_right"

    # --------------------------------------------------------
    # Edges
    # --------------------------------------------------------

    if near_left:
        return "left"

    if near_right:
        return "right"

    if near_top:
        return "top"

    if near_bottom:
        return "bottom"

    return None

# ============================================================
# MOUSE CALLBACK
# ============================================================

def mouse_callback(
    event,
    x,
    y,
    flags,
    param
):

    global drawing_new
    global dragging_box
    global resizing_box

    global start_x
    global start_y

    global drag_offset_x
    global drag_offset_y

    global resize_mode
    global selected_box

    # x, y arrive in on-screen window coordinates. Convert to the
    # original image's pixel coordinates before doing any box math.

    img_x, img_y = canvas_to_image(x, y)

    # ========================================================
    # LEFT BUTTON DOWN
    # ========================================================

    if event == cv2.EVENT_LBUTTONDOWN:

        clicked_box = find_box(
            img_x,
            img_y
        )

        # ----------------------------------------------------
        # Existing box
        # ----------------------------------------------------

        if clicked_box != -1:

            selected_box = clicked_box

            box = boxes[
                selected_box
            ]

            mode = get_resize_mode(
                img_x,
                img_y,
                box
            )

            # ------------------------------------------------
            # Resize
            # ------------------------------------------------

            if mode is not None:

                resizing_box = True

                resize_mode = mode

                start_x = img_x
                start_y = img_y

            # ------------------------------------------------
            # Move
            # ------------------------------------------------

            else:

                dragging_box = True

                x1, y1, x2, y2 = box

                drag_offset_x = (
                    img_x - x1
                )

                drag_offset_y = (
                    img_y - y1
                )

            redraw()

        # ----------------------------------------------------
        # Empty area → new box
        # ----------------------------------------------------

        else:

            selected_box = -1

            drawing_new = True

            start_x = img_x
            start_y = img_y

    # ========================================================
    # MOUSE MOVE
    # ========================================================

    elif event == cv2.EVENT_MOUSEMOVE:

        # ----------------------------------------------------
        # Drawing new box
        # ----------------------------------------------------

        if drawing_new:

            temp = display.copy()

            sx, sy = image_to_canvas(start_x, start_y)

            cv2.rectangle(
                temp,
                (sx, sy),
                (x, y),
                (0, 255, 0),
                2
            )

            cv2.imshow(
                "Aircraft Annotation",
                temp
            )

        # ----------------------------------------------------
        # Moving whole box
        # ----------------------------------------------------

        elif dragging_box:

            x1, y1, x2, y2 = boxes[
                selected_box
            ]

            width = x2 - x1
            height = y2 - y1

            new_x1 = (
                img_x - drag_offset_x
            )

            new_y1 = (
                img_y - drag_offset_y
            )

            new_x2 = (
                new_x1 + width
            )

            new_y2 = (
                new_y1 + height
            )

            boxes[selected_box] = clamp_box(
                [
                    new_x1,
                    new_y1,
                    new_x2,
                    new_y2
                ]
            )

            redraw()

        # ----------------------------------------------------
        # Resizing box
        # ----------------------------------------------------

        elif resizing_box:

            x1, y1, x2, y2 = boxes[
                selected_box
            ]

            # ------------------------------------------------
            # Left edge
            # ------------------------------------------------

            if resize_mode == "left":

                x1 = min(
                    img_x,
                    x2 - MIN_BOX_SIZE
                )

            # ------------------------------------------------
            # Right edge
            # ------------------------------------------------

            elif resize_mode == "right":

                x2 = max(
                    img_x,
                    x1 + MIN_BOX_SIZE
                )

            # ------------------------------------------------
            # Top edge
            # ------------------------------------------------

            elif resize_mode == "top":

                y1 = min(
                    img_y,
                    y2 - MIN_BOX_SIZE
                )

            # ------------------------------------------------
            # Bottom edge
            # ------------------------------------------------

            elif resize_mode == "bottom":

                y2 = max(
                    img_y,
                    y1 + MIN_BOX_SIZE
                )

            # ------------------------------------------------
            # Top-left corner
            # ------------------------------------------------

            elif resize_mode == "top_left":

                x1 = min(
                    img_x,
                    x2 - MIN_BOX_SIZE
                )

                y1 = min(
                    img_y,
                    y2 - MIN_BOX_SIZE
                )

            # ------------------------------------------------
            # Top-right corner
            # ------------------------------------------------

            elif resize_mode == "top_right":

                x2 = max(
                    img_x,
                    x1 + MIN_BOX_SIZE
                )

                y1 = min(
                    img_y,
                    y2 - MIN_BOX_SIZE
                )

            # ------------------------------------------------
            # Bottom-left corner
            # ------------------------------------------------

            elif resize_mode == "bottom_left":

                x1 = min(
                    img_x,
                    x2 - MIN_BOX_SIZE
                )

                y2 = max(
                    img_y,
                    y1 + MIN_BOX_SIZE
                )

            # ------------------------------------------------
            # Bottom-right corner
            # ------------------------------------------------

            elif resize_mode == "bottom_right":

                x2 = max(
                    img_x,
                    x1 + MIN_BOX_SIZE
                )

                y2 = max(
                    img_y,
                    y1 + MIN_BOX_SIZE
                )

            boxes[selected_box] = clamp_box(
                [
                    x1,
                    y1,
                    x2,
                    y2
                ]
            )

            redraw()

    # ========================================================
    # LEFT BUTTON UP
    # ========================================================

    elif event == cv2.EVENT_LBUTTONUP:

        # ----------------------------------------------------
        # Finish creating new box
        # ----------------------------------------------------

        if drawing_new:

            drawing_new = False

            x1 = min(
                start_x,
                img_x
            )

            y1 = min(
                start_y,
                img_y
            )

            x2 = max(
                start_x,
                img_x
            )

            y2 = max(
                start_y,
                img_y
            )

            if (
                abs(x2 - x1)
                >= MIN_BOX_SIZE
                and
                abs(y2 - y1)
                >= MIN_BOX_SIZE
            ):

                boxes.append(
                    clamp_box([
                        x1,
                        y1,
                        x2,
                        y2
                    ])
                )

                selected_box = (
                    len(boxes) - 1
                )

            redraw()

        # ----------------------------------------------------
        # Finish moving
        # ----------------------------------------------------

        dragging_box = False

        # ----------------------------------------------------
        # Finish resizing
        # ----------------------------------------------------

        resizing_box = False

        resize_mode = None

# ============================================================
# SAVE CSV
# ============================================================

def save_annotation(
    image_path,
    class_name,
    image_width,
    image_height,
    boxes
):

    csv_path = image_path.with_suffix(
        ".csv"
    )

    with open(
        csv_path,
        "w",
        newline=""
    ) as f:

        writer = csv.writer(f)

        writer.writerow([
            "filename",
            "width",
            "height",
            "class",
            "xmin",
            "ymin",
            "xmax",
            "ymax"
        ])

        filename = image_path.stem

        for x1, y1, x2, y2 in boxes:

            writer.writerow([
                filename,
                image_width,
                image_height,
                class_name,
                x1,
                y1,
                x2,
                y2
            ])

    print(
        f"\nSaved: {csv_path}"
    )

# ============================================================
# CREATE WINDOW
# ============================================================

cv2.namedWindow(
    "Aircraft Annotation",
    cv2.WINDOW_NORMAL
)

cv2.resizeWindow(
    "Aircraft Annotation",
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT
)

cv2.setMouseCallback(
    "Aircraft Annotation",
    mouse_callback
)

# ============================================================
# MAIN LOOP
# ============================================================

while True:

    image_path, class_name = pending_images[
        current_index
    ]

    # --------------------------------------------------------
    # READ IMAGE
    # --------------------------------------------------------

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            f"Could not read: {image_path}"
        )

        current_index += 1

        if (
            current_index
            >= len(pending_images)
        ):
            break

        continue

    image_height, image_width = (
        image.shape[:2]
    )

    # --------------------------------------------------------
    # FIT IMAGE TO FIXED WINDOW
    # --------------------------------------------------------

    scale_w = DISPLAY_WIDTH / image_width
    scale_h = DISPLAY_HEIGHT / image_height

    display_scale = min(scale_w, scale_h)

    scaled_w = int(image_width * display_scale)
    scaled_h = int(image_height * display_scale)

    offset_x = (DISPLAY_WIDTH - scaled_w) // 2
    offset_y = (DISPLAY_HEIGHT - scaled_h) // 2

    boxes = []

    selected_box = -1

    drawing_new = False
    dragging_box = False
    resizing_box = False

    # --------------------------------------------------------
    # PROGRESS IN TERMINAL ONLY
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        f"[{current_index + 1}/"
        f"{len(pending_images)}]"
    )

    print(
        f"Image : {image_path.name}"
    )

    print(
        f"Class : {class_name}"
    )

    print(
        f"Remaining: "
        f"{len(pending_images) - current_index - 1}"
    )

    print(
        "=" * 60
    )

    redraw()

    # --------------------------------------------------------
    # KEYBOARD LOOP
    # --------------------------------------------------------

    while True:

        key = cv2.waitKey(0) & 0xFF

        # ====================================================
        # SAVE
        # ====================================================

        if key == ord("s"):

            if not boxes:

                print(
                    "No bounding box drawn."
                )

                continue

            save_annotation(
                image_path,
                class_name,
                image_width,
                image_height,
                boxes
            )

            break

        # ====================================================
        # RESET
        # ====================================================

        elif key == ord("r"):

            boxes = []

            selected_box = -1

            redraw()

            print(
                "All boxes cleared."
            )

        # ====================================================
        # SKIP
        # ====================================================

        elif key == ord("n"):

            print(
                "Skipped."
            )

            break

        # ====================================================
        # DELETE WHOLE IMAGE
        # ====================================================

        elif key == ord("d"):

            csv_path = image_path.with_suffix(".csv")

            try:

                image_path.unlink()

                if csv_path.exists():

                    csv_path.unlink()

                print(
                    f"Deleted image: {image_path.name}"
                )

            except Exception as e:

                print(
                    f"Could not delete {image_path.name}: {e}"
                )

            break

        # ====================================================
        # QUIT
        # ====================================================

        elif key == ord("q"):

            cv2.destroyAllWindows()

            print("\n")
            print(
                "=" * 60
            )

            print(
                "Annotation stopped."
            )

            print(
                "Saved CSV files are safe."
            )

            print(
                "=" * 60
            )

            exit()

        # ====================================================
        # DELETE SELECTED BOX
        # ====================================================

        elif key == 8:

            if selected_box >= 0:

                boxes.pop(
                    selected_box
                )

                selected_box = -1

                redraw()

                print(
                    "Selected box deleted."
                )

        # ====================================================
        # ARROW LEFT
        # ====================================================

        elif key == 81:

            if selected_box >= 0:

                x1, y1, x2, y2 = boxes[
                    selected_box
                ]

                boxes[selected_box] = clamp_box(
                    [
                        x1 - MOVE_STEP,
                        y1,
                        x2 - MOVE_STEP,
                        y2
                    ]
                )

                redraw()

        # ====================================================
        # ARROW UP
        # ====================================================

        elif key == 82:

            if selected_box >= 0:

                x1, y1, x2, y2 = boxes[
                    selected_box
                ]

                boxes[selected_box] = clamp_box(
                    [
                        x1,
                        y1 - MOVE_STEP,
                        x2,
                        y2 - MOVE_STEP
                    ]
                )

                redraw()

        # ====================================================
        # ARROW RIGHT
        # ====================================================

        elif key == 83:

            if selected_box >= 0:

                x1, y1, x2, y2 = boxes[
                    selected_box
                ]

                boxes[selected_box] = clamp_box(
                    [
                        x1 + MOVE_STEP,
                        y1,
                        x2 + MOVE_STEP,
                        y2
                    ]
                )

                redraw()

        # ====================================================
        # ARROW DOWN
        # ====================================================

        elif key == 84:

            if selected_box >= 0:

                x1, y1, x2, y2 = boxes[
                    selected_box
                ]

                boxes[selected_box] = clamp_box(
                    [
                        x1,
                        y1 + MOVE_STEP,
                        x2,
                        y2 + MOVE_STEP
                    ]
                )

                redraw()

        # ====================================================
        # RESIZE +
        # ====================================================

        elif key in (
            ord("+"),
            ord("=")
        ):

            if selected_box >= 0:

                x1, y1, x2, y2 = boxes[
                    selected_box
                ]

                boxes[selected_box] = clamp_box(
                    [
                        x1 - RESIZE_STEP,
                        y1 - RESIZE_STEP,
                        x2 + RESIZE_STEP,
                        y2 + RESIZE_STEP
                    ]
                )

                redraw()

        # ====================================================
        # RESIZE -
        # ====================================================

        elif key in (
            ord("-"),
            ord("_")
        ):

            if selected_box >= 0:

                x1, y1, x2, y2 = boxes[
                    selected_box
                ]

                if (
                    x2 - x1
                    > MIN_BOX_SIZE
                    and
                    y2 - y1
                    > MIN_BOX_SIZE
                ):

                    boxes[selected_box] = clamp_box(
                        [
                            x1 + RESIZE_STEP,
                            y1 + RESIZE_STEP,
                            x2 - RESIZE_STEP,
                            y2 - RESIZE_STEP
                        ]
                    )

                    redraw()

        # ====================================================
        # SELECT BOX 1-9
        # ====================================================

        elif ord("1") <= key <= ord("9"):

            box_number = (
                key - ord("1")
            )

            if box_number < len(boxes):

                selected_box = box_number

                redraw()

                print(
                    f"Selected Box "
                    f"{box_number + 1}"
                )

    # --------------------------------------------------------
    # NEXT IMAGE
    # --------------------------------------------------------

    current_index += 1

    if (
        current_index
        >= len(pending_images)
    ):
        break

# ============================================================
# FINISHED
# ============================================================

cv2.destroyAllWindows()

print("\n")
print("=" * 60)
print("ANNOTATION COMPLETED")
print("=" * 60)

print(
    f"Total images      : {len(images)}"
)

print(
    f"Already annotated : {completed_count}"
)

print(
    f"Processed          : {current_index}"
)

print("=" * 60)
