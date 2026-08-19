from pathlib import Path

from PIL import Image
from torch.utils.data import Dataset


IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp",
}


class AircraftDataset(Dataset):

    def __init__(
        self,
        dataframe,
        image_dir,
        transform=None,
        class_to_idx=None,
    ):

        self.df = dataframe.reset_index(drop=True)

        self.image_dir = Path(image_dir)

        self.transform = transform

        # ----------------------------------------------------
        # Class mapping
        # ----------------------------------------------------

        if class_to_idx is None:

            classes = sorted(
                self.df["class"]
                .astype(str)
                .unique()
            )

            self.class_to_idx = {
                cls: idx
                for idx, cls in enumerate(classes)
            }

        else:

            self.class_to_idx = class_to_idx

        self.idx_to_class = {
            idx: cls
            for cls, idx in self.class_to_idx.items()
        }

        # ----------------------------------------------------
        # Cache image paths
        # ----------------------------------------------------

        print("Indexing image paths...")

        self.path_cache = {}

        for filename in self.df["filename"].unique():

            filename = str(filename).strip()

            resolved_path = (
                self._resolve_image_path(filename)
            )

            if resolved_path is not None:

                self.path_cache[filename] = (
                    resolved_path
                )

        print(
            f"Indexed {len(self.path_cache)} "
            f"unique images."
        )

    # ========================================================
    # Resolve image
    # ========================================================

    def _resolve_image_path(self, filename):

        filename = str(filename).strip()

        filename_path = Path(filename)

        # ----------------------------------------------------
        # Direct filename
        # ----------------------------------------------------

        direct_path = (
            self.image_dir /
            filename_path.name
        )

        if direct_path.is_file():

            return direct_path

        # ----------------------------------------------------
        # Filename already has extension
        # ----------------------------------------------------

        if (
            filename_path.suffix.lower()
            in IMAGE_EXTENSIONS
        ):

            candidate = (
                self.image_dir /
                filename_path.name
            )

            if candidate.is_file():

                return candidate

        # ----------------------------------------------------
        # Try possible extensions
        # ----------------------------------------------------

        else:

            for ext in IMAGE_EXTENSIONS:

                candidate = (
                    self.image_dir /
                    f"{filename}{ext}"
                )

                if candidate.is_file():

                    return candidate

        # ----------------------------------------------------
        # Recursive fallback
        # ----------------------------------------------------

        stem = filename_path.stem

        for ext in IMAGE_EXTENSIONS:

            matches = list(
                self.image_dir.rglob(
                    f"{stem}{ext}"
                )
            )

            if matches:

                return matches[0]

        return None

    # ========================================================
    # Length
    # ========================================================

    def __len__(self):

        return len(self.df)

    # ========================================================
    # Get item
    # ========================================================

    def __getitem__(self, index):

        row = self.df.iloc[index]

        filename = str(
            row["filename"]
        ).strip()

        image_path = self.path_cache.get(
            filename
        )

        if (
            image_path is None
            or not image_path.is_file()
        ):

            raise FileNotFoundError(
                f"Could not find image: "
                f"{filename}"
            )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        # ----------------------------------------------------
        # Bounding box
        # ----------------------------------------------------

        xmin = max(
            0,
            int(row["xmin"])
        )

        ymin = max(
            0,
            int(row["ymin"])
        )

        xmax = min(
            image.width,
            int(row["xmax"])
        )

        ymax = min(
            image.height,
            int(row["ymax"])
        )

        if (
            xmax <= xmin
            or ymax <= ymin
        ):

            raise ValueError(
                f"Invalid bounding box for "
                f"{filename}: "
                f"({xmin}, {ymin}, "
                f"{xmax}, {ymax})"
            )

        # ----------------------------------------------------
        # Crop aircraft
        # ----------------------------------------------------

        image = image.crop(
            (
                xmin,
                ymin,
                xmax,
                ymax,
            )
        )

        # ----------------------------------------------------
        # Transform
        # ----------------------------------------------------

        if self.transform is not None:

            image = self.transform(image)

        # ----------------------------------------------------
        # Label
        # ----------------------------------------------------

        aircraft_class = str(
            row["class"]
        )

        label = self.class_to_idx[
            aircraft_class
        ]

        return image, label
