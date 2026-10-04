from pathlib import Path

import cv2
from skimage import data


# Project directories
BASE_DIR = Path(__file__).resolve().parent.parent
ORIGINAL_DIR = BASE_DIR / "data" / "original"
OBJECTS_DIR = BASE_DIR / "data" / "objects"

ORIGINAL_DIR.mkdir(parents=True, exist_ok=True)
OBJECTS_DIR.mkdir(parents=True, exist_ok=True)


def save_grayscale(image, output_path):
    """Convert an image to grayscale and save it as PNG."""
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    success = cv2.imwrite(str(output_path), gray)

    if not success:
        raise RuntimeError(f"Could not save: {output_path}")

    print(f"✓ Saved: {output_path}")


def main():
    # 1. Smooth regions
    save_grayscale(
        data.moon(),
        ORIGINAL_DIR / "smooth_moon.png"
    )

    # 2. Fine texture
    save_grayscale(
        data.grass(),
        ORIGINAL_DIR / "texture_grass.png"
    )

    # 3. Strong edges
    save_grayscale(
        data.camera(),
        ORIGINAL_DIR / "strong_edges_camera.png"
    )

    # 4. Real photograph
    save_grayscale(
        data.coffee(),
        ORIGINAL_DIR / "real_photo_coffee.png"
    )

    # 5. Multiple objects for segmentation/counting
    save_grayscale(
        data.coins(),
        OBJECTS_DIR / "objects_coins.png"
    )

    print("\nDataset preparation complete.")


if __name__ == "__main__":
    main()
