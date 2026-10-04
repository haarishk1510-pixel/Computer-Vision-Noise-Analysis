import cv2
import numpy as np
from pathlib import Path
import pandas as pd
import time

# ---------------------------------------------------------
# PATHS
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "objects" / "objects_coins.png"
OUTPUT_DIR = BASE_DIR / "outputs" / "objects"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ---------------------------------------------------------
# LOAD IMAGE
# ---------------------------------------------------------
image = cv2.imread(str(INPUT_PATH), cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Could not load: {INPUT_PATH}")

print("=" * 60)
print("OBJECT DETECTION / SEGMENTATION PIPELINE")
print("=" * 60)

print(f"Input image : {INPUT_PATH}")
print(f"Image shape : {image.shape}")


# ---------------------------------------------------------
# 1. GAUSSIAN BLUR
# ---------------------------------------------------------
blurred = cv2.GaussianBlur(image, (5, 5), 0)

cv2.imwrite(
    str(OUTPUT_DIR / "01_blurred.png"),
    blurred
)


# ---------------------------------------------------------
# 2. OTSU THRESHOLDING
# ---------------------------------------------------------
threshold_value, binary = cv2.threshold(
    blurred,
    0,
    255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

cv2.imwrite(
    str(OUTPUT_DIR / "02_binary_otsu.png"),
    binary
)

print(f"Otsu threshold : {threshold_value:.2f}")


# ---------------------------------------------------------
# 3. MORPHOLOGICAL CLEANING
# ---------------------------------------------------------
kernel = np.ones((3, 3), np.uint8)

cleaned = cv2.morphologyEx(
    binary,
    cv2.MORPH_OPEN,
    kernel,
    iterations=1
)

cleaned = cv2.morphologyEx(
    cleaned,
    cv2.MORPH_CLOSE,
    kernel,
    iterations=2
)

cv2.imwrite(
    str(OUTPUT_DIR / "03_morphology.png"),
    cleaned
)


# ---------------------------------------------------------
# 4. CONTOUR DETECTION
# ---------------------------------------------------------
contours, _ = cv2.findContours(
    cleaned,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print(f"Total contours detected : {len(contours)}")


# ---------------------------------------------------------
# 5. FILTER SMALL CONTOURS
# ---------------------------------------------------------
MIN_AREA = 500

valid_contours = [
    contour
    for contour in contours
    if cv2.contourArea(contour) >= MIN_AREA
]

print(f"Valid objects detected  : {len(valid_contours)}")


# ---------------------------------------------------------
# 6. DRAW DETECTED OBJECTS
# ---------------------------------------------------------
result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)

object_data = []

for idx, contour in enumerate(valid_contours, start=1):

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    x, y, w, h = cv2.boundingRect(contour)

    # Centroid
    moments = cv2.moments(contour)

    if moments["m00"] != 0:
        cx = int(moments["m10"] / moments["m00"])
        cy = int(moments["m01"] / moments["m00"])
    else:
        cx = x + w // 2
        cy = y + h // 2

    # Circularity
    if perimeter != 0:
        circularity = (
            4 * np.pi * area / (perimeter * perimeter)
        )
    else:
        circularity = 0

    object_data.append({
        "Object_ID": idx,
        "Area": area,
        "Perimeter": perimeter,
        "Centroid_X": cx,
        "Centroid_Y": cy,
        "Width": w,
        "Height": h,
        "Circularity": circularity
    })

    # Bounding box
    cv2.rectangle(
        result,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )

    # Centroid
    cv2.circle(
        result,
        (cx, cy),
        5,
        (0, 0, 255),
        -1
    )

    # Object label
    cv2.putText(
        result,
        f"Object {idx}",
        (x, max(y - 8, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )


# ---------------------------------------------------------
# 7. SAVE RESULT
# ---------------------------------------------------------
result_path = OUTPUT_DIR / "04_detected_objects.png"

cv2.imwrite(
    str(result_path),
    result
)


# ---------------------------------------------------------
# 8. SAVE OBJECT MEASUREMENTS
# ---------------------------------------------------------
df = pd.DataFrame(object_data)

csv_path = OUTPUT_DIR / "object_measurements.csv"

df.to_csv(
    csv_path,
    index=False
)


# ---------------------------------------------------------
# 9. SUMMARY
# ---------------------------------------------------------
print()
print("=" * 60)
print("OBJECT ANALYSIS RESULTS")
print("=" * 60)

print(f"Objects detected : {len(valid_contours)}")

if not df.empty:
    print(f"Mean area        : {df['Area'].mean():.2f}")
    print(f"Mean perimeter   : {df['Perimeter'].mean():.2f}")
    print(f"Mean circularity : {df['Circularity'].mean():.4f}")

print()
print(f"Saved detection result : {result_path}")
print(f"Saved measurements     : {csv_path}")

print("=" * 60)
print("OBJECT PIPELINE COMPLETED SUCCESSFULLY")
print("=" * 60)
