import cv2
import numpy as np
import pandas as pd
import os
import math

# ============================================================
# REGION & SHAPE DESCRIPTOR ANALYSIS
# ============================================================

INPUT_IMAGE = "outputs/objects/03_morphology.png"
OUTPUT_DIR = "outputs/shape_analysis"

os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 65)
print("REGION / SHAPE DESCRIPTOR ANALYSIS")
print("=" * 65)

# ------------------------------------------------------------
# Load image
# ------------------------------------------------------------

image = cv2.imread(INPUT_IMAGE, cv2.IMREAD_GRAYSCALE)

if image is None:
    raise FileNotFoundError(f"Input image not found: {INPUT_IMAGE}")

print(f"\nInput image : {INPUT_IMAGE}")
print(f"Image shape : {image.shape}")

# ------------------------------------------------------------
# Ensure binary image
# ------------------------------------------------------------

_, binary = cv2.threshold(
    image, 0, 255,
    cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# ------------------------------------------------------------
# Find external contours
# ------------------------------------------------------------

contours, _ = cv2.findContours(
    binary,
    cv2.RETR_EXTERNAL,
    cv2.CHAIN_APPROX_SIMPLE
)

print(f"Total contours found : {len(contours)}")

# ------------------------------------------------------------
# Filter small objects
# ------------------------------------------------------------

MIN_AREA = 100

valid_contours = [
    c for c in contours
    if cv2.contourArea(c) >= MIN_AREA
]

print(f"Valid objects         : {len(valid_contours)}")

# ------------------------------------------------------------
# Create visualization
# ------------------------------------------------------------

original_color = cv2.cvtColor(binary, cv2.COLOR_GRAY2BGR)

records = []

# ------------------------------------------------------------
# Extract descriptors
# ------------------------------------------------------------

for i, contour in enumerate(valid_contours, start=1):

    area = cv2.contourArea(contour)

    perimeter = cv2.arcLength(contour, True)

    # Circularity
    if perimeter > 0:
        circularity = (4 * math.pi * area) / (perimeter ** 2)
    else:
        circularity = 0

    # Bounding rectangle
    x, y, w, h = cv2.boundingRect(contour)

    # Aspect ratio
    aspect_ratio = w / h if h > 0 else 0

    # Extent
    bounding_area = w * h
    extent = area / bounding_area if bounding_area > 0 else 0

    # Convex hull
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    # Solidity
    solidity = area / hull_area if hull_area > 0 else 0

    # Equivalent diameter
    equivalent_diameter = math.sqrt(
        (4 * area) / math.pi
    )

    # Moments / centroid
    moments = cv2.moments(contour)

    if moments["m00"] != 0:
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
    else:
        cx = 0
        cy = 0

    # --------------------------------------------------------
    # Store measurements
    # --------------------------------------------------------

    records.append({
        "Object_ID": i,
        "Area": area,
        "Perimeter": perimeter,
        "Circularity": circularity,
        "BoundingBox_X": x,
        "BoundingBox_Y": y,
        "Width": w,
        "Height": h,
        "Aspect_Ratio": aspect_ratio,
        "Extent": extent,
        "Solidity": solidity,
        "Equivalent_Diameter": equivalent_diameter,
        "Centroid_X": cx,
        "Centroid_Y": cy
    })

    # --------------------------------------------------------
    # Draw bounding box
    # --------------------------------------------------------

    cv2.rectangle(
        original_color,
        (x, y),
        (x + w, y + h),
        (255, 0, 0),
        2
    )

    # Centroid
    cv2.circle(
        original_color,
        (int(cx), int(cy)),
        4,
        (0, 0, 255),
        -1
    )

    # Object label
    cv2.putText(
        original_color,
        f"Obj {i}",
        (x, max(y - 5, 15)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (0, 255, 0),
        1,
        cv2.LINE_AA
    )

# ------------------------------------------------------------
# DataFrame
# ------------------------------------------------------------

df = pd.DataFrame(records)

csv_path = os.path.join(
    OUTPUT_DIR,
    "region_shape_descriptors.csv"
)

df.to_csv(csv_path, index=False)

# ------------------------------------------------------------
# Summary statistics
# ------------------------------------------------------------

summary = df[
    [
        "Area",
        "Perimeter",
        "Circularity",
        "Aspect_Ratio",
        "Extent",
        "Solidity",
        "Equivalent_Diameter"
    ]
].agg(["mean", "min", "max"])

summary_path = os.path.join(
    OUTPUT_DIR,
    "shape_descriptor_summary.csv"
)

summary.to_csv(summary_path)

# ------------------------------------------------------------
# Save visualization
# ------------------------------------------------------------

visualization_path = os.path.join(
    OUTPUT_DIR,
    "shape_descriptor_visualization.png"
)

cv2.imwrite(
    visualization_path,
    original_color
)

# ------------------------------------------------------------
# Print results
# ------------------------------------------------------------

print("\n" + "=" * 65)
print("SHAPE DESCRIPTOR RESULTS")
print("=" * 65)

print(f"\nObjects analysed : {len(df)}")

if not df.empty:

    print(f"Mean Area       : {df['Area'].mean():.2f}")
    print(f"Mean Perimeter  : {df['Perimeter'].mean():.2f}")
    print(f"Mean Circularity: {df['Circularity'].mean():.4f}")
    print(f"Mean Aspect Ratio: {df['Aspect_Ratio'].mean():.4f}")
    print(f"Mean Extent     : {df['Extent'].mean():.4f}")
    print(f"Mean Solidity   : {df['Solidity'].mean():.4f}")
    print(
        f"Mean Equivalent Diameter: "
        f"{df['Equivalent_Diameter'].mean():.2f}"
    )

print("\nGenerated files:")
print(f" -> {csv_path}")
print(f" -> {summary_path}")
print(f" -> {visualization_path}")

print("\n" + "=" * 65)
print("REGION / SHAPE ANALYSIS COMPLETED SUCCESSFULLY")
print("=" * 65)
