import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# Allow importing existing project modules
SRC_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SRC_DIR))

from filters import (
    mean_filter,
    gaussian_filter,
    median_filter,
    bilateral_filter,
    benchmark_filter
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_PATH = BASE_DIR / "data" / "objects" / "objects_coins.png"
OUTPUT_DIR = BASE_DIR / "outputs" / "filter_impact"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# CONFIGURATION
# ============================================================

MIN_AREA = 500

DESCRIPTOR_COLUMNS = [
    "Area",
    "Perimeter",
    "Circularity",
    "Aspect_Ratio",
    "Extent",
    "Solidity",
    "Equivalent_Diameter"
]


# ============================================================
# NOISE GENERATION
# ============================================================

def add_gaussian_noise(image, sigma=25, seed=42):
    rng = np.random.default_rng(seed)

    noise = rng.normal(
        0,
        sigma,
        image.shape
    )

    noisy = image.astype(np.float32) + noise

    return np.clip(
        noisy,
        0,
        255
    ).astype(np.uint8)


def add_salt_pepper_noise(image, amount=0.10, seed=42):
    rng = np.random.default_rng(seed)

    noisy = image.copy()

    random_map = rng.random(image.shape)

    salt = random_map < (amount / 2)
    pepper = random_map > (1 - amount / 2)

    noisy[salt] = 255
    noisy[pepper] = 0

    return noisy


# ============================================================
# SEGMENTATION + OBJECT DESCRIPTORS
# ============================================================

def analyze_objects(image, save_prefix=None):
    """
    Uses the same segmentation logic as the existing
    object_pipeline.py:
        Gaussian blur
        Otsu threshold
        Morphological opening
        Morphological closing
        External contours
        Minimum area filtering
    """

    # 1. Gaussian blur
    blurred = cv2.GaussianBlur(
        image,
        (5, 5),
        0
    )

    # 2. Otsu threshold
    threshold_value, binary = cv2.threshold(
        blurred,
        0,
        255,
        cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # 3. Morphological cleaning
    kernel = np.ones(
        (3, 3),
        np.uint8
    )

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

    # 4. Contours
    contours, _ = cv2.findContours(
        cleaned,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    # 5. Minimum area filtering
    valid_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) >= MIN_AREA
    ]

    # 6. Object measurements
    object_data = []

    for idx, contour in enumerate(
        valid_contours,
        start=1
    ):

        area = cv2.contourArea(contour)

        perimeter = cv2.arcLength(
            contour,
            True
        )

        x, y, w, h = cv2.boundingRect(
            contour
        )

        # Circularity
        if perimeter != 0:
            circularity = (
                4 * np.pi * area
                / (perimeter * perimeter)
            )
        else:
            circularity = 0

        # Aspect ratio
        aspect_ratio = (
            w / h
            if h != 0
            else 0
        )

        # Extent
        bounding_area = w * h

        extent = (
            area / bounding_area
            if bounding_area != 0
            else 0
        )

        # Solidity
        hull = cv2.convexHull(contour)
        hull_area = cv2.contourArea(hull)

        solidity = (
            area / hull_area
            if hull_area != 0
            else 0
        )

        # Equivalent diameter
        equivalent_diameter = np.sqrt(
            (4 * area) / np.pi
        )

        object_data.append({
            "Object_ID": idx,
            "Area": area,
            "Perimeter": perimeter,
            "Circularity": circularity,
            "Aspect_Ratio": aspect_ratio,
            "Extent": extent,
            "Solidity": solidity,
            "Equivalent_Diameter": equivalent_diameter
        })

    df = pd.DataFrame(object_data)

    # Save segmentation image
    if save_prefix is not None:

        segmentation_path = (
            OUTPUT_DIR /
            f"{save_prefix}_segmentation.png"
        )

        cv2.imwrite(
            str(segmentation_path),
            cleaned
        )

        # Detection visualization
        detection = cv2.cvtColor(
            image,
            cv2.COLOR_GRAY2BGR
        )

        for idx, contour in enumerate(
            valid_contours,
            start=1
        ):

            x, y, w, h = cv2.boundingRect(
                contour
            )

            cv2.rectangle(
                detection,
                (x, y),
                (x + w, y + h),
                (255, 0, 0),
                2
            )

            cv2.putText(
                detection,
                f"Obj {idx}",
                (x, max(y - 8, 15)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (0, 255, 0),
                1,
                cv2.LINE_AA
            )

        detection_path = (
            OUTPUT_DIR /
            f"{save_prefix}_objects.png"
        )

        cv2.imwrite(
            str(detection_path),
            detection
        )

    # Summary
    if df.empty:

        summary = {
            "Object_Count": 0,
            "Mean_Area": 0,
            "Mean_Perimeter": 0,
            "Mean_Circularity": 0,
            "Mean_Aspect_Ratio": 0,
            "Mean_Extent": 0,
            "Mean_Solidity": 0,
            "Mean_Equivalent_Diameter": 0
        }

    else:

        summary = {
            "Object_Count": len(df),
            "Mean_Area": df["Area"].mean(),
            "Mean_Perimeter": df["Perimeter"].mean(),
            "Mean_Circularity": df["Circularity"].mean(),
            "Mean_Aspect_Ratio": df["Aspect_Ratio"].mean(),
            "Mean_Extent": df["Extent"].mean(),
            "Mean_Solidity": df["Solidity"].mean(),
            "Mean_Equivalent_Diameter":
                df["Equivalent_Diameter"].mean()
        }

    return threshold_value, df, summary


# ============================================================
# DESCRIPTOR FIDELITY
# ============================================================

def calculate_descriptor_fidelity(
    reference,
    current
):

    errors = []

    for column in DESCRIPTOR_COLUMNS:

        ref_value = reference[column]
        current_value = current[column]

        if abs(ref_value) < 1e-12:
            continue

        relative_error = abs(
            current_value - ref_value
        ) / abs(ref_value)

        errors.append(relative_error)

    if not errors:
        return 0.0

    fidelity = (
        1 - np.mean(errors)
    ) * 100

    return float(
        np.clip(
            fidelity,
            0,
            100
        )
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 72)
    print("FILTER IMPACT ON OBJECT COUNTING AND SHAPE DESCRIPTORS")
    print("=" * 72)

    # --------------------------------------------------------
    # Load original object image
    # --------------------------------------------------------

    image = cv2.imread(
        str(INPUT_PATH),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {INPUT_PATH}"
        )

    print(f"\nInput image : {INPUT_PATH}")
    print(f"Image shape : {image.shape}")

    # --------------------------------------------------------
    # Reference analysis
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("REFERENCE OBJECT ANALYSIS")
    print("=" * 72)

    ref_threshold, ref_df, ref_summary = analyze_objects(
        image,
        save_prefix="reference"
    )

    reference_descriptors = {
        "Area":
            ref_summary["Mean_Area"],
        "Perimeter":
            ref_summary["Mean_Perimeter"],
        "Circularity":
            ref_summary["Mean_Circularity"],
        "Aspect_Ratio":
            ref_summary["Mean_Aspect_Ratio"],
        "Extent":
            ref_summary["Mean_Extent"],
        "Solidity":
            ref_summary["Mean_Solidity"],
        "Equivalent_Diameter":
            ref_summary["Mean_Equivalent_Diameter"]
    }

    reference_count = ref_summary["Object_Count"]

    print(f"Reference object count : {reference_count}")
    print(f"Reference Otsu threshold : {ref_threshold:.2f}")

    print("\nReference descriptors:")

    for name, value in reference_descriptors.items():
        print(f"{name:<22}: {value:.4f}")

    # --------------------------------------------------------
    # Generate controlled noisy object images
    # --------------------------------------------------------

    noisy_images = {
        "Gaussian_Noise":
            add_gaussian_noise(
                image,
                sigma=25,
                seed=42
            ),

        "Salt_Pepper_Noise":
            add_salt_pepper_noise(
                image,
                amount=0.10,
                seed=42
            )
    }

    # --------------------------------------------------------
    # Filters
    # --------------------------------------------------------

    filters = {

        "Mean": lambda img:
            mean_filter(img, 5),

        "Gaussian": lambda img:
            gaussian_filter(
                img,
                5,
                1.0
            ),

        "Median": lambda img:
            median_filter(
                img,
                5
            ),

        "Bilateral": lambda img:
            bilateral_filter(
                img,
                9,
                75,
                75
            )
    }

    results = []

    print("\n" + "=" * 72)
    print("FILTER-WISE OBJECT ANALYSIS")
    print("=" * 72)

    # --------------------------------------------------------
    # Run experiment
    # --------------------------------------------------------

    for noise_name, noisy_image in noisy_images.items():

        print(f"\nNOISE TYPE: {noise_name}")
        print("-" * 72)

        noisy_path = (
            OUTPUT_DIR /
            f"{noise_name.lower()}.png"
        )

        cv2.imwrite(
            str(noisy_path),
            noisy_image
        )

        for filter_name, filter_function in filters.items():

            print(
                f"\n{filter_name} Filter"
            )

            # Apply filter + measure runtime
            filtered_image, runtime = (
                benchmark_filter(
                    filter_function,
                    noisy_image
                )
            )

            prefix = (
                f"{noise_name.lower()}_"
                f"{filter_name.lower()}"
            )

            filtered_path = (
                OUTPUT_DIR /
                f"{prefix}_filtered.png"
            )

            cv2.imwrite(
                str(filtered_path),
                filtered_image
            )

            # Segment and analyze
            threshold, df, summary = (
                analyze_objects(
                    filtered_image,
                    save_prefix=prefix
                )
            )

            object_count = summary[
                "Object_Count"
            ]

            # Count accuracy
            count_error = abs(
                object_count -
                reference_count
            )

            count_accuracy = (
                max(
                    0,
                    1 -
                    count_error /
                    max(reference_count, 1)
                )
                * 100
            )

            # Descriptor fidelity
            current_descriptors = {
                "Area":
                    summary["Mean_Area"],
                "Perimeter":
                    summary["Mean_Perimeter"],
                "Circularity":
                    summary["Mean_Circularity"],
                "Aspect_Ratio":
                    summary["Mean_Aspect_Ratio"],
                "Extent":
                    summary["Mean_Extent"],
                "Solidity":
                    summary["Mean_Solidity"],
                "Equivalent_Diameter":
                    summary["Mean_Equivalent_Diameter"]
            }

            descriptor_fidelity = (
                calculate_descriptor_fidelity(
                    reference_descriptors,
                    current_descriptors
                )
            )

            results.append({

                "Noise":
                    noise_name,

                "Filter":
                    filter_name,

                "Runtime_ms":
                    runtime,

                "Reference_Count":
                    reference_count,

                "Detected_Count":
                    object_count,

                "Count_Error":
                    count_error,

                "Count_Accuracy_Percent":
                    count_accuracy,

                "Descriptor_Fidelity_Percent":
                    descriptor_fidelity,

                "Mean_Area":
                    summary["Mean_Area"],

                "Mean_Perimeter":
                    summary["Mean_Perimeter"],

                "Mean_Circularity":
                    summary["Mean_Circularity"],

                "Mean_Aspect_Ratio":
                    summary["Mean_Aspect_Ratio"],

                "Mean_Extent":
                    summary["Mean_Extent"],

                "Mean_Solidity":
                    summary["Mean_Solidity"],

                "Mean_Equivalent_Diameter":
                    summary[
                        "Mean_Equivalent_Diameter"
                    ],

                "Otsu_Threshold":
                    threshold
            })

            print(
                f"  Objects detected      : {object_count}"
            )

            print(
                f"  Count accuracy        : "
                f"{count_accuracy:.2f}%"
            )

            print(
                f"  Descriptor fidelity   : "
                f"{descriptor_fidelity:.2f}%"
            )

            print(
                f"  Runtime               : "
                f"{runtime:.2f} ms"
            )

    # --------------------------------------------------------
    # DataFrame
    # --------------------------------------------------------

    results_df = pd.DataFrame(results)

    results_csv = (
        OUTPUT_DIR /
        "filter_impact_results.csv"
    )

    results_df.to_csv(
        results_csv,
        index=False
    )

    # --------------------------------------------------------
    # Aggregate by filter
    # --------------------------------------------------------

    aggregate = (
        results_df
        .groupby("Filter")
        .agg({
            "Count_Accuracy_Percent": "mean",
            "Descriptor_Fidelity_Percent": "mean",
            "Runtime_ms": "mean",
            "Count_Error": "mean"
        })
        .reset_index()
    )

    # Normalize runtime:
    # lower runtime = better
    max_runtime = aggregate[
        "Runtime_ms"
    ].max()

    if max_runtime > 0:

        aggregate[
            "Runtime_Score_Percent"
        ] = (
            1 -
            aggregate["Runtime_ms"]
            / max_runtime
        ) * 100

    else:

        aggregate[
            "Runtime_Score_Percent"
        ] = 100

    # Final object-analysis score
    #
    # Count accuracy       = 50%
    # Descriptor fidelity = 40%
    # Runtime              = 10%

    aggregate["Object_Impact_Score"] = (

        0.50 *
        aggregate[
            "Count_Accuracy_Percent"
        ]

        +

        0.40 *
        aggregate[
            "Descriptor_Fidelity_Percent"
        ]

        +

        0.10 *
        aggregate[
            "Runtime_Score_Percent"
        ]
    )

    aggregate = aggregate.sort_values(
        "Object_Impact_Score",
        ascending=False
    )

    aggregate_csv = (
        OUTPUT_DIR /
        "filter_object_impact_ranking.csv"
    )

    aggregate.to_csv(
        aggregate_csv,
        index=False
    )

    # --------------------------------------------------------
    # Descriptor stability summary
    # --------------------------------------------------------

    descriptor_rows = []

    for filter_name in filters.keys():

        subset = results_df[
            results_df["Filter"] ==
            filter_name
        ]

        row = {
            "Filter": filter_name
        }

        for descriptor in [
            "Mean_Area",
            "Mean_Perimeter",
            "Mean_Circularity",
            "Mean_Aspect_Ratio",
            "Mean_Extent",
            "Mean_Solidity",
            "Mean_Equivalent_Diameter"
        ]:

            values = subset[
                descriptor
            ].values

            reference_column = {
                "Mean_Area":
                    reference_descriptors[
                        "Area"
                    ],

                "Mean_Perimeter":
                    reference_descriptors[
                        "Perimeter"
                    ],

                "Mean_Circularity":
                    reference_descriptors[
                        "Circularity"
                    ],

                "Mean_Aspect_Ratio":
                    reference_descriptors[
                        "Aspect_Ratio"
                    ],

                "Mean_Extent":
                    reference_descriptors[
                        "Extent"
                    ],

                "Mean_Solidity":
                    reference_descriptors[
                        "Solidity"
                    ],

                "Mean_Equivalent_Diameter":
                    reference_descriptors[
                        "Equivalent_Diameter"
                    ]
            }[descriptor]

            if abs(reference_column) > 1e-12:

                relative_error = (
                    np.mean(
                        np.abs(
                            values -
                            reference_column
                        )
                    )
                    / abs(reference_column)
                ) * 100

            else:

                relative_error = 0

            row[
                descriptor +
                "_Relative_Error_Percent"
            ] = relative_error

        descriptor_rows.append(row)

    descriptor_df = pd.DataFrame(
        descriptor_rows
    )

    descriptor_csv = (
        OUTPUT_DIR /
        "descriptor_stability.csv"
    )

    descriptor_df.to_csv(
        descriptor_csv,
        index=False
    )

    # --------------------------------------------------------
    # Visualization
    # --------------------------------------------------------

    plt.figure(figsize=(11, 7))

    x = np.arange(
        len(aggregate)
    )

    width = 0.25

    plt.bar(
        x - width,
        aggregate[
            "Count_Accuracy_Percent"
        ],
        width,
        label="Count Accuracy (%)"
    )

    plt.bar(
        x,
        aggregate[
            "Descriptor_Fidelity_Percent"
        ],
        width,
        label="Descriptor Fidelity (%)"
    )

    plt.bar(
        x + width,
        aggregate[
            "Object_Impact_Score"
        ],
        width,
        label="Overall Impact Score"
    )

    plt.xticks(
        x,
        aggregate["Filter"]
    )

    plt.ylabel("Score (%)")

    plt.xlabel("Filter")

    plt.title(
        "Effect of Image Filtering on Object Analysis"
    )

    plt.ylim(
        0,
        110
    )

    plt.legend()

    plt.grid(
        axis="y",
        alpha=0.25
    )

    plt.tight_layout()

    chart_path = (
        OUTPUT_DIR /
        "filter_impact_comparison.png"
    )

    plt.savefig(
        chart_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    # --------------------------------------------------------
    # Final summary
    # --------------------------------------------------------

    best_filter = aggregate.iloc[0]

    summary_path = (
        OUTPUT_DIR /
        "filter_impact_summary.txt"
    )

    with open(
        summary_path,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "FILTER IMPACT ON OBJECT ANALYSIS\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        file.write(
            f"Reference object count: "
            f"{reference_count}\n\n"
        )

        file.write(
            "OBJECT-ANALYSIS FILTER RANKING\n"
        )

        file.write(
            "-" * 60 + "\n"
        )

        for rank, (_, row) in enumerate(
            aggregate.iterrows(),
            start=1
        ):

            file.write(
                f"{rank}. {row['Filter']}\n"
            )

            file.write(
                f"   Count Accuracy: "
                f"{row['Count_Accuracy_Percent']:.2f}%\n"
            )

            file.write(
                f"   Descriptor Fidelity: "
                f"{row['Descriptor_Fidelity_Percent']:.2f}%\n"
            )

            file.write(
                f"   Mean Runtime: "
                f"{row['Runtime_ms']:.2f} ms\n"
            )

            file.write(
                f"   Object Impact Score: "
                f"{row['Object_Impact_Score']:.2f}\n\n"
            )

        file.write(
            "=" * 60 + "\n"
        )

        file.write(
            f"BEST FILTER FOR OBJECT ANALYSIS: "
            f"{best_filter['Filter']}\n"
        )

        file.write(
            f"OBJECT IMPACT SCORE: "
            f"{best_filter['Object_Impact_Score']:.2f}\n"
        )

    # --------------------------------------------------------
    # Console output
    # --------------------------------------------------------

    print("\n" + "=" * 72)
    print("FINAL OBJECT-ANALYSIS FILTER RANKING")
    print("=" * 72)

    for rank, (_, row) in enumerate(
        aggregate.iterrows(),
        start=1
    ):

        print(
            f"{rank}. {row['Filter']:<12} "
            f"| Count Accuracy: "
            f"{row['Count_Accuracy_Percent']:>6.2f}% "
            f"| Descriptor Fidelity: "
            f"{row['Descriptor_Fidelity_Percent']:>6.2f}% "
            f"| Score: "
            f"{row['Object_Impact_Score']:>6.2f}"
        )

    print("\n" + "=" * 72)

    print(
        f"BEST FILTER FOR OBJECT ANALYSIS: "
        f"{best_filter['Filter']}"
    )

    print(
        f"Object impact score: "
        f"{best_filter['Object_Impact_Score']:.2f}"
    )

    print("\nGenerated files:")

    print(
        f" -> {results_csv}"
    )

    print(
        f" -> {aggregate_csv}"
    )

    print(
        f" -> {descriptor_csv}"
    )

    print(
        f" -> {chart_path}"
    )

    print(
        f" -> {summary_path}"
    )

    print("\n" + "=" * 72)
    print("FILTER IMPACT ANALYSIS COMPLETED SUCCESSFULLY")
    print("=" * 72)


if __name__ == "__main__":
    main()
