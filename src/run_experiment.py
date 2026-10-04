from pathlib import Path
import time

import cv2
import pandas as pd

from filters import (
    mean_filter,
    gaussian_filter,
    median_filter,
    bilateral_filter
)

from metrics import evaluate_image


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent.parent

ORIGINAL_DIR = BASE_DIR / "data" / "original"
NOISY_DIR = BASE_DIR / "data" / "noisy"
FILTERED_DIR = BASE_DIR / "data" / "filtered"

METRICS_DIR = BASE_DIR / "outputs" / "metrics"

FILTERED_DIR.mkdir(
    parents=True,
    exist_ok=True
)

METRICS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# DATASET DEFINITIONS
# ============================================================

ORIGINAL_IMAGES = [
    "real_photo_coffee.png",
    "smooth_moon.png",
    "strong_edges_camera.png",
    "texture_grass.png"
]


NOISE_TYPES = [
    "gaussian_sigma_10",
    "gaussian_sigma_25",
    "salt_pepper_5percent",
    "salt_pepper_20percent",
    "speckle",
    "poisson"
]


# ============================================================
# FILTER CONFIGURATION
# ============================================================

FILTERS = {

    "Mean": lambda image:
        mean_filter(
            image,
            kernel_size=5
        ),

    "Gaussian": lambda image:
        gaussian_filter(
            image,
            kernel_size=5,
            sigma=1.0
        ),

    "Median": lambda image:
        median_filter(
            image,
            kernel_size=5
        ),

    "Bilateral": lambda image:
        bilateral_filter(
            image,
            diameter=9,
            sigma_color=75,
            sigma_space=75
        )
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_original_image(noisy_filename):

    for original_name in ORIGINAL_IMAGES:

        prefix = original_name.replace(
            ".png",
            ""
        )

        if noisy_filename.startswith(prefix + "_"):

            return (
                ORIGINAL_DIR /
                original_name
            )

    return None


def identify_noise(noisy_filename):

    for noise in NOISE_TYPES:

        if noisy_filename.endswith(
            noise + ".png"
        ):

            return noise

    return "unknown"


# ============================================================
# MAIN EXPERIMENT
# ============================================================

def main():

    results = []

    noisy_files = sorted(
        NOISY_DIR.glob("*.png")
    )

    print()
    print("=" * 75)
    print("COMPLETE FILTER EXPERIMENT")
    print("=" * 75)

    print(
        f"Found {len(noisy_files)} noisy images."
    )

    print(
        f"Filters to test: {len(FILTERS)}"
    )

    print(
        f"Expected filtered images: "
        f"{len(noisy_files) * len(FILTERS)}"
    )

    print("=" * 75)

    total_start = time.perf_counter()

    for image_number, noisy_path in enumerate(
        noisy_files,
        start=1
    ):

        noisy_filename = noisy_path.name

        original_path = find_original_image(
            noisy_filename
        )

        if original_path is None:

            print(
                f"WARNING: Could not identify "
                f"original for {noisy_filename}"
            )

            continue

        noise_type = identify_noise(
            noisy_filename
        )

        original = cv2.imread(
            str(original_path),
            cv2.IMREAD_GRAYSCALE
        )

        noisy = cv2.imread(
            str(noisy_path),
            cv2.IMREAD_GRAYSCALE
        )

        if original is None or noisy is None:

            print(
                f"ERROR reading {noisy_filename}"
            )

            continue

        print()
        print(
            f"[{image_number:02d}/"
            f"{len(noisy_files):02d}] "
            f"{noisy_filename}"
        )

        print(
            f"    Noise: {noise_type}"
        )

        for filter_name, filter_function in FILTERS.items():

            print(
                f"    → {filter_name:<10}",
                end=" ",
                flush=True
            )

            start_time = time.perf_counter()

            filtered = filter_function(
                noisy
            )

            runtime_ms = (
                time.perf_counter()
                - start_time
            ) * 1000

            # ------------------------------------------------
            # SAVE FILTERED IMAGE
            # ------------------------------------------------

            original_stem = (
                noisy_filename
                .replace(".png", "")
            )

            output_filename = (
                f"{original_stem}"
                f"_{filter_name.lower()}.png"
            )

            output_path = (
                FILTERED_DIR /
                output_filename
            )

            cv2.imwrite(
                str(output_path),
                filtered
            )

            # ------------------------------------------------
            # CALCULATE METRICS
            # ------------------------------------------------

            metric_values = evaluate_image(
                original,
                filtered
            )

            row = {

                "Image":
                    original_path.name,

                "Noisy_Image":
                    noisy_filename,

                "Noise":
                    noise_type,

                "Filter":
                    filter_name,

                "MAE":
                    metric_values["MAE"],

                "MSE":
                    metric_values["MSE"],

                "PSNR":
                    metric_values["PSNR"],

                "SSIM":
                    metric_values["SSIM"],

                "Edge_Preservation":
                    metric_values[
                        "Edge_Preservation"
                    ],

                "Runtime_ms":
                    runtime_ms
            }

            results.append(row)

            print(
                f"{runtime_ms:8.2f} ms"
            )

    # ========================================================
    # SAVE RESULTS
    # ========================================================

    dataframe = pd.DataFrame(
        results
    )

    csv_path = (
        METRICS_DIR /
        "filter_results.csv"
    )

    dataframe.to_csv(
        csv_path,
        index=False
    )

    total_runtime = (
        time.perf_counter()
        - total_start
    )

    print()
    print("=" * 75)
    print("EXPERIMENT COMPLETED")
    print("=" * 75)

    print(
        f"Filtered images generated: "
        f"{len(dataframe)}"
    )

    print(
        f"CSV saved to:"
    )

    print(
        csv_path
    )

    print(
        f"Total experiment time: "
        f"{total_runtime:.2f} seconds"
    )

    print("=" * 75)


if __name__ == "__main__":
    main()
