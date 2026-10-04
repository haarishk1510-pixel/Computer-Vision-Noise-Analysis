import cv2
import numpy as np
import pandas as pd

from pathlib import Path
from skimage.metrics import structural_similarity


# ============================================================
# BASIC IMAGE QUALITY METRICS
# ============================================================

def calculate_mae(original, filtered):
    """
    Mean Absolute Error.
    Lower is better.
    """
    original = original.astype(np.float32)
    filtered = filtered.astype(np.float32)

    return float(
        np.mean(np.abs(original - filtered))
    )


def calculate_mse(original, filtered):
    """
    Mean Squared Error.
    Lower is better.
    """
    original = original.astype(np.float32)
    filtered = filtered.astype(np.float32)

    return float(
        np.mean((original - filtered) ** 2)
    )


def calculate_psnr(original, filtered):
    """
    Peak Signal-to-Noise Ratio.
    Higher is better.
    """

    mse = calculate_mse(
        original,
        filtered
    )

    if mse == 0:
        return float("inf")

    return float(
        10 * np.log10(
            (255.0 ** 2) / mse
        )
    )


def calculate_ssim(original, filtered):
    """
    Structural Similarity Index.
    Higher is better.
    """

    return float(
        structural_similarity(
            original,
            filtered,
            data_range=255
        )
    )


# ============================================================
# EDGE PRESERVATION
# ============================================================

def calculate_edge_preservation(
    original,
    filtered
):
    """
    Measures how well important edges are preserved.

    Uses Canny edge detection and calculates
    the F1 score between original and filtered
    edge maps.

    Higher is better.
    """

    original_edges = cv2.Canny(
        original,
        100,
        200
    )

    filtered_edges = cv2.Canny(
        filtered,
        100,
        200
    )

    original_edges = (
        original_edges > 0
    )

    filtered_edges = (
        filtered_edges > 0
    )

    true_positive = np.logical_and(
        original_edges,
        filtered_edges
    ).sum()

    false_positive = np.logical_and(
        ~original_edges,
        filtered_edges
    ).sum()

    false_negative = np.logical_and(
        original_edges,
        ~filtered_edges
    ).sum()

    precision_denominator = (
        true_positive + false_positive
    )

    recall_denominator = (
        true_positive + false_negative
    )

    if precision_denominator == 0:
        precision = 0.0
    else:
        precision = (
            true_positive /
            precision_denominator
        )

    if recall_denominator == 0:
        recall = 0.0
    else:
        recall = (
            true_positive /
            recall_denominator
        )

    if precision + recall == 0:
        return 0.0

    f1 = (
        2 * precision * recall /
        (precision + recall)
    )

    return float(f1)


# ============================================================
# ALL METRICS TOGETHER
# ============================================================

def evaluate_image(
    original,
    filtered
):
    """
    Calculate all quality metrics.
    """

    return {
        "MAE": calculate_mae(
            original,
            filtered
        ),

        "MSE": calculate_mse(
            original,
            filtered
        ),

        "PSNR": calculate_psnr(
            original,
            filtered
        ),

        "SSIM": calculate_ssim(
            original,
            filtered
        ),

        "Edge_Preservation":
            calculate_edge_preservation(
                original,
                filtered
            )
    }


# ============================================================
# QUICK TEST
# ============================================================

def main():

    base_dir = (
        Path(__file__)
        .resolve()
        .parent
        .parent
    )

    original_path = (
        base_dir
        / "data"
        / "original"
        / "smooth_moon.png"
    )

    noisy_path = (
        base_dir
        / "data"
        / "noisy"
        / "smooth_moon_gaussian_sigma_10.png"
    )

    original = cv2.imread(
        str(original_path),
        cv2.IMREAD_GRAYSCALE
    )

    noisy = cv2.imread(
        str(noisy_path),
        cv2.IMREAD_GRAYSCALE
    )

    if original is None:
        raise FileNotFoundError(
            f"Could not read {original_path}"
        )

    if noisy is None:
        raise FileNotFoundError(
            f"Could not read {noisy_path}"
        )

    print()
    print("=" * 60)
    print("IMAGE QUALITY METRICS TEST")
    print("=" * 60)

    results = evaluate_image(
        original,
        noisy
    )

    for metric, value in results.items():

        print(
            f"{metric:<22}: {value:.6f}"
        )

    print("=" * 60)
    print("Metrics test completed successfully.")
    print("=" * 60)


if __name__ == "__main__":
    main()

