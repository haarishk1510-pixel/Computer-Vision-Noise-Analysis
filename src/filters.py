from pathlib import Path
import time

import cv2
import numpy as np


# ============================================================
# FROM-SCRATCH FILTERS
# ============================================================

def mean_filter(image, kernel_size=5):
    """
    Mean / box filter implemented from scratch using NumPy.
    """

    pad = kernel_size // 2

    padded = np.pad(
        image.astype(np.float32),
        pad,
        mode="reflect"
    )

    output = np.zeros_like(image, dtype=np.float32)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            window = padded[
                i:i + kernel_size,
                j:j + kernel_size
            ]

            output[i, j] = np.mean(window)

    return np.clip(output, 0, 255).astype(np.uint8)


def create_gaussian_kernel(kernel_size=5, sigma=1.0):
    """
    Generate a normalized 2D Gaussian kernel from scratch.
    """

    radius = kernel_size // 2

    x = np.arange(
        -radius,
        radius + 1,
        dtype=np.float32
    )

    y = x[:, None]

    kernel = np.exp(
        -(x**2 + y**2) / (2 * sigma**2)
    )

    kernel /= kernel.sum()

    return kernel


def gaussian_filter(image, kernel_size=5, sigma=1.0):
    """
    Gaussian filter implemented from scratch using NumPy.
    """

    kernel = create_gaussian_kernel(
        kernel_size,
        sigma
    )

    pad = kernel_size // 2

    padded = np.pad(
        image.astype(np.float32),
        pad,
        mode="reflect"
    )

    output = np.zeros_like(
        image,
        dtype=np.float32
    )

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            window = padded[
                i:i + kernel_size,
                j:j + kernel_size
            ]

            output[i, j] = np.sum(
                window * kernel
            )

    return np.clip(
        output,
        0,
        255
    ).astype(np.uint8)


def median_filter(image, kernel_size=5):
    """
    Median filter implemented from scratch using NumPy.
    """

    pad = kernel_size // 2

    padded = np.pad(
        image,
        pad,
        mode="reflect"
    )

    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            window = padded[
                i:i + kernel_size,
                j:j + kernel_size
            ]

            output[i, j] = np.median(window)

    return output.astype(np.uint8)


# ============================================================
# OPENCV REFERENCE FILTERS
# ============================================================

def opencv_mean_filter(image, kernel_size=5):
    return cv2.blur(
        image,
        (kernel_size, kernel_size)
    )


def opencv_gaussian_filter(
    image,
    kernel_size=5,
    sigma=1.0
):
    return cv2.GaussianBlur(
        image,
        (kernel_size, kernel_size),
        sigmaX=sigma
    )


def opencv_median_filter(image, kernel_size=5):
    return cv2.medianBlur(
        image,
        kernel_size
    )


def bilateral_filter(
    image,
    diameter=9,
    sigma_color=75,
    sigma_space=75
):
    """
    Bilateral filter using OpenCV.
    """

    return cv2.bilateralFilter(
        image,
        diameter,
        sigma_color,
        sigma_space
    )


# ============================================================
# RUNTIME MEASUREMENT
# ============================================================

def benchmark_filter(filter_function, image):
    """
    Measure execution time of a filter.
    """

    start = time.perf_counter()

    result = filter_function(image)

    elapsed = (
        time.perf_counter() - start
    ) * 1000

    return result, elapsed


# ============================================================
# QUICK TEST
# ============================================================

def main():

    base_dir = Path(__file__).resolve().parent.parent

    test_image_path = (
        base_dir
        / "data"
        / "noisy"
        / "smooth_moon_gaussian_sigma_10.png"
    )

    image = cv2.imread(
        str(test_image_path),
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not read {test_image_path}"
        )

    print("Testing filters...")
    print("=" * 50)

    filters = {
        "Mean": lambda img: mean_filter(img, 5),

        "Gaussian": lambda img:
            gaussian_filter(img, 5, 1.0),

        "Median": lambda img:
            median_filter(img, 5),

        "Bilateral": lambda img:
            bilateral_filter(img, 9, 75, 75),
    }

    for name, function in filters.items():

        result, runtime = benchmark_filter(
            function,
            image
        )

        print(
            f"{name:<12} "
            f"{runtime:>10.3f} ms "
            f"| output shape: {result.shape}"
        )

    print("=" * 50)
    print("Filter test completed successfully.")


if __name__ == "__main__":
    main()
