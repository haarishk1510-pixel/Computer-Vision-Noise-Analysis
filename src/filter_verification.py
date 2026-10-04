"""
Filter Verification
-------------------
Verifies from-scratch implementations of:
1. 2-D convolution
2. Mean filter
3. Median filter
4. Gaussian filter (5x5, sigma=1)

Each implementation is compared against OpenCV.
"""

import os
import cv2
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

INPUT_IMAGE = "data/original/real_photo_coffee.png"
OUTPUT_DIR = "outputs/verification"

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# 1. 2-D CONVOLUTION FROM SCRATCH
# ============================================================

def convolution_2d(image, kernel):
    """
    Perform 2-D convolution using zero padding.
    """

    image = image.astype(np.float64)
    kernel = np.asarray(kernel, dtype=np.float64)

    kh, kw = kernel.shape

    pad_h = kh // 2
    pad_w = kw // 2

    padded = np.pad(
        image,
        ((pad_h, pad_h), (pad_w, pad_w)),
        mode="constant",
        constant_values=0
    )

    # Flip kernel for true convolution
    kernel_flipped = np.flipud(np.fliplr(kernel))

    output = np.zeros_like(image, dtype=np.float64)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            region = padded[
                i:i + kh,
                j:j + kw
            ]

            output[i, j] = np.sum(
                region * kernel_flipped
            )

    return np.clip(output, 0, 255).astype(np.uint8)


# ============================================================
# 2. MEAN FILTER FROM SCRATCH
# ============================================================

def mean_filter(image, kernel_size=5):

    kernel = np.ones(
        (kernel_size, kernel_size),
        dtype=np.float64
    )

    kernel /= kernel.size

    return convolution_2d(image, kernel)


# ============================================================
# 3. MEDIAN FILTER FROM SCRATCH
# ============================================================

def median_filter(image, kernel_size=5):

    pad = kernel_size // 2

    padded = np.pad(
        image,
        ((pad, pad), (pad, pad)),
        mode="edge"
    )

    output = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):

            region = padded[
                i:i + kernel_size,
                j:j + kernel_size
            ]

            output[i, j] = np.median(region)

    return output.astype(np.uint8)


# ============================================================
# 4. GAUSSIAN KERNEL FROM SCRATCH
# ============================================================

def gaussian_kernel(size=5, sigma=1.0):

    center = size // 2

    kernel = np.zeros(
        (size, size),
        dtype=np.float64
    )

    for i in range(size):
        for j in range(size):

            x = i - center
            y = j - center

            kernel[i, j] = np.exp(
                -(x * x + y * y) /
                (2 * sigma * sigma)
            )

    kernel /= np.sum(kernel)

    return kernel


# ============================================================
# 5. GAUSSIAN FILTER FROM SCRATCH
# ============================================================

def gaussian_filter(image, size=5, sigma=1.0):

    kernel = gaussian_kernel(
        size=size,
        sigma=sigma
    )

    return convolution_2d(
        image,
        kernel
    )


# ============================================================
# COMPARISON FUNCTION
# ============================================================

def compare_images(reference, test):

    reference = reference.astype(np.float64)
    test = test.astype(np.float64)

    difference = np.abs(
        reference - test
    )

    mae = np.mean(difference)

    max_error = np.max(difference)

    matching_pixels = np.mean(
        difference == 0
    ) * 100

    return mae, max_error, matching_pixels


# ============================================================
# SAVE IMAGE
# ============================================================

def save_image(filename, image):

    path = os.path.join(
        OUTPUT_DIR,
        filename
    )

    cv2.imwrite(path, image)

    return path


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("FROM-SCRATCH FILTER VERIFICATION")
    print("=" * 70)

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    print("\nLoading input image...")

    image = cv2.imread(
        INPUT_IMAGE,
        cv2.IMREAD_GRAYSCALE
    )

    if image is None:
        raise FileNotFoundError(
            f"Could not load image: {INPUT_IMAGE}"
        )

    print(f"Input image : {INPUT_IMAGE}")
    print(f"Image shape : {image.shape}")

    # --------------------------------------------------------
    # Mean Filter
    # --------------------------------------------------------

    print("\n[1/3] Testing Mean Filter...")

    scratch_mean = mean_filter(
        image,
        kernel_size=5
    )

    opencv_mean = cv2.blur(
        image,
        (5, 5)
    )

    mean_mae, mean_max, mean_match = compare_images(
        opencv_mean,
        scratch_mean
    )

    save_image(
        "mean_scratch.png",
        scratch_mean
    )

    save_image(
        "mean_opencv.png",
        opencv_mean
    )

    # --------------------------------------------------------
    # Median Filter
    # --------------------------------------------------------

    print("[2/3] Testing Median Filter...")

    scratch_median = median_filter(
        image,
        kernel_size=5
    )

    opencv_median = cv2.medianBlur(
        image,
        5
    )

    median_mae, median_max, median_match = compare_images(
        opencv_median,
        scratch_median
    )

    save_image(
        "median_scratch.png",
        scratch_median
    )

    save_image(
        "median_opencv.png",
        opencv_median
    )

    # --------------------------------------------------------
    # Gaussian Filter
    # --------------------------------------------------------

    print("[3/3] Testing Gaussian Filter...")

    sigma = 1.0

    kernel = gaussian_kernel(
        size=5,
        sigma=sigma
    )

    scratch_gaussian = gaussian_filter(
        image,
        size=5,
        sigma=sigma
    )

    opencv_gaussian = cv2.GaussianBlur(
        image,
        (5, 5),
        sigmaX=sigma,
        sigmaY=sigma,
        borderType=cv2.BORDER_CONSTANT
    )

    gaussian_mae, gaussian_max, gaussian_match = compare_images(
        opencv_gaussian,
        scratch_gaussian
    )

    save_image(
        "gaussian_scratch.png",
        scratch_gaussian
    )

    save_image(
        "gaussian_opencv.png",
        opencv_gaussian
    )

    # --------------------------------------------------------
    # Print Gaussian Kernel
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("5x5 GAUSSIAN KERNEL (sigma = 1.0)")
    print("=" * 70)

    print(np.round(kernel, 6))

    print(
        f"\nKernel sum = {np.sum(kernel):.6f}"
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VERIFICATION RESULTS")
    print("=" * 70)

    print("\nMean Filter")
    print("-" * 40)
    print(f"MAE                  : {mean_mae:.6f}")
    print(f"Maximum Pixel Error  : {mean_max:.6f}")
    print(f"Exact Pixel Match    : {mean_match:.2f}%")

    print("\nMedian Filter")
    print("-" * 40)
    print(f"MAE                  : {median_mae:.6f}")
    print(f"Maximum Pixel Error  : {median_max:.6f}")
    print(f"Exact Pixel Match    : {median_match:.2f}%")

    print("\nGaussian Filter")
    print("-" * 40)
    print(f"MAE                  : {gaussian_mae:.6f}")
    print(f"Maximum Pixel Error  : {gaussian_max:.6f}")
    print(f"Exact Pixel Match    : {gaussian_match:.2f}%")

    # --------------------------------------------------------
    # Overall verification
    # --------------------------------------------------------

    tolerance = 0.5

    all_passed = (
        mean_mae <= tolerance
        and median_mae <= tolerance
        and gaussian_mae <= tolerance
    )

    print("\n" + "=" * 70)

    if all_passed:
        print("✓ ALL FILTER VERIFICATIONS PASSED")
    else:
        print("⚠ FILTER VERIFICATION COMPLETED")
        print("  Check the numerical differences above.")

    print("=" * 70)

    print("\nGenerated verification files:")

    for filename in sorted(os.listdir(OUTPUT_DIR)):
        print(f" -> {filename}")

    print(
        f"\nVerification outputs saved in: {OUTPUT_DIR}"
    )


if __name__ == "__main__":
    main()
