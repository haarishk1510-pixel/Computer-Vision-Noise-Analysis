from pathlib import Path

import cv2
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# Configuration
# ============================================================

RANDOM_SEED = 42


# ============================================================
# Noise Generation Functions
# ============================================================

def add_gaussian_noise(image, sigma, rng):
    """
    Add additive Gaussian noise.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image with values in [0, 255].
    sigma : float
        Standard deviation of Gaussian noise.
    rng : np.random.Generator
        Reproducible random-number generator.

    Returns
    -------
    np.ndarray
        Noisy uint8 image.
    """

    image_float = image.astype(np.float32)

    noise = rng.normal(
        loc=0.0,
        scale=sigma,
        size=image.shape
    )

    noisy = image_float + noise

    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(image, amount, rng):
    """
    Add salt-and-pepper impulse noise.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    amount : float
        Fraction of pixels affected.
    rng : np.random.Generator

    Returns
    -------
    np.ndarray
        Noisy uint8 image.
    """

    noisy = image.copy()

    total_pixels = image.size
    number_of_noisy_pixels = int(amount * total_pixels)

    indices = rng.choice(
        total_pixels,
        size=number_of_noisy_pixels,
        replace=False
    )

    half = number_of_noisy_pixels // 2

    flat = noisy.reshape(-1)

    flat[indices[:half]] = 0
    flat[indices[half:]] = 255

    return noisy


def add_speckle_noise(image, rng, variance=0.04):
    """
    Add multiplicative speckle noise.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.
    rng : np.random.Generator
        Reproducible random-number generator.
    variance : float
        Speckle noise variance.

    Returns
    -------
    np.ndarray
        Noisy uint8 image.
    """

    image_float = image.astype(np.float32) / 255.0

    noise = rng.normal(
        loc=0.0,
        scale=np.sqrt(variance),
        size=image.shape
    )

    noisy = image_float + image_float * noise

    noisy = np.clip(noisy, 0, 1)

    return (noisy * 255).astype(np.uint8)


def add_poisson_noise(image):
    """
    Add Poisson (shot/counting) noise.

    Parameters
    ----------
    image : np.ndarray
        Grayscale image.

    Returns
    -------
    np.ndarray
        Noisy uint8 image.
    """

    image_float = image.astype(np.float32) / 255.0

    # Convert normalized image into a photon-count-like scale.
    values = 2 ** 8

    noisy = np.random.poisson(
        image_float * values
    ) / values

    noisy = np.clip(noisy, 0, 1)

    return (noisy * 255).astype(np.uint8)


# ============================================================
# Histogram Generation
# ============================================================

def save_histogram(image, title, output_path):
    """
    Save grayscale histogram for an image.
    """

    plt.figure(figsize=(8, 5))

    plt.hist(
        image.ravel(),
        bins=256,
        range=(0, 256)
    )

    plt.title(title)
    plt.xlabel("Pixel Intensity")
    plt.ylabel("Frequency")
    plt.xlim(0, 255)
    plt.tight_layout()

    plt.savefig(
        output_path,
        dpi=200,
        bbox_inches="tight"
    )

    plt.close()


# ============================================================
# Main Experiment
# ============================================================

def main():

    base_dir = Path(__file__).resolve().parent.parent

    input_dir = base_dir / "data" / "original"
    output_dir = base_dir / "data" / "noisy"
    histogram_dir = base_dir / "outputs" / "histograms"

    output_dir.mkdir(parents=True, exist_ok=True)
    histogram_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    image_files = sorted(input_dir.glob("*.png"))

    if not image_files:
        raise FileNotFoundError(
            "No PNG images found in data/original/"
        )

    for image_path in image_files:

        print(f"\nProcessing: {image_path.name}")

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_GRAYSCALE
        )

        if image is None:
            print(f"WARNING: Could not read {image_path}")
            continue

        image_name = image_path.stem

        # ----------------------------------------------------
        # Gaussian Noise
        # ----------------------------------------------------

        for sigma in [10, 25]:

            noisy = add_gaussian_noise(
                image,
                sigma,
                rng
            )

            filename = (
                f"{image_name}_gaussian_sigma_{sigma}.png"
            )

            output_path = output_dir / filename

            cv2.imwrite(
                str(output_path),
                noisy
            )

            save_histogram(
                noisy,
                f"{image_name} - Gaussian Noise σ={sigma}",
                histogram_dir / f"{filename}.png"
            )

        # ----------------------------------------------------
        # Salt & Pepper Noise
        # ----------------------------------------------------

        for amount in [0.05, 0.20]:

            noisy = add_salt_pepper_noise(
                image,
                amount,
                rng
            )

            percentage = int(amount * 100)

            filename = (
                f"{image_name}_salt_pepper_{percentage}percent.png"
            )

            output_path = output_dir / filename

            cv2.imwrite(
                str(output_path),
                noisy
            )

            save_histogram(
                noisy,
                f"{image_name} - Salt & Pepper {percentage}%",
                histogram_dir / f"{filename}.png"
            )

        # ----------------------------------------------------
        # Speckle Noise
        # ----------------------------------------------------

        noisy = add_speckle_noise(
            image,
            rng
        )

        filename = f"{image_name}_speckle.png"

        cv2.imwrite(
            str(output_dir / filename),
            noisy
        )

        save_histogram(
            noisy,
            f"{image_name} - Speckle Noise",
            histogram_dir / f"{filename}.png"
        )

        # ----------------------------------------------------
        # Poisson Noise
        # ----------------------------------------------------

        noisy = add_poisson_noise(image)

        filename = f"{image_name}_poisson.png"

        cv2.imwrite(
            str(output_dir / filename),
            noisy
        )

        save_histogram(
            noisy,
            f"{image_name} - Poisson Noise",
            histogram_dir / f"{filename}.png"
        )

    print("\n========================================")
    print("Noise generation completed successfully.")
    print("========================================")


if __name__ == "__main__":
    main()
