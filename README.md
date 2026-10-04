# Computer Vision Noise Analysis and Image Filter Evaluation

A complete Computer Vision project for analyzing image noise, evaluating spatial denoising filters, measuring image-quality performance, and performing object and shape analysis.

---

## Project Overview

This project implements an end-to-end Computer Vision pipeline for studying the effect of different noise models on digital images and evaluating the performance of multiple image denoising techniques.

The system:

- Generates controlled noise on multiple input images.
- Applies different spatial filtering techniques.
- Evaluates filtered images using quantitative image-quality metrics.
- Measures edge preservation and computational runtime.
- Performs object detection and object measurement.
- Extracts region and shape descriptors.
- Verifies custom filter implementations against OpenCV operations.
- Produces visualizations, numerical results, and a final filter ranking.

The main objective is to determine which filtering technique provides the best balance between noise reduction, image quality, structural preservation, and computational efficiency.

---

## Objectives

- Analyze different types of image noise.
- Apply Mean, Gaussian, Median, and Bilateral filters.
- Compare filter performance using PSNR, SSIM, and MAE.
- Evaluate edge preservation after filtering.
- Compare the computational runtime of different filters.
- Determine the best-performing filter using composite scoring.
- Perform object detection and measurement.
- Extract region and shape descriptors from detected objects.
- Generate visual and numerical analysis results.
- Verify scratch implementations against equivalent OpenCV operations.
- Produce a reproducible Computer Vision analysis pipeline.

---

## Project Pipeline

```text
                         Original Images
                               |
                               v
                        Noise Generation
                               |
                               v
                         Noisy Images
                               |
                               v
                  Multiple Filtering Techniques
                               |
                 +-------------+-------------+
                 |                           |
                 v                           v
          Image Quality Metrics       Edge Preservation
                 |                           |
                 +-------------+-------------+
                               |
                               v
                       Runtime Analysis
                               |
                               v
                       Composite Scoring
                               |
                               v
                      Final Filter Ranking
                               |
                               v
                       Object Detection
                               |
                               v
                    Region / Shape Analysis
                               |
                               v
                       Final Project Results
