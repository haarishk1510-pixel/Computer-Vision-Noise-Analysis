# Computer Vision Noise Analysis and Image Filter Evaluation

## Project Overview

This project presents a complete Computer Vision pipeline for analyzing image noise and evaluating different image denoising filters.

The system generates controlled noise, applies multiple spatial filtering techniques, evaluates the results using quantitative image-quality metrics, analyzes edge preservation and runtime performance, and performs object detection with region and shape descriptor analysis.

## Objectives

- Analyze different types of image noise.
- Apply Mean, Gaussian, Median, and Bilateral filters.
- Compare filters using PSNR, SSIM, and MAE.
- Evaluate edge preservation.
- Compare computational runtime.
- Determine the best-performing filter using composite scoring.
- Perform object detection and measurement.
- Extract region and shape descriptors.
- Generate visual and numerical analysis results.
- Verify scratch implementations against OpenCV operations.

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
      +-----------------------+
      |                       |
      v                       v
Image Quality Metrics    Edge Preservation
      |                       |
      +-----------+-----------+
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
## Technologies Used

- Python 3
- OpenCV
- NumPy
- Pandas
- SciPy
- scikit-image
- Matplotlib
- Jupyter Notebook

## Filters Evaluated

- Mean Filter
- Gaussian Filter
- Median Filter
- Bilateral Filter

## Noise Models

- Gaussian Noise
- Poisson Noise
- Salt-and-Pepper Noise
- Speckle Noise

## Evaluation Metrics

- PSNR
- SSIM
- MAE
- Edge Preservation
- Runtime

## Project Structure

```text
Computer-Vision-Noise-Analysis/
├── data/
│   ├── original/
│   ├── noisy/
│   ├── filtered/
│   └── objects/
├── notebooks/
├── outputs/
│   ├── metrics/
│   ├── visualizations/
│   ├── histograms/
│   ├── objects/
│   ├── shape_analysis/
│   ├── verification/
│   └── final_analysis/
├── report/
│   └── final_report.md
├── src/
├── README.md
├── requirements.txt
└── .gitignore


How to run:
git clone https://github.com/haarishk1510-pixel/Computer-Vision-Noise-Analysis.git
cd Computer-Vision-Noise-Analysis

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
