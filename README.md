# Computer Vision Noise Analysis and Image Filter Evaluation

## Project Overview

A complete Computer Vision project for analyzing image noise, evaluating spatial denoising filters, measuring image-quality performance, and performing object and shape analysis.

The system generates controlled noise on multiple images, applies different filtering techniques, evaluates the results using quantitative metrics, analyzes edge preservation and runtime performance, and produces a final filter ranking.

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
- Produce a reproducible Computer Vision analysis pipeline.

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
```

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
- Composite Score

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
```

## How to Run

```bash
git clone https://github.com/haarishk1510-pixel/Computer-Vision-Noise-Analysis.git
cd Computer-Vision-Noise-Analysis

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

## Run the Experiment

```bash
python src/run_experiment.py
```

## Generate Analysis

```bash
python src/analyze_results.py
python src/visualize_results.py
python src/final_analysis.py
```

## Object and Shape Analysis

```bash
python src/object_pipeline.py
python src/region_shape_analysis.py
```

## Verification

```bash
python src/filter_verification.py
```

## Results

The project generates:

- Filter performance metrics
- PSNR, SSIM, and MAE comparisons
- Edge-preservation analysis
- Runtime comparisons
- Noise comparison visualizations
- Filter ranking and composite scores
- Object detection results
- Object measurements
- Region and shape descriptors
- Scratch implementation verification
- Final consolidated analysis

All generated results are available in the `outputs/` directory.

## Conclusion

This project provides an end-to-end evaluation framework for studying image noise and denoising techniques. The combined quantitative metrics, structural analysis, runtime measurements, object processing, and final ranking provide a comprehensive comparison of filtering methods.
