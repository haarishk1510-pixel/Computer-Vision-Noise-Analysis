import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ============================================================
# CONFIGURATION
# ============================================================

RESULTS_DIR = "outputs/metrics"
OUTPUT_DIR = "outputs/visualizations"

os.makedirs(OUTPUT_DIR, exist_ok=True)

CSV_FILE = os.path.join(RESULTS_DIR, "filter_results.csv")

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("PROFESSIONAL RESULTS VISUALIZATION")
print("=" * 70)

df = pd.read_csv(CSV_FILE)

print("\nLoaded results:")
print(f"Rows: {len(df)}")
print(f"Columns: {list(df.columns)}")

# Normalize column names
df.columns = [c.strip() for c in df.columns]

# ============================================================
# COLUMN DETECTION
# ============================================================

def find_column(possible_names):
    for name in possible_names:
        for col in df.columns:
            if col.lower() == name.lower():
                return col
    return None


filter_col = find_column(["Filter", "filter"])
noise_col = find_column(["Noise", "noise"])
mae_col = find_column(["MAE", "mae"])
mse_col = find_column(["MSE", "mse"])
psnr_col = find_column(["PSNR", "psnr"])
ssim_col = find_column(["SSIM", "ssim"])
edge_col = find_column([
    "Edge_Preservation",
    "Edge Preservation",
    "edge_preservation"
])
runtime_col = find_column([
    "Runtime_ms",
    "Runtime",
    "runtime_ms"
])

required = {
    "Filter": filter_col,
    "Noise": noise_col,
    "MAE": mae_col,
    "PSNR": psnr_col,
    "SSIM": ssim_col,
    "Edge Preservation": edge_col,
    "Runtime": runtime_col
}

print("\nDetected columns:")
for name, col in required.items():
    print(f"{name:22} -> {col}")

missing = [name for name, col in required.items() if col is None]

if missing:
    print("\nERROR: Missing columns:")
    for item in missing:
        print(" -", item)
    raise SystemExit(1)

# ============================================================
# STYLE
# ============================================================

plt.rcParams.update({
    "figure.figsize": (10, 6),
    "axes.grid": True,
    "grid.alpha": 0.25,
    "font.size": 11
})

# ============================================================
# 1. PSNR COMPARISON
# ============================================================

summary = df.groupby(filter_col)[psnr_col].mean().sort_values(
    ascending=False
)

plt.figure()

summary.plot(kind="bar")

plt.title("Average PSNR by Filtering Technique")
plt.xlabel("Filter")
plt.ylabel("PSNR (dB)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "01_psnr_comparison.png"),
    dpi=300
)

plt.close()

# ============================================================
# 2. SSIM COMPARISON
# ============================================================

summary = df.groupby(filter_col)[ssim_col].mean().sort_values(
    ascending=False
)

plt.figure()

summary.plot(kind="bar")

plt.title("Average SSIM by Filtering Technique")
plt.xlabel("Filter")
plt.ylabel("SSIM")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "02_ssim_comparison.png"),
    dpi=300
)

plt.close()

# ============================================================
# 3. MAE COMPARISON
# ============================================================

summary = df.groupby(filter_col)[mae_col].mean().sort_values()

plt.figure()

summary.plot(kind="bar")

plt.title("Average MAE by Filtering Technique")
plt.xlabel("Filter")
plt.ylabel("Mean Absolute Error")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "03_mae_comparison.png"),
    dpi=300
)

plt.close()

# ============================================================
# 4. EDGE PRESERVATION
# ============================================================

summary = df.groupby(filter_col)[edge_col].mean().sort_values(
    ascending=False
)

plt.figure()

summary.plot(kind="bar")

plt.title("Average Edge Preservation by Filter")
plt.xlabel("Filter")
plt.ylabel("Edge Preservation")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "04_edge_preservation.png"),
    dpi=300
)

plt.close()

# ============================================================
# 5. RUNTIME COMPARISON
# ============================================================

summary = df.groupby(filter_col)[runtime_col].mean().sort_values()

plt.figure()

summary.plot(kind="bar")

plt.title("Average Filtering Runtime")
plt.xlabel("Filter")
plt.ylabel("Runtime (ms)")
plt.xticks(rotation=0)
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "05_runtime_comparison.png"),
    dpi=300
)

plt.close()

# ============================================================
# 6. NOISE TYPE COMPARISON
# ============================================================

noise_summary = df.groupby(noise_col)[psnr_col].mean().sort_values(
    ascending=False
)

plt.figure(figsize=(11, 6))

noise_summary.plot(kind="bar")

plt.title("Average PSNR Across Noise Types")
plt.xlabel("Noise Type")
plt.ylabel("PSNR (dB)")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()

plt.savefig(
    os.path.join(OUTPUT_DIR, "06_noise_comparison.png"),
    dpi=300
)

plt.close()

# ============================================================
# 7. COMPOSITE SCORE
# ============================================================

composite_file = os.path.join(
    RESULTS_DIR,
    "filter_composite_scores.csv"
)

if os.path.exists(composite_file):

    composite_df = pd.read_csv(composite_file)

    composite_filter = find_column(["Filter", "filter"])

    composite_score = find_column([
        "Composite_Score",
        "Composite Score",
        "composite_score"
    ])

    if composite_filter and composite_score:

        summary = composite_df.groupby(
            composite_filter
        )[composite_score].mean().sort_values(
            ascending=False
        )

        plt.figure()

        summary.plot(kind="bar")

        plt.title("Average Composite Score by Filter")
        plt.xlabel("Filter")
        plt.ylabel("Composite Score")
        plt.xticks(rotation=0)
        plt.tight_layout()

        plt.savefig(
            os.path.join(
                OUTPUT_DIR,
                "07_composite_score.png"
            ),
            dpi=300
        )

        plt.close()

# ============================================================
# 8. METRIC HEATMAP
# ============================================================

heatmap_data = df.groupby(filter_col)[
    [mae_col, psnr_col, ssim_col, edge_col, runtime_col]
].mean()

# Normalize each metric between 0 and 1
normalized = heatmap_data.copy()

for col in normalized.columns:

    minimum = normalized[col].min()
    maximum = normalized[col].max()

    if maximum != minimum:
        normalized[col] = (
            normalized[col] - minimum
        ) / (maximum - minimum)

# MAE and Runtime are lower-is-better
normalized[mae_col] = 1 - normalized[mae_col]
normalized[runtime_col] = 1 - normalized[runtime_col]

plt.figure(figsize=(10, 6))

plt.imshow(
    normalized.values,
    aspect="auto"
)

plt.colorbar(label="Normalized Performance")

plt.xticks(
    range(len(normalized.columns)),
    normalized.columns,
    rotation=30,
    ha="right"
)

plt.yticks(
    range(len(normalized.index)),
    normalized.index
)

plt.title("Filter Performance Heatmap")

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "08_filter_heatmap.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# 9. FINAL COMPARISON
# ============================================================

overall = df.groupby(filter_col).agg({
    mae_col: "mean",
    psnr_col: "mean",
    ssim_col: "mean",
    edge_col: "mean",
    runtime_col: "mean"
})

# Normalize
score = pd.DataFrame(index=overall.index)

for col in [psnr_col, ssim_col, edge_col]:

    minimum = overall[col].min()
    maximum = overall[col].max()

    if maximum != minimum:
        score[col] = (
            overall[col] - minimum
        ) / (maximum - minimum)
    else:
        score[col] = 1

for col in [mae_col, runtime_col]:

    minimum = overall[col].min()
    maximum = overall[col].max()

    if maximum != minimum:
        score[col] = 1 - (
            (overall[col] - minimum)
            / (maximum - minimum)
        )
    else:
        score[col] = 1

score["Overall Score"] = score.mean(axis=1)

score = score.sort_values(
    "Overall Score",
    ascending=False
)

plt.figure(figsize=(10, 6))

score["Overall Score"].plot(kind="bar")

plt.title("Overall Filter Performance")
plt.xlabel("Filter")
plt.ylabel("Normalized Overall Score")
plt.xticks(rotation=0)

plt.tight_layout()

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "09_final_comparison.png"
    ),
    dpi=300
)

plt.close()

# ============================================================
# SAVE FINAL RANKING
# ============================================================

ranking_file = os.path.join(
    OUTPUT_DIR,
    "final_filter_ranking.csv"
)

score.to_csv(ranking_file)

# ============================================================
# FINAL REPORT
# ============================================================

best_filter = score["Overall Score"].idxmax()
best_score = score["Overall Score"].max()

print("\n" + "=" * 70)
print("VISUALIZATION COMPLETED SUCCESSFULLY")
print("=" * 70)

print(f"\nBest overall filter: {best_filter}")
print(f"Overall normalized score: {best_score:.4f}")

print("\nGenerated visualizations:")

for filename in sorted(os.listdir(OUTPUT_DIR)):
    if filename.endswith(".png"):
        print(" ->", filename)

print("\nFinal ranking saved to:")
print(ranking_file)

print("\nAll visualization files saved in:")
print(OUTPUT_DIR)

print("=" * 70)
