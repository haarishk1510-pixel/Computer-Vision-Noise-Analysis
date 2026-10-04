from pathlib import Path
import pandas as pd
import numpy as np


BASE_DIR = Path(__file__).resolve().parent.parent

CSV_PATH = (
    BASE_DIR
    / "outputs"
    / "metrics"
    / "filter_results.csv"
)

OUTPUT_DIR = (
    BASE_DIR
    / "outputs"
    / "metrics"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# LOAD RESULTS
# ============================================================

df = pd.read_csv(CSV_PATH)

print()
print("=" * 75)
print("FILTER PERFORMANCE ANALYSIS")
print("=" * 75)

print(f"Total experiment records: {len(df)}")
print(
    f"Images: {df['Image'].nunique()}"
)
print(
    f"Noise conditions: {df['Noise'].nunique()}"
)
print(
    f"Filters: {df['Filter'].nunique()}"
)


# ============================================================
# AVERAGE PERFORMANCE BY NOISE + FILTER
# ============================================================

summary = (
    df
    .groupby(["Noise", "Filter"])
    [
        [
            "MAE",
            "MSE",
            "PSNR",
            "SSIM",
            "Edge_Preservation",
            "Runtime_ms"
        ]
    ]
    .mean()
    .reset_index()
)


summary_path = (
    OUTPUT_DIR
    / "filter_summary.csv"
)

summary.to_csv(
    summary_path,
    index=False
)


# ============================================================
# PRINT SUMMARY
# ============================================================

print()
print("=" * 75)
print("AVERAGE PERFORMANCE BY NOISE TYPE")
print("=" * 75)

for noise in summary["Noise"].unique():

    print()
    print(f"NOISE: {noise}")
    print("-" * 75)

    subset = summary[
        summary["Noise"] == noise
    ].copy()

    subset = subset.sort_values(
        "PSNR",
        ascending=False
    )

    print(
        subset[
            [
                "Filter",
                "MAE",
                "PSNR",
                "SSIM",
                "Edge_Preservation",
                "Runtime_ms"
            ]
        ].to_string(
            index=False,
            float_format=lambda x:
                f"{x:.4f}"
        )
    )


# ============================================================
# BEST FILTERS
# ============================================================

best_psnr = (
    summary
    .loc[
        summary.groupby("Noise")["PSNR"]
        .idxmax()
    ]
    [
        [
            "Noise",
            "Filter",
            "PSNR",
            "SSIM",
            "MAE",
            "Edge_Preservation",
            "Runtime_ms"
        ]
    ]
    .sort_values("Noise")
)


best_ssim = (
    summary
    .loc[
        summary.groupby("Noise")["SSIM"]
        .idxmax()
    ]
    [
        [
            "Noise",
            "Filter",
            "SSIM"
        ]
    ]
)


best_mae = (
    summary
    .loc[
        summary.groupby("Noise")["MAE"]
        .idxmin()
    ]
    [
        [
            "Noise",
            "Filter",
            "MAE"
        ]
    ]
)


# ============================================================
# SAVE BEST-FILTER TABLE
# ============================================================

best_psnr_path = (
    OUTPUT_DIR
    / "best_filter_by_psnr.csv"
)

best_psnr.to_csv(
    best_psnr_path,
    index=False
)


best_ssim.to_csv(
    OUTPUT_DIR / "best_filter_by_ssim.csv",
    index=False
)


best_mae.to_csv(
    OUTPUT_DIR / "best_filter_by_mae.csv",
    index=False
)


# ============================================================
# OVERALL FILTER PERFORMANCE
# ============================================================

overall = (
    df
    .groupby("Filter")
    [
        [
            "MAE",
            "MSE",
            "PSNR",
            "SSIM",
            "Edge_Preservation",
            "Runtime_ms"
        ]
    ]
    .mean()
    .reset_index()
)


print()
print("=" * 75)
print("OVERALL FILTER PERFORMANCE")
print("=" * 75)

print(
    overall.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)


overall.to_csv(
    OUTPUT_DIR / "overall_filter_performance.csv",
    index=False
)


# ============================================================
# SIMPLE COMPOSITE SCORE
# ============================================================

# Normalize metrics so higher = better.

score_df = summary.copy()


def normalize_high(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return np.ones(
            len(series)
        )

    return (
        (series - minimum) /
        (maximum - minimum)
    )


def normalize_low(series):

    minimum = series.min()
    maximum = series.max()

    if maximum == minimum:
        return np.ones(
            len(series)
        )

    return (
        (maximum - series) /
        (maximum - minimum)
    )


score_df["PSNR_score"] = (
    normalize_high(
        score_df["PSNR"]
    )
)

score_df["SSIM_score"] = (
    normalize_high(
        score_df["SSIM"]
    )
)

score_df["Edge_score"] = (
    normalize_high(
        score_df["Edge_Preservation"]
    )
)

score_df["MAE_score"] = (
    normalize_low(
        score_df["MAE"]
    )
)

score_df["Runtime_score"] = (
    normalize_low(
        score_df["Runtime_ms"]
    )
)


# Equal-weight composite score

score_df["Composite_Score"] = (
    score_df["PSNR_score"]
    + score_df["SSIM_score"]
    + score_df["Edge_score"]
    + score_df["MAE_score"]
    + score_df["Runtime_score"]
) / 5


score_path = (
    OUTPUT_DIR
    / "filter_composite_scores.csv"
)

score_df.to_csv(
    score_path,
    index=False
)


# ============================================================
# BEST FILTER USING COMPOSITE SCORE
# ============================================================

best_overall = (
    score_df
    .loc[
        score_df.groupby("Noise")
        ["Composite_Score"]
        .idxmax()
    ]
    [
        [
            "Noise",
            "Filter",
            "Composite_Score",
            "PSNR",
            "SSIM",
            "MAE",
            "Edge_Preservation",
            "Runtime_ms"
        ]
    ]
    .sort_values("Noise")
)


best_overall_path = (
    OUTPUT_DIR
    / "best_filter_overall.csv"
)

best_overall.to_csv(
    best_overall_path,
    index=False
)


print()
print("=" * 75)
print("BEST FILTER BY COMPOSITE SCORE")
print("=" * 75)

print(
    best_overall.to_string(
        index=False,
        float_format=lambda x:
            f"{x:.4f}"
    )
)


print()
print("=" * 75)
print("ANALYSIS COMPLETED")
print("=" * 75)

print(
    f"Summary saved to: {summary_path}"
)

print(
    f"Best PSNR saved to: {best_psnr_path}"
)

print(
    "Overall performance saved."
)

print(
    "Composite ranking saved."
)

print("=" * 75)
