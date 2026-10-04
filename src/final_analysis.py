import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt





# ============================================================

# CONFIGURATION

# ============================================================



BASE_DIR = os.getcwd()



METRICS_DIR = os.path.join(BASE_DIR, "outputs", "metrics")

OBJECTS_DIR = os.path.join(BASE_DIR, "outputs", "objects")

SHAPE_DIR = os.path.join(BASE_DIR, "outputs", "shape_analysis")

VERIFICATION_DIR = os.path.join(BASE_DIR, "outputs", "verification")

VIS_DIR = os.path.join(BASE_DIR, "outputs", "visualizations")

FINAL_DIR = os.path.join(BASE_DIR, "outputs", "final_analysis")



os.makedirs(FINAL_DIR, exist_ok=True)





# ============================================================

# HELPER FUNCTIONS

# ============================================================



def load_csv(path, name):

    """Load CSV safely and report status."""

    if not os.path.exists(path):

        print(f"WARNING: {name} not found:")

        print(f"        {path}")

        return pd.DataFrame()



    try:

        df = pd.read_csv(path)

        print(f"Loaded {name}: {len(df)} rows")

        return df

    except Exception as e:

        print(f"ERROR loading {name}: {e}")

        return pd.DataFrame()





def safe_float(value, default=0.0):

    """Safely convert value to float."""

    try:

        value = float(value)



        if np.isnan(value) or np.isinf(value):

            return default



        return value



    except (ValueError, TypeError):

        return default





def format_filter_name(name):

    """Make filter names presentation-friendly."""

    if pd.isna(name):

        return "Unknown"



    return str(name)





# ============================================================

# MAIN ANALYSIS

# ============================================================



print("=" * 70)

print("FINAL INTEGRATED COMPUTER VISION ANALYSIS")

print("=" * 70)





# ============================================================

# [1/7] LOAD FILTER ANALYSIS RESULTS

# ============================================================



print("\n[1/7] Loading filter analysis results...")



filter_results_path = os.path.join(

    METRICS_DIR,

    "filter_results.csv"

)



filter_summary_path = os.path.join(

    METRICS_DIR,

    "filter_summary.csv"

)



overall_results_path = os.path.join(

    METRICS_DIR,

    "overall_filter_performance.csv"

)



composite_scores_path = os.path.join(

    METRICS_DIR,

    "filter_composite_scores.csv"

)



best_overall_path = os.path.join(

    METRICS_DIR,

    "best_filter_overall.csv"

)



filter_results = load_csv(

    filter_results_path,

    "filter results"

)



filter_summary = load_csv(

    filter_summary_path,

    "filter summary"

)



overall_results = load_csv(

    overall_results_path,

    "overall results"

)



composite_scores = load_csv(

    composite_scores_path,

    "composite scores"

)



best_overall = load_csv(

    best_overall_path,

    "best filter overall"

)





# ============================================================
# [2/7] DETERMINE FINAL FILTER RANKING
# ============================================================

print("\n[2/7] Determining final filter ranking...")

# These two tables serve different purposes:
#   1. overall_results / filter_composite_scores -> overall ranking
#   2. best_filter_overall -> best filter for EACH noise condition
# Keeping them separate prevents the noise-specific table from being
# accidentally lost while generating the overall ranking.

ranking = pd.DataFrame()
best_by_noise = pd.DataFrame()
best_filter = "Unavailable"
best_score = 0.0

# ------------------------------------------------------------
# A. LOAD / BUILD BEST FILTER BY NOISE CONDITION
# ------------------------------------------------------------

if not best_overall.empty:
    best_by_noise = best_overall.copy()

    # Normalize column names
    best_by_noise.columns = [str(c).strip() for c in best_by_noise.columns]

    required_noise_columns = [
        "Noise",
        "Filter",
        "Composite_Score"
    ]

    if all(c in best_by_noise.columns for c in required_noise_columns):
        best_by_noise["Composite_Score"] = pd.to_numeric(
            best_by_noise["Composite_Score"],
            errors="coerce"
        )

        best_by_noise = best_by_noise.dropna(
            subset=["Noise", "Filter", "Composite_Score"]
        )

        # If the source contains multiple rows for the same noise,
        # retain only the highest composite-score filter.
        best_by_noise = (
            best_by_noise
            .sort_values(
                ["Noise", "Composite_Score"],
                ascending=[True, False]
            )
            .groupby("Noise", as_index=False)
            .first()
        )
    else:
        best_by_noise = pd.DataFrame()

# Fallback: use an already-generated final noise table if available.
if best_by_noise.empty:
    saved_noise_path = os.path.join(
        FINAL_DIR,
        "best_filter_by_noise.csv"
    )

    if os.path.exists(saved_noise_path):
        saved_noise = load_csv(
            saved_noise_path,
            "saved best filter by noise"
        )

        saved_noise.columns = [
            str(c).strip() for c in saved_noise.columns
        ]

        if all(
            c in saved_noise.columns
            for c in ["Noise", "Filter", "Composite_Score"]
        ):
            saved_noise["Composite_Score"] = pd.to_numeric(
                saved_noise["Composite_Score"],
                errors="coerce"
            )

            best_by_noise = saved_noise.dropna(
                subset=["Noise", "Filter", "Composite_Score"]
            ).copy()

# Final fallback: derive the best filter directly from the full
# composite-score table.
if best_by_noise.empty and not composite_scores.empty:
    candidate = composite_scores.copy()
    candidate.columns = [str(c).strip() for c in candidate.columns]

    if all(
        c in candidate.columns
        for c in ["Noise", "Filter", "Composite_Score"]
    ):
        candidate["Composite_Score"] = pd.to_numeric(
            candidate["Composite_Score"],
            errors="coerce"
        )

        candidate = candidate.dropna(
            subset=["Noise", "Filter", "Composite_Score"]
        )

        best_by_noise = (
            candidate
            .sort_values(
                ["Noise", "Composite_Score"],
                ascending=[True, False]
            )
            .groupby("Noise", as_index=False)
            .first()
        )

# Save the authoritative noise-specific table.
if not best_by_noise.empty:
    best_noise_columns = [
        c for c in [
            "Noise",
            "Filter",
            "Composite_Score",
            "PSNR",
            "SSIM",
            "MAE",
            "Edge_Preservation",
            "Runtime_ms"
        ]
        if c in best_by_noise.columns
    ]

    best_by_noise = best_by_noise[best_noise_columns]

    best_by_noise_path = os.path.join(
        FINAL_DIR,
        "best_filter_by_noise.csv"
    )

    best_by_noise.to_csv(
        best_by_noise_path,
        index=False
    )

# ------------------------------------------------------------
# B. DETERMINE OVERALL FILTER RANKING
# ------------------------------------------------------------

# Prefer the precomputed overall performance table because it represents
# all evaluation records, rather than only the winning filter for each noise.
ranking_source = pd.DataFrame()

if not overall_results.empty:
    ranking_source = overall_results.copy()
    ranking_source.columns = [str(c).strip() for c in ranking_source.columns]

# Fallback to the full composite-score table if necessary.
if ranking_source.empty or not all(
    c in ranking_source.columns
    for c in ["Filter", "Composite_Score"]
):
    if not composite_scores.empty:
        candidate = composite_scores.copy()
        candidate.columns = [str(c).strip() for c in candidate.columns]

        if all(
            c in candidate.columns
            for c in ["Filter", "Composite_Score"]
        ):
            candidate["Composite_Score"] = pd.to_numeric(
                candidate["Composite_Score"],
                errors="coerce"
            )
            candidate = candidate.dropna(
                subset=["Filter", "Composite_Score"]
            )

            # Aggregate all noise/filter evaluation records.
            agg = {
                "Average_Composite_Score": (
                    "Composite_Score", "mean"
                )
            }

            optional_mapping = {
                "PSNR": "Average_PSNR",
                "SSIM": "Average_SSIM",
                "MAE": "Average_MAE",
                "Edge_Preservation": "Average_Edge_Preservation",
                "Runtime_ms": "Average_Runtime_ms"
            }

            for source_col, output_col in optional_mapping.items():
                if source_col in candidate.columns:
                    candidate[source_col] = pd.to_numeric(
                        candidate[source_col],
                        errors="coerce"
                    )
                    agg[output_col] = (source_col, "mean")

            ranking_source = (
                candidate
                .groupby("Filter", as_index=False)
                .agg(**agg)
            )

            ranking_source["Noise_Conditions"] = (
                candidate.groupby("Filter")["Noise"].nunique().values
                if "Noise" in candidate.columns
                else 0
            )

# Build the final ranking in a consistent format.
if not ranking_source.empty and "Filter" in ranking_source.columns:
    if "Average_Composite_Score" not in ranking_source.columns:
        if "Composite_Score" in ranking_source.columns:
            ranking_source["Average_Composite_Score"] = pd.to_numeric(
                ranking_source["Composite_Score"],
                errors="coerce"
            )
        else:
            ranking_source["Average_Composite_Score"] = np.nan

    # Normalize optional columns.
    rename_map = {
        "PSNR": "Average_PSNR",
        "SSIM": "Average_SSIM",
        "MAE": "Average_MAE",
        "Edge_Preservation": "Average_Edge_Preservation",
        "Runtime_ms": "Average_Runtime_ms"
    }

    for source, target in rename_map.items():
        if target not in ranking_source.columns and source in ranking_source.columns:
            ranking_source[target] = pd.to_numeric(
                ranking_source[source],
                errors="coerce"
            )

    for column in [
        "Average_Composite_Score",
        "Average_PSNR",
        "Average_SSIM",
        "Average_MAE",
        "Average_Edge_Preservation",
        "Average_Runtime_ms"
    ]:
        if column not in ranking_source.columns:
            ranking_source[column] = np.nan

    if "Noise_Conditions" not in ranking_source.columns:
        ranking_source["Noise_Conditions"] = 0

    ranking = ranking_source[
        [
            "Filter",
            "Average_Composite_Score",
            "Average_PSNR",
            "Average_SSIM",
            "Average_MAE",
            "Average_Edge_Preservation",
            "Average_Runtime_ms",
            "Noise_Conditions"
        ]
    ].copy()

    ranking["Average_Composite_Score"] = pd.to_numeric(
        ranking["Average_Composite_Score"],
        errors="coerce"
    )

    ranking = ranking.dropna(
        subset=["Average_Composite_Score"]
    )

    ranking = (
        ranking
        .sort_values(
            "Average_Composite_Score",
            ascending=False
        )
        .reset_index(drop=True)
    )

    ranking["Rank"] = ranking.index + 1

    ranking = ranking[
        [
            "Rank",
            "Filter",
            "Average_Composite_Score",
            "Average_PSNR",
            "Average_SSIM",
            "Average_MAE",
            "Average_Edge_Preservation",
            "Average_Runtime_ms",
            "Noise_Conditions"
        ]
    ]

    ranking_path = os.path.join(
        FINAL_DIR,
        "final_filter_ranking.csv"
    )

    ranking.to_csv(
        ranking_path,
        index=False
    )

    if not ranking.empty:
        best_filter = str(ranking.iloc[0]["Filter"])
        best_score = safe_float(
            ranking.iloc[0]["Average_Composite_Score"]
        )

        print("\nFINAL FILTER RANKING")
        print("-" * 70)

        for _, row in ranking.iterrows():
            print(
                f"{int(row['Rank'])}. "
                f"{row['Filter']} | "
                f"Average Composite Score: "
                f"{safe_float(row['Average_Composite_Score']):.4f}"
            )

        print("-" * 70)
        print(f"BEST OVERALL FILTER: {best_filter}")
        print(f"AVERAGE COMPOSITE SCORE: {best_score:.4f}")

else:
    ranking = pd.DataFrame()
    print("WARNING: Overall filter ranking could not be determined.")


# ============================================================
# [3/7] LOAD OBJECT ANALYSIS
# ============================================================




print("\n[3/7] Loading object analysis...")



object_measurements_path = os.path.join(

    OBJECTS_DIR,

    "object_measurements.csv"

)



object_measurements = load_csv(

    object_measurements_path,

    "object measurements"

)





object_count = len(object_measurements)



print(

    f"Objects measurement rows : "

    f"{object_count}"

)





# ============================================================

# [4/7] LOAD REGION / SHAPE DESCRIPTORS

# ============================================================



print("\n[4/7] Loading shape descriptors...")



shape_descriptors_path = os.path.join(

    SHAPE_DIR,

    "region_shape_descriptors.csv"

)



shape_summary_path = os.path.join(

    SHAPE_DIR,

    "shape_descriptor_summary.csv"

)



shape_descriptors = load_csv(

    shape_descriptors_path,

    "shape descriptors"

)



shape_summary = load_csv(

    shape_summary_path,

    "shape descriptor summary"

)



print(

    f"Shape descriptor rows : "

    f"{len(shape_descriptors)}"

)



if not shape_summary.empty:

    print("Shape descriptor summary loaded")





# ============================================================

# EXTRACT SHAPE STATISTICS

# ============================================================



shape_stats = {}



if not shape_summary.empty:



    # Usually first row contains mean values

    mean_row = shape_summary.iloc[0]



    for column in [

        "Area",

        "Perimeter",

        "Circularity",

        "Aspect_Ratio",

        "Extent",

        "Solidity",

        "Equivalent_Diameter"

    ]:



        if column in shape_summary.columns:



            shape_stats[column] = safe_float(

                mean_row[column]

            )



else:



    shape_stats = {

        "Area": 0.0,

        "Perimeter": 0.0,

        "Circularity": 0.0,

        "Aspect_Ratio": 0.0,

        "Extent": 0.0,

        "Solidity": 0.0,

        "Equivalent_Diameter": 0.0

    }





# ============================================================

# [5/7] CREATE FINAL PROJECT SUMMARY

# ============================================================



print("\n[5/7] Creating final project summary...")





summary_lines = []



summary_lines.append(

    "COMPUTER VISION NOISE ANALYSIS PROJECT"

)



summary_lines.append(

    "=" * 55

)



summary_lines.append("")



# ------------------------------------------------------------

# PROJECT SCOPE

# ------------------------------------------------------------



summary_lines.append("PROJECT SCOPE")

summary_lines.append("-" * 55)



summary_lines.append(

    "Analysis of image noise, filtering techniques, "

    "image-quality metrics, edge preservation, "

    "runtime performance, object detection, "

    "and region/shape descriptors."

)



summary_lines.append("")



# ------------------------------------------------------------

# FILTER PERFORMANCE

# ------------------------------------------------------------



summary_lines.append("FILTER PERFORMANCE")

summary_lines.append("-" * 55)



if best_filter != "Unavailable":



    summary_lines.append(

        f"Best overall filter: {best_filter}"

    )



    summary_lines.append(

        f"Average composite score: "

        f"{best_score:.4f}"

    )



else:



    summary_lines.append(

        "Best overall filter could not be automatically determined."

    )



summary_lines.append("")



summary_lines.append(

    f"Filter evaluation records: "

    f"{len(filter_results)}"

)



if not filter_results.empty and "Filter" in filter_results.columns:



    filters = (

        filter_results["Filter"]

        .dropna()

        .unique()

        .tolist()

    )



    summary_lines.append(

        "Filters evaluated: "

        + ", ".join(map(str, filters))

    )



else:



    summary_lines.append(

        "Filters evaluated: Mean, Gaussian, Median, Bilateral"

    )





if (

    not filter_results.empty

    and "Noise" in filter_results.columns

):



    noise_conditions = (

        filter_results["Noise"]

        .dropna()

        .unique()

        .tolist()

    )



    summary_lines.append(

        f"Noise conditions evaluated: "

        f"{len(noise_conditions)}"

    )



else:



    summary_lines.append(

        "Noise conditions evaluated: 6"

    )



summary_lines.append("")





# ------------------------------------------------------------

# BEST FILTER BY NOISE CONDITION

# ------------------------------------------------------------



summary_lines.append(

    "BEST FILTER BY NOISE CONDITION"

)



summary_lines.append(

    "-" * 55

)



if not best_by_noise.empty:



    for _, row in best_by_noise.iterrows():



        noise = str(row["Noise"])

        filter_name = str(row["Filter"])



        score = safe_float(

            row["Composite_Score"]

        )



        summary_lines.append(

            f"{noise}: {filter_name} "

            f"(Composite Score: {score:.4f})"

        )



else:



    summary_lines.append(

        "Noise-specific filter ranking unavailable."

    )



summary_lines.append("")





# ------------------------------------------------------------

# OBJECT ANALYSIS

# ------------------------------------------------------------



summary_lines.append("OBJECT ANALYSIS")

summary_lines.append("-" * 55)



summary_lines.append(

    f"Object measurement records: "

    f"{object_count}"

)



summary_lines.append("")





# ------------------------------------------------------------

# REGION / SHAPE ANALYSIS

# ------------------------------------------------------------



summary_lines.append(

    "REGION / SHAPE ANALYSIS"

)



summary_lines.append(

    "-" * 55

)



summary_lines.append(

    f"Objects analyzed for shape descriptors: "

    f"{len(shape_descriptors)}"

)



summary_lines.append(

    f"Mean Area: "

    f"{shape_stats.get('Area', 0.0):.4f}"

)



summary_lines.append(

    f"Mean Perimeter: "

    f"{shape_stats.get('Perimeter', 0.0):.4f}"

)



summary_lines.append(

    f"Mean Circularity: "

    f"{shape_stats.get('Circularity', 0.0):.4f}"

)



summary_lines.append(

    f"Mean Aspect Ratio: "

    f"{shape_stats.get('Aspect_Ratio', 0.0):.4f}"

)



summary_lines.append(

    f"Mean Extent: "

    f"{shape_stats.get('Extent', 0.0):.4f}"

)



summary_lines.append(

    f"Mean Solidity: "

    f"{shape_stats.get('Solidity', 0.0):.4f}"

)



summary_lines.append(

    f"Mean Equivalent Diameter: "

    f"{shape_stats.get('Equivalent_Diameter', 0.0):.4f}"

)



summary_lines.append("")





# ------------------------------------------------------------

# IMPLEMENTATION VERIFICATION

# ------------------------------------------------------------



summary_lines.append(

    "IMPLEMENTATION VERIFICATION"

)



summary_lines.append(

    "-" * 55

)



summary_lines.append(

    "Scratch implementations were compared against "

    "OpenCV filtering operations."

)



summary_lines.append(

    "Verification outputs are stored under "

    "outputs/verification/."

)



summary_lines.append("")





# ------------------------------------------------------------

# GENERATED VISUALIZATIONS

# ------------------------------------------------------------



summary_lines.append(

    "GENERATED VISUALIZATIONS"

)



summary_lines.append(

    "-" * 55

)



visualization_files = [

    "01_psnr_comparison.png",

    "02_ssim_comparison.png",

    "03_mae_comparison.png",

    "04_edge_preservation.png",

    "05_runtime_comparison.png",

    "06_noise_comparison.png",

    "08_filter_heatmap.png",

    "09_final_comparison.png"

]



for filename in visualization_files:



    path = os.path.join(

        VIS_DIR,

        filename

    )



    if os.path.exists(path):



        summary_lines.append(

            f"- {filename}"

        )





summary_lines.append("")





# ------------------------------------------------------------

# FINAL CONCLUSION

# ------------------------------------------------------------



summary_lines.append(

    "CONCLUSION"

)



summary_lines.append(

    "-" * 55

)



if best_filter != "Unavailable":



    summary_lines.append(

        f"The analysis identifies {best_filter} as the "

        f"best overall filtering technique among the "

        f"noise-specific best-performing filters, based "

        f"on the average composite score."

    )



else:



    summary_lines.append(

        "The project evaluates image restoration under "

        "multiple noise conditions using quantitative "

        "image-quality metrics."

    )



summary_lines.append(

    "The project evaluates image restoration under "

    "multiple noise conditions using PSNR, SSIM, MAE, "

    "edge preservation, runtime analysis, object detection, "

    "and shape descriptors."

)



summary_lines.append(

    "The results provide a quantitative basis for selecting "

    "an appropriate filtering technique for noisy image data."

)



summary_lines.append("")





# ------------------------------------------------------------

# SAVE SUMMARY

# ------------------------------------------------------------



summary_path = os.path.join(

    FINAL_DIR,

    "final_project_summary.txt"

)



with open(

    summary_path,

    "w",

    encoding="utf-8"

) as file:



    file.write(

        "\n".join(summary_lines)

    )



print(

    f"Saved final summary: {summary_path}"

)





# ============================================================

# [6/7] GENERATE FINAL CONSOLIDATED CHART

# ============================================================



print(

    "\n[6/7] Generating final consolidated chart..."

)





if not ranking.empty:



    chart_path = os.path.join(

        FINAL_DIR,

        "final_consolidated_analysis.png"

    )



    chart_data = ranking.copy()



    chart_data = chart_data.sort_values(

        "Average_Composite_Score",

        ascending=True

    )



    plt.figure(

        figsize=(10, 6)

    )



    plt.barh(

        chart_data["Filter"],

        chart_data["Average_Composite_Score"]

    )



    plt.xlabel(

        "Average Composite Score"

    )



    plt.ylabel(

        "Filter"

    )



    plt.title(

        "Final Filter Performance Ranking"

    )



    plt.xlim(

        0,

        max(

            1.0,

            chart_data[

                "Average_Composite_Score"

            ].max() * 1.1

        )

    )



    plt.grid(

        axis="x",

        alpha=0.3

    )



    plt.tight_layout()



    plt.savefig(

        chart_path,

        dpi=300,

        bbox_inches="tight"

    )



    plt.close()



    print(

        f"Saved final consolidated chart: "

        f"{chart_path}"

    )



else:



    chart_path = ""



    print(

        "WARNING: Ranking unavailable; "

        "final chart was not generated."

    )





# ============================================================

# [7/7] CREATE OUTPUT INVENTORY

# ============================================================



print(

    "\n[7/7] Creating final output inventory..."

)





inventory = []





# Search all project output files

OUTPUT_ROOT = os.path.join(

    BASE_DIR,

    "outputs"

)



for root, dirs, files in os.walk(

    OUTPUT_ROOT

):



    for filename in files:



        full_path = os.path.join(

            root,

            filename

        )



        relative_path = os.path.relpath(

            full_path,

            BASE_DIR

        )



        try:

            size_kb = (

                os.path.getsize(full_path)

                / 1024

            )

        except OSError:

            size_kb = 0.0



        inventory.append(

            {

                "File": relative_path,

                "Size_KB": round(

                    size_kb,

                    2

                )

            }

        )





inventory_df = pd.DataFrame(

    inventory

)



inventory_df = inventory_df.sort_values(

    "File"

)



inventory_path = os.path.join(

    FINAL_DIR,

    "output_inventory.csv"

)



inventory_df.to_csv(

    inventory_path,

    index=False

)



print(

    f"Saved output inventory: "

    f"{inventory_path}"

)





# ============================================================

# FINAL STATUS

# ============================================================



print("\n" + "=" * 70)

print(

    "FINAL INTEGRATED ANALYSIS COMPLETED SUCCESSFULLY"

)

print("=" * 70)



print("\nGenerated final-analysis files:")



print(

    " -> outputs/final_analysis/"

    "final_project_summary.txt"

)



print(

    " -> outputs/final_analysis/"

    "final_filter_ranking.csv"

)



print(

    " -> outputs/final_analysis/"

    "best_filter_by_noise.csv"

)



if chart_path:

    print(

        " -> outputs/final_analysis/"

        "final_consolidated_analysis.png"

    )



print(

    " -> outputs/final_analysis/"

    "output_inventory.csv"

)



print("\nFINAL RESULT")



if best_filter != "Unavailable":



    print(

        f"Best overall filter : {best_filter}"

    )



    print(

        f"Composite score     : {best_score:.4f}"

    )



else:



    print(

        "Best overall filter : Unavailable"

    )



print(

    f"Objects analyzed     : "

    f"{len(shape_descriptors)}"

)



print(

    "Project analysis pipeline is now consolidated."

)



print("=" * 70)
