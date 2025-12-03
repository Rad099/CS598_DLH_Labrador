"""
Created by Ridwan Alrefai on 11/23/2025

University of Illinois at Urbana-Champaign
CS 598 DLH - Labrador

Analytics / preprocessing script for the COVID diagnosis baseline dataset.

Usage (from repo root):
    python analysis_covid_baseline.py

It will:
  - Load data/processed/covid_diagnosis_baselines_data.csv
  - Compute basic dataset stats
  - Compute label distribution
  - Compute missingness per column
  - Compute summary stats for numeric features
  - Compute per-class means for numeric features
  - Save everything to results/covid_baseline_*
  - Generate a label distribution plot + Age histogram by class



"""


import os
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# --------- CONFIG ---------
INPUT_PATH = "data/processed/covid_diagnosis_baselines_data.csv"
OUT_DIR = "results/covid_baseline"

# Columns you told me exist:
ALL_COLUMNS = [
    "Sex", "Age",
    "50893","50912","50863","50927","50931","50878","50861","50954",
    "PCR","KAL","NAT",
    "51301","51279","51222","51221","51250","51248","51249","51265",
    "51256","51244","51254","51200","51146",
    "52075","51133","52074","52073","52069",
    "Suspect",
    "target",
]

LABEL_COL = "target"
CATEGORICAL_COLS = ["Sex", "Suspect"]  

NUMERIC_COLS = [c for c in ALL_COLUMNS if c not in CATEGORICAL_COLS + [LABEL_COL]]


def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    # Basic sanity check: ensure expected columns exist
    missing = set(ALL_COLUMNS) - set(df.columns)
    if missing:
        print(f"[WARN] Missing expected columns in CSV: {missing}")
    # Drop rows with missing label, if any
    if df[LABEL_COL].isna().any():
        print("[INFO] Dropping rows with missing label")
        df = df.dropna(subset=[LABEL_COL])
    return df


def basic_stats(df: pd.DataFrame) -> dict:
    return {
        "num_rows": int(len(df)),
        "num_columns": int(len(df.columns)),
        "columns": df.columns.tolist(),
    }


def label_stats(df: pd.DataFrame) -> dict:
    vc = df[LABEL_COL].value_counts(dropna=False).to_dict()
   
    total = len(df)
    proportions = {str(k): float(v) / total for k, v in vc.items()}
    return {
        "counts": {str(k): int(v) for k, v in vc.items()},
        "proportions": proportions,
    }


def missingness_stats(df: pd.DataFrame) -> dict:
    return df.isna().sum().astype(int).to_dict()


def numeric_summary(df: pd.DataFrame) -> dict:
    numeric_df = df[NUMERIC_COLS].select_dtypes(include=[np.number])
    desc = numeric_df.describe().T  # rows = features, cols = stats
    # Convert to nested dict: {feature: {stat: value}}
    summary = {}
    for col in desc.index:
        summary[col] = {stat: float(desc.loc[col, stat]) for stat in desc.columns}
    return summary


def per_class_means(df: pd.DataFrame) -> dict:
    numeric_df = df[NUMERIC_COLS].select_dtypes(include=[np.number])
    grouped = numeric_df.join(df[LABEL_COL]).groupby(LABEL_COL).mean()
    out = {}
    for cls in grouped.index:
        out[str(cls)] = {col: float(grouped.loc[cls, col]) for col in grouped.columns}
    return out


def save_json(obj: dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


def plot_label_distribution(df: pd.DataFrame, out_path: str) -> None:
    plt.figure()
    df[LABEL_COL].value_counts().sort_index().plot(kind="bar")
    plt.title("COVID Diagnosis Baseline - Label Distribution")
    plt.xlabel("target")
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def plot_age_hist_by_class(df: pd.DataFrame, out_path: str) -> None:
    if "Age" not in df.columns:
        print("[WARN] Age column not found; skipping age histogram.")
        return
    plt.figure()
    classes = sorted(df[LABEL_COL].unique())
    for cls in classes:
        subset = df[df[LABEL_COL] == cls]["Age"].dropna()
        plt.hist(
            subset,
            bins=20,
            alpha=0.5,
            label=f"target={cls}",
        )
    plt.xlabel("Age")
    plt.ylabel("Count")
    plt.title("Age Distribution by COVID Target Class")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()


def main():
    print(f"[INFO] Loading data from {INPUT_PATH}")
    df = load_data(INPUT_PATH)

    ensure_outdir(OUT_DIR)

    # 1) Basic dataset stats
    stats = basic_stats(df)
    print(f"[INFO] Basic stats: {stats}")

    # 2) Label distribution
    label_info = label_stats(df)
    print(f"[INFO] Label distribution: {label_info}")

    # 3) Missingness
    missing = missingness_stats(df)

    # 4) Numeric summary stats
    num_summary = numeric_summary(df)

    # 5) Per-class means
    class_means = per_class_means(df)

    # Bundle everything into one JSON
    summary = {
        "basic_stats": stats,
        "label_stats": label_info,
        "missingness": missing,
        "numeric_summary": num_summary,
        "per_class_means": class_means,
        "feature_groups": {
            "demographics": ["Sex", "Age"],
            "covid_specific": ["PCR", "KAL", "NAT"],
            "lab_values": [
                "50893","50912","50863","50927","50931","50878","50861","50954",
                "51301","51279","51222","51221","51250","51248","51249","51265",
                "51256","51244","51254","51200","51146",
                "52075","51133","52074","52073","52069",
            ],
            "categorical_variables": ["Sex", "Suspect"],
            "label": [LABEL_COL],
        },
    }

    json_path = os.path.join(OUT_DIR, "covid_baseline_stats.json")
    print(f"[INFO] Saving JSON stats to {json_path}")
    save_json(summary, json_path)

    # Plots
    label_plot_path = os.path.join(OUT_DIR, "covid_label_distribution.png")
    age_plot_path = os.path.join(OUT_DIR, "covid_age_hist_by_class.png")

    print(f"[INFO] Saving label distribution plot to {label_plot_path}")
    plot_label_distribution(df, label_plot_path)

    print(f"[INFO] Saving age histogram plot to {age_plot_path}")
    plot_age_hist_by_class(df, age_plot_path)

    print(f"[DONE] Analysis complete. Outputs written to {OUT_DIR}")


if __name__ == "__main__":
    main()
