"""
Written by Ridwan Alrefai

University of Illinois at Urbana-Champaign

CS 598 DLH - Labrador

This script trains a simple baseline classifier on the COVID diagnosis dataset.


"""

import os
import json
import time


import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
)


INPUT_CSV = "data/processed/covid_diagnosis_baselines_data.csv"
OUT_DIR = "CS598_results/covid_baseline"
LABEL_COL = "target"
RANDOM_STATE = 42
TEST_SIZE = 0.3   # 30% of total for temp (val+test)
VAL_SIZE = 0.5    # 50% of temp → 15% val, 15% test overall

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)



def load_data(path: str, label_col: str):
    df = pd.read_csv(path)
    if label_col not in df.columns:
        raise ValueError(f"Label column '{label_col}' not found in {path}")
    X = df.drop(columns=[label_col])
    y = df[label_col].values
    return X, y, df.columns.tolist()


def split_data(X, y):
    # First split: train vs (val+test)
    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    # Second split: val vs test (50/50 of temp)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=VAL_SIZE,
        stratify=y_temp,
        random_state=RANDOM_STATE,
    )

    splits = {
        "train": (X_train, y_train),
        "val": (X_val, y_val),
        "test": (X_test, y_test),
    }
    return splits
    

def build_pipeline():
    # Baseline: impute missing values, standardize, then logistic regression
    pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",  # helps if classes are imbalanced
                    solver="lbfgs",
                ),
            ),
        ]
    )
    return pipe

def evaluate_model(model, X, y, split_name: str):
    # Probabilities for positive class
    probs = model.predict_proba(X)[:, 1]
    preds = model.predict(X)

    metrics = {
        "split": split_name,
        "roc_auc": float(roc_auc_score(y, probs)),
        "pr_auc": float(average_precision_score(y, probs)),
        "f1": float(f1_score(y, preds)),
        "num_samples": int(len(y)),
    }
    return metrics


def main():
    print(f"[INFO] Loading data from {INPUT_CSV}")
    X, y, columns = load_data(INPUT_CSV, LABEL_COL)
    print(f"[INFO] Data shape: X={X.shape}, y={y.shape}")
    print(f"[INFO] Number of features: {X.shape[1]}")

    print("[INFO] Splitting into train/val/test")
    splits = split_data(X, y)
    X_train, y_train = splits["train"]
    X_val, y_val = splits["val"]
    X_test, y_test = splits["test"]

    print(
        f"[INFO] Split sizes: "
        f"train={len(y_train)}, val={len(y_val)}, test={len(y_test)}"
    )

    ensure_outdir(OUT_DIR)

    print("[INFO] Building baseline pipeline (StandardScaler + LogisticRegression)")
    model = build_pipeline()

    print("[INFO] Training baseline model...")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"[INFO] Training complete in {train_time:.2f} seconds")

    print("[INFO] Evaluating model on all splits...")
    all_metrics = []
    for split_name, (X_split, y_split) in [
        ("train", (X_train, y_train)),
        ("val", (X_val, y_val)),
        ("test", (X_test, y_test)),
    ]:
        metrics = evaluate_model(model, X_split, y_split, split_name)
        all_metrics.append(metrics)
        print(
            f"[{split_name.upper()}] "
            f"ROC-AUC={metrics['roc_auc']:.4f}, "
            f"PR-AUC={metrics['pr_auc']:.4f}, "
            f"F1={metrics['f1']:.4f}, "
            f"n={metrics['num_samples']}"
        )

    # Hyperparameters & training details for the paper
    hyperparams = {
        "model": "LogisticRegression",
        "pipeline": "StandardScaler -> LogisticRegression",
        "label_column": LABEL_COL,
        "random_state": RANDOM_STATE,
        "splits": {
            "train_size": int(len(y_train)),
            "val_size": int(len(y_val)),
            "test_size": int(len(y_test)),
            "test_fraction_of_total": float(TEST_SIZE * VAL_SIZE),
        },
        "logistic_regression_params": {
            "max_iter": 1000,
            "class_weight": "balanced",
            "solver": "lbfgs",
        },
        "training_time_seconds": float(train_time),
        "features": columns,
    }

    metrics_path = os.path.join(OUT_DIR, "metrics.json")
    hyperparams_path = os.path.join(OUT_DIR, "hyperparams.json")

    print(f"[INFO] Saving metrics to {metrics_path}")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=4)

    print(f"[INFO] Saving hyperparameters to {hyperparams_path}")
    with open(hyperparams_path, "w") as f:
        json.dump(hyperparams, f, indent=4)

    print("[DONE] Baseline training and evaluation completed.")


if __name__ == "__main__":
    main()





