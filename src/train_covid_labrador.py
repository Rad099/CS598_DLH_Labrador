



import os
import json
import time
import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

from lab_transformers.models.labrador.finetuning_wrapper import (
    LabradorFinetuneWrapper,
)

NPZ_PATH = "data/evaluations/covid_diagnosis_labrador_data.npz"  # or your actual filename
TRAIN_CONFIG_PATH = "configs/covid_diagnosis/covid_diagnosis_labrador.json"  # training hyperparams (optional)
MODEL_CONFIG_PATH = "model_weights/labrador/config.json"  # real Labrador model params
BASE_MODEL_PATH = "model_weights/labrador/variables/variables" # weights checkpoint
OUT_DIR = "CS598_results/covid_labrador"


RANDOM_STATE = 42
TEST_SIZE = 0.3   # 30% of total for temp (val+test)
VAL_SIZE = 0.5    # 50% of temp → 15% val, 15% test overall

np.random.seed(RANDOM_STATE)
tf.random.set_seed(RANDOM_STATE)

def ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def load_json_config(path: str) -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            cfg = json.load(f)
        print(f"[INFO] Loaded config from {path}")
    else:
        print(f"[WARN] Config file {path} not found. Using empty dict.")
        cfg = {}
    return cfg

def load_npz(path: str):
    if not os.path.exists(path):
        raise FileNotFoundError(f"NPZ file not found at {path}")
    data = np.load(path)
    print(f"[INFO] Loaded NPZ from {path}")
    print(f"[INFO] Keys in NPZ: {list(data.keys())}")

    X_cat = data["categorical_input"]
    X_cont = data["continuous_input"]
    y = data["label"].reshape(-1)

    non_mimic_features = None
    if "non_mimic_features_discrete" in data.files or "non_mimic_features_continuous" in data.files:
        print("[INFO] Found non-mimic feature arrays. Concatenating them.")
        parts = []
        if "non_mimic_features_discrete" in data.files:
            parts.append(data["non_mimic_features_discrete"])
        if "non_mimic_features_continuous" in data.files:
            parts.append(data["non_mimic_features_continuous"])
        non_mimic_features = np.concatenate(parts, axis=1)

    # 🔴 Sanitize NaNs in all inputs
    X_cat = np.nan_to_num(X_cat, nan=0)
    X_cont = np.nan_to_num(X_cont, nan=0.0)
    if non_mimic_features is not None:
        non_mimic_features = np.nan_to_num(non_mimic_features, nan=0.0)

    print(f"[INFO] Shapes: categorical_input={X_cat.shape}, continuous_input={X_cont.shape}, y={y.shape}")
    if non_mimic_features is not None:
        print(f"[INFO] non_mimic_features shape={non_mimic_features.shape}")

    return X_cat, X_cont, non_mimic_features, y




def make_splits(y: np.ndarray):
    idx = np.arange(len(y))

    train_idx, temp_idx, y_train, y_temp = train_test_split(
        idx, y, test_size=TEST_SIZE, stratify=y, random_state=RANDOM_STATE
    )
    val_idx, test_idx, y_val, y_test = train_test_split(
        temp_idx, y_temp, test_size=VAL_SIZE, stratify=y_temp, random_state=RANDOM_STATE
    )

    splits = {
        "train": train_idx,
        "val": val_idx,
        "test": test_idx,
    }

    print(
        f"[INFO] Split sizes: "
        f"train={len(train_idx)}, val={len(val_idx)}, test={len(test_idx)}"
    )

    return splits

def make_tf_dataset(
    X_cat: np.ndarray,
    X_cont: np.ndarray,
    non_mimic: np.ndarray,
    y: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
):
    # Slice arrays by indices
    x_cat_slice = X_cat[indices]
    x_cont_slice = X_cont[indices]
    y_slice = y[indices].astype("float32")

    inputs = {
        "categorical_input": x_cat_slice,
        "continuous_input": x_cont_slice,
    }

    if non_mimic is not None:
        inputs["non_mimic_features"] = non_mimic[indices].astype("float32")

    ds = tf.data.Dataset.from_tensor_slices((inputs, y_slice))
    if shuffle:
        ds = ds.shuffle(buffer_size=len(indices), seed=RANDOM_STATE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


def build_model(train_cfg: dict, model_cfg: dict, seq_len: int) -> tf.keras.Model:
    """
    Build a LabradorFinetuneWrapper using:
    - model_cfg from model_weights/labrador/config.json
    - seq_len from the COVID NPZ (for max_seq_length)
    """

    # Copy so we don't mutate the original dict
    model_params = dict(model_cfg)

    # Ensure include_head is a proper boolean if it's serialized as string
    if isinstance(model_params.get("include_head", True), str):
        model_params["include_head"] = model_params["include_head"].lower() == "true"

    # Inject max_seq_length if missing
    if "max_seq_length" not in model_params:
        print(f"[INFO] 'max_seq_length' not found in model_params. Setting it to seq_len={seq_len}")
        model_params["max_seq_length"] = int(seq_len)

    # Training-level hyperparameters
    dropout_rate = train_cfg.get("dropout_rate", model_params.get("dropout_rate", 0.1))
    output_size = train_cfg.get("output_size", 1)
    output_activation = train_cfg.get("output_activation", "sigmoid")

    print("[INFO] Building LabradorFinetuneWrapper with pretrained weights...")
    print(f"[INFO] base_model_path = {BASE_MODEL_PATH}")
    print(f"[INFO] model_params (subset): "
          f"vocab_size={model_params.get('vocab_size')}, "
          f"embedding_dim={model_params.get('embedding_dim')}, "
          f"heads={model_params.get('transformer_heads')}, "
          f"blocks={model_params.get('transformer_blocks')}, "
          f"max_seq_length={model_params.get('max_seq_length')}")

    model = LabradorFinetuneWrapper(
        base_model_path=BASE_MODEL_PATH,   # load pretrained weights
        output_size=output_size,
        output_activation=output_activation,
        model_params=model_params,
        dropout_rate=dropout_rate,
        add_extra_dense_layer=False,
        train_base_model=False,            # fine-tune the backbone
    )

    return model


def main():
    ensure_outdir(OUT_DIR)

    train_cfg = load_json_config(TRAIN_CONFIG_PATH)
    model_cfg = load_json_config(MODEL_CONFIG_PATH)

    # Training hyperparameters with defaults if not provided
    batch_size = train_cfg.get("batch_size", 64)
    learning_rate = train_cfg.get("learning_rate", 1e-4)
    num_epochs = train_cfg.get("epochs", 20)

    print(f"[INFO] Training hyperparameters: batch_size={batch_size}, lr={learning_rate}, epochs={num_epochs}")

    # Load data
    X_cat, X_cont, non_mimic, y = load_npz(NPZ_PATH)
    splits = make_splits(y)

    seq_len = X_cat.shape[1]
    print(f"[INFO] Using seq_len={seq_len} for max_seq_length")


    train_idx = splits["train"]
    val_idx = splits["val"]
    test_idx = splits["test"]

    # Build datasets
    train_ds = make_tf_dataset(X_cat, X_cont, non_mimic, y, train_idx, batch_size, shuffle=True)
    val_ds = make_tf_dataset(X_cat, X_cont, non_mimic, y, val_idx, batch_size, shuffle=False)
    test_ds = make_tf_dataset(X_cat, X_cont, non_mimic, y, test_idx, batch_size, shuffle=False)

    # Build model
    model = build_model(train_cfg, model_cfg, seq_len)

    # Compile
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )

    # Train
    print("[INFO] Starting training...")
    start_time = time.time()
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=num_epochs,
    )
    train_time = time.time() - start_time
    print(f"[INFO] Training complete in {train_time:.2f} seconds")

    # Evaluate on val & test with sklearn metrics
    def eval_split(ds, split_name: str):
        y_true_all = []
        y_prob_all = []
        for batch_x, batch_y in ds:
            y_prob = model(batch_x, training=False).numpy().ravel()
            y_true_all.append(batch_y.numpy())
            y_prob_all.append(y_prob)

        y_true_all = np.concatenate(y_true_all).reshape(-1)
        y_prob_all = np.concatenate(y_prob_all).reshape(-1)
        y_pred_all = (y_prob_all >= 0.5).astype(int)

        metrics = {
            "split": split_name,
            "roc_auc": float(roc_auc_score(y_true_all, y_prob_all)),
            "pr_auc": float(average_precision_score(y_true_all, y_prob_all)),
            "f1": float(f1_score(y_true_all, y_pred_all)),
            "num_samples": int(len(y_true_all)),
        }
        return metrics

    val_metrics = eval_split(val_ds, "val")
    test_metrics = eval_split(test_ds, "test")

    print(
        f"[VAL]  ROC-AUC={val_metrics['roc_auc']:.4f}, "
        f"PR-AUC={val_metrics['pr_auc']:.4f}, "
        f"F1={val_metrics['f1']:.4f}, "
        f"n={val_metrics['num_samples']}"
    )
    print(
        f"[TEST] ROC-AUC={test_metrics['roc_auc']:.4f}, "
        f"PR-AUC={test_metrics['pr_auc']:.4f}, "
        f"F1={test_metrics['f1']:.4f}, "
        f"n={test_metrics['num_samples']}"
    )

    # Save metrics & hyperparameters
    metrics_out = {
        "val": val_metrics,
        "test": test_metrics,
    }

    hyperparams = {
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": num_epochs,
        "random_state": RANDOM_STATE,
        "base_model_path": BASE_MODEL_PATH,
        "config_path": MODEL_CONFIG_PATH,
        "train_size": int(len(train_idx)),
        "val_size": int(len(val_idx)),
        "test_size": int(len(test_idx)),
        "training_time_seconds": float(train_time),
    }

    metrics_path = os.path.join(OUT_DIR, "metrics.json")
    hyperparams_path = os.path.join(OUT_DIR, "hyperparams.json")

    print(f"[INFO] Saving metrics to {metrics_path}")
    with open(metrics_path, "w") as f:
        json.dump(metrics_out, f, indent=4)

    print(f"[INFO] Saving hyperparameters to {hyperparams_path}")
    with open(hyperparams_path, "w") as f:
        json.dump(hyperparams, f, indent=4)

    print("[DONE] Labrador fine-tuning and evaluation completed.")


if __name__ == "__main__":
    main()


