"""
Unified model error analysis on validation split.

Generates:
- Error confidence histogram (mispredicted samples)
- Confusion matrix (full validation split, mapped to local labels when filtering by dataset)

Usage:
  python error_analysis.py --model lgbm --dataset heloc --save-dir analysis_outputs --show
"""

import argparse
import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix

# Ensure src is on path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data_utils import (
    load_unified_train_data,
    DATASET_ID_MAP,
    LABEL_OFFSET_MAP,
    DATASET_CLASS_COUNTS,
)
from predict import load_unified_model, load_unified_metadata

DEFAULT_SAVE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "analysis_outputs")


def map_to_local(labels, dataset):
    offset = LABEL_OFFSET_MAP[dataset]
    return labels - offset


def compute_confidence(proba):
    if proba is None:
        return None
    return float(np.max(proba))


def plot_confusion(cm, class_names, out_path, title):
    plt.figure(figsize=(6, 5))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title(title)
    plt.colorbar()
    ticks = np.arange(len(class_names))
    plt.xticks(ticks, class_names, rotation=45)
    plt.yticks(ticks, class_names)
    thresh = cm.max() / 2.0 if cm.size else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(
                j,
                i,
                format(cm[i, j], "d"),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Unified model error analysis on validation split")
    parser.add_argument("--model", type=str, default="lgbm", help="Model name used in training")
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["all", "covtype", "heloc", "higgs"],
        help="Restrict analysis to one dataset (by dataset_id feature)",
    )
    parser.add_argument(
        "--save-dir",
        type=str,
        default=DEFAULT_SAVE_DIR,
        help="Directory to save plots",
    )
    parser.add_argument(
        "--show",
        action="store_true",
        help="Display plots interactively (may block)",
    )
    args = parser.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    # Load data and model
    X, y, union_features = load_unified_train_data()
    dataset_ids = X[:, -1].astype(int)
    from sklearn.model_selection import train_test_split

    X_tr, X_val, y_tr, y_val, ds_tr, ds_val = train_test_split(
        X, y, dataset_ids, test_size=0.15, random_state=42, stratify=y
    )

    model = load_unified_model(args.model)

    # Predict on validation split
    preds = model.predict(X_val)
    proba = model.predict_proba(X_val) if hasattr(model, "predict_proba") else None

    # Optional dataset filter
    mask = np.ones(len(y_val), dtype=bool)
    if args.dataset != "all":
        target_id = DATASET_ID_MAP[args.dataset]
        mask = ds_val == target_id
    y_val_f = y_val[mask]
    preds_f = preds[mask]
    proba_f = proba[mask] if proba is not None else None
    ds_val_f = ds_val[mask]

    # For per-dataset, map unified labels to local
    if args.dataset != "all":
        y_val_local = map_to_local(y_val_f, args.dataset)
        preds_local = map_to_local(preds_f, args.dataset)
        n_classes = DATASET_CLASS_COUNTS[args.dataset]
        class_names = [f"class_{i}" for i in range(n_classes)]
        cm = confusion_matrix(y_val_local, preds_local, labels=list(range(n_classes)))
    else:
        n_classes = len(np.unique(y))
        class_names = [str(i) for i in range(n_classes)]
        cm = confusion_matrix(y_val_f, preds_f, labels=list(range(n_classes)))

    cm_path = os.path.join(args.save_dir, f"{args.model}_{args.dataset}_confusion_full.png")
    plot_confusion(cm, class_names, cm_path, f"Confusion (validation) - {args.dataset}")
    print(f"Saved confusion matrix: {cm_path}")

    # Error confidences
    errors_idx = preds_f != y_val_f
    if proba_f is not None:
        confidences = []
        for i, err in enumerate(errors_idx):
            if err:
                conf = compute_confidence(proba_f[i])
                confidences.append(conf)
    else:
        confidences = []

    if confidences:
        plt.hist(confidences, bins=20)
        plt.title("Error Sample Confidence Distribution")
        plt.xlabel("confidence")
        plt.ylabel("count")
        hist_path = os.path.join(args.save_dir, f"{args.model}_{args.dataset}_error_confidence_hist.png")
        plt.tight_layout()
        plt.savefig(hist_path, dpi=200)
        if args.show:
            plt.show()
        plt.close()
        print(f"Saved confidence histogram: {hist_path}")
    else:
        print("No confidence data for errors (predict_proba not available or no errors).")


if __name__ == "__main__":
    main()
