"""
Prediction script
Generate test-set predictions and create submission files (per dataset).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import json
import warnings

warnings.filterwarnings("ignore")

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import (
    load_unified_train_data,
    load_unified_test_data,
    map_global_to_dataset_labels,
    DATASET_ID_MAP,
    LABEL_OFFSET_MAP,
    DATASET_CLASS_COUNTS,
)

# Directory configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
SUBMISSION_DIR = os.path.join(BASE_DIR, "submissions")
os.makedirs(SUBMISSION_DIR, exist_ok=True)


def _get_unified_metadata_path(model_type: str) -> str:
    return os.path.join(MODEL_DIR, f"unified_{model_type}_metadata.json")


def load_unified_model(model_type="lgbm"):
    """Load a single unified model trained across all datasets."""
    # For LightGBM, prefer the booster text model to avoid pickle segfaults.
    if model_type == "lgbm":
        txt_path = os.path.join(MODEL_DIR, f"unified_{model_type}_model.txt")
        if not os.path.exists(txt_path):
            raise FileNotFoundError(
                f"Unified LightGBM text model not found: {txt_path}\n"
                f"Please re-train unified model to generate it: "
                f"python src/train.py --dataset unified --model lgbm"
            )

        import lightgbm as lgb

        booster = lgb.Booster(model_file=txt_path)

        class _LGBMBoosterWrapper:
            def __init__(self, booster_obj):
                self.booster = booster_obj
                self.name = "LightGBM-Booster"

            def predict(self, X):
                pred = self.booster.predict(X)
                if pred.ndim == 1:
                    # binary style output
                    return (pred > 0.5).astype(int)
                return pred.argmax(axis=1)

            def predict_proba(self, X):
                pred = self.booster.predict(X)
                if pred.ndim == 1:
                    return np.vstack([1 - pred, pred]).T
                return pred

        return _LGBMBoosterWrapper(booster)

    model_path = os.path.join(MODEL_DIR, f"unified_{model_type}_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Unified model not found: {model_path}\n"
            f"Please train it first using: python train.py --dataset unified --model {model_type}"
        )
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    return model


def load_unified_metadata(model_type="lgbm"):
    """
    Load unified feature/label metadata saved during training.
    Falls back to recomputing from train data if the metadata file is missing.
    """
    metadata_path = _get_unified_metadata_path(model_type)
    if os.path.exists(metadata_path):
        with open(metadata_path, "r") as f:
            return json.load(f)

    # Fallback: recompute union features (may be slower).
    _, _, union_features = load_unified_train_data()
    return {
        "union_features": union_features,
        "dataset_id_map": DATASET_ID_MAP,
        "label_offset_map": LABEL_OFFSET_MAP,
        "class_counts": DATASET_CLASS_COUNTS,
    }


def predict_unified_model(
    model_type="lgbm",
    save_submissions=True,
    save_combined=True,
    combined_filename="combined_submission.csv",
    submission_prefix=None,
    verbose=True,
):
    """
    Run prediction using a SINGLE unified model trained across all datasets.
    """
    if submission_prefix is None:
        submission_prefix = f"{model_type}_"

    model = load_unified_model(model_type)
    metadata = load_unified_metadata(model_type)
    union_features = metadata.get("union_features", [])

    test_parts = load_unified_test_data(union_features)
    submissions = {}

    for dataset_name, part in test_parts.items():
        if verbose:
            print(f"Predicting {dataset_name.upper()} with unified model")

        preds_global = model.predict(part["X"])
        preds_local = map_global_to_dataset_labels(dataset_name, preds_global)

        if dataset_name == "covtype":
            preds_local = preds_local.astype(int) + 1
        else:
            preds_local = preds_local.astype(int)

        submission = pd.DataFrame(
            {"ID": part["ids"], "Prediction": preds_local}
        )

        if save_submissions:
            filename = f"{submission_prefix}{dataset_name}_test_submission.csv"
            submission_path = os.path.join(SUBMISSION_DIR, filename)
            submission.to_csv(submission_path, index=False)
            if verbose:
                print(f"Saved: {submission_path}")

        submissions[dataset_name] = submission

    if save_combined and submissions:
        combined_path = combine_submissions(
            submissions,
            filename=combined_filename,
            verbose=verbose,
        )
        if verbose:
            print(f"Combined submission saved to: {combined_path}")

    return submissions


def combine_submissions(submissions_dict, filename="combined_submission.csv", verbose=True):
    """
    Combine per-dataset submissions into a single file (ID, Prediction).

    Rows are concatenated then sorted by ID to keep deterministic ordering.
    """
    frames = []
    for name, df in submissions_dict.items():
        if not {"ID", "Prediction"}.issubset(df.columns):
            if verbose:
                print(f"Skipping {name}: missing required columns")
            continue
        frames.append(df[["ID", "Prediction"]])

    if not frames:
        raise ValueError("No valid submissions to combine")

    combined = pd.concat(frames, axis=0)
    combined = combined.sort_values("ID")

    output_path = os.path.join(SUBMISSION_DIR, filename)
    combined.to_csv(output_path, index=False)
    return output_path


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions for tabular datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="unified",
        choices=["unified"],
        help="Dataset to predict (only unified mode is supported)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lgbm",
        choices=["rf", "lgbm", "xgb", "lr", "mlp", "ensemble", "baseline"],
        help="Model type",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Comma-separated list of models to predict (e.g., 'baseline,lgbm'). Overrides --model.",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save individual submission files (combined can still be saved)",
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Do not save the combined submission file",
    )
    parser.add_argument(
        "--combined-only",
        action="store_true",
        help="Only save the combined submission (skip per-dataset files)",
    )
    parser.add_argument(
        "--combined-filename",
        type=str,
        default="combined_submission.csv",
        help="Filename for the combined submission (relative to submissions/)",
    )
    parser.add_argument(
        "--submission-prefix",
        type=str,
        default=None,
        help="Prefix for saved submission files (defaults to 'baseline_' when model=baseline)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )

    args = parser.parse_args()
    verbose = not args.quiet
    save_individual = (not args.no_save) and (not args.combined_only)
    save_combined = (not args.no_combined) or args.combined_only
    model_list = (
        [m.strip() for m in args.models.split(",")] if args.models else [args.model]
    )

    for model_name in model_list:
        # Derive per-model prefixes/combined filenames when running multiple models.
        prefix = args.submission_prefix
        if prefix is None:
            prefix = "baseline_" if model_name == "baseline" else f"{model_name}_"
        combined_name = args.combined_filename
        if len(model_list) > 1 and args.combined_filename == "combined_submission.csv":
            combined_name = f"{model_name}_combined_submission.csv"

        predict_unified_model(
            model_type=model_name,
            save_submissions=save_individual,
            save_combined=save_combined,
            combined_filename=combined_name,
            submission_prefix=prefix,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()
