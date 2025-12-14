"""
Unified training script (single model for all datasets).
"""

import os
import sys
import argparse
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report
import warnings
import json

warnings.filterwarnings("ignore")

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import (
    load_unified_train_data,
    DATASET_ID_MAP,
    LABEL_OFFSET_MAP,
    DATASET_CLASS_COUNTS,
)
from models_tabular import (
    get_model,
    get_default_models,
    LightGBMModel,
    EnsembleModel,
    HAS_LIGHTGBM,
    HAS_XGBOOST,
    HAS_TABPFN,
)

# Directory configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def _get_unified_metadata_path(model_type: str) -> str:
    return os.path.join(MODEL_DIR, f"unified_{model_type}_metadata.json")


def train_unified_model(
    model_type: str = "lgbm",
    use_cv: bool = True,
    cv_folds: int = 5,
    save_model: bool = True,
    verbose: bool = True,
):
    """
    Train a SINGLE model across all datasets (covtype/heloc/higgs) with a unified
    feature space and unified label space.
    """
    if verbose:
        print("Training unified model across COVTYPE + HELOC + HIGGS")
        print(f"Model: {model_type}")

    X_train, y_train, union_features = load_unified_train_data()

    if verbose:
        print(f"Unified training samples: {X_train.shape[0]}")
        print(f"Unified features:        {X_train.shape[1]} (includes dataset_id)")
        print(f"Unified classes:         {len(np.unique(y_train))}")

    # Optional downsampling for TabPFN baseline to respect pretraining limits.
    if model_type.lower() in {"baseline", "tabpfn"}:
        max_samples = int(os.environ.get("TABPFN_SAMPLE_MAX", "50000"))
        if X_train.shape[0] > max_samples:
            if verbose:
                print(
                    f"Downsampling to {max_samples} samples for TabPFN "
                    f"(from {X_train.shape[0]})"
                )
            X_train, _, y_train, _ = train_test_split(
                X_train,
                y_train,
                train_size=max_samples,
                stratify=y_train,
                random_state=42,
            )

    # Build model (single set of hyperparameters for all datasets).
    n_classes = int(len(np.unique(y_train)))

    if model_type == "ensemble":
        models = get_default_models()
        model = EnsembleModel(models, voting="soft")
    else:
        if model_type == "xgb":
            objective = "multi:softprob" if n_classes > 2 else "binary:logistic"
            model = get_model(
                model_type,
                objective=objective,
                num_class=n_classes if n_classes > 2 else None,
                tree_method="hist",
                n_jobs=1,  # conservative to avoid platform-specific segfaults
            )
        else:
            model = get_model(model_type)

    cv_scores = None

    # Cross-validation (unified stratified k-fold).
    if use_cv and not isinstance(model, EnsembleModel):
        if verbose:
            print(f"\nPerforming {cv_folds}-fold cross-validation on unified data...")
        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model.model, X_train, y_train, cv=skf, scoring="accuracy"
        )
        if verbose:
            print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"Fold scores: {[f'{s:.4f}' for s in cv_scores]}")

    # Hold-out validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    if verbose:
        print("\nTraining model on train split...")

    model.fit(X_tr, y_tr)

    val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)

    val_pred_proba = model.predict_proba(X_val)
    wrong_indices = np.where(y_val != val_pred)[0]

    error_analysis_data = []
    for idx in wrong_indices:
        sample_entry = {
            "sample_index": int(idx),
            "features": X_val[idx].tolist(),
            "true_label": int(y_val[idx]),
            "predicted_label": int(val_pred[idx]),
            "prediction_probabilities": (
                val_pred_proba[idx].tolist() if val_pred_proba is not None else None
            ),
            "confidence": float(np.max(val_pred_proba[idx])) if val_pred_proba is not None else None,
        }
        error_analysis_data.append(sample_entry)

    error_analysis_path = os.path.join(MODEL_DIR, f"unified_{model_type}_error_analysis.json")
    with open(error_analysis_path, "w") as f:
        json.dump(error_analysis_data, f, indent=2, default=str)

    if verbose:
        print(f"Validation Accuracy (unified hold-out): {val_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, val_pred))

    # Optionally retrain on full data before saving for inference.
    if save_model:
        if verbose:
            print("\nRetraining on full unified data for saving...")
        model.fit(X_train, y_train)

        # LightGBM pickles can segfault when unpickling; save booster safely.
        if isinstance(model, LightGBMModel):
            booster = model.model.booster_
            model_path = os.path.join(MODEL_DIR, f"unified_{model_type}_model.txt")
            booster.save_model(model_path)
        else:
            model_path = os.path.join(MODEL_DIR, f"unified_{model_type}_model.pkl")
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
        if verbose:
            print(f"Unified model saved to: {model_path}")

        metadata = {
            "union_features": union_features,
            "dataset_id_map": DATASET_ID_MAP,
            "label_offset_map": LABEL_OFFSET_MAP,
            "class_counts": DATASET_CLASS_COUNTS,
            "n_classes": int(len(np.unique(y_train))),
        }
        metadata_path = _get_unified_metadata_path(model_type)
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        if verbose:
            print(f"Unified metadata saved to: {metadata_path}")

    return {
        "model": model,
        "dataset": "unified",
        "val_accuracy": val_accuracy,
        "cv_scores": cv_scores.tolist() if cv_scores is not None else None,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train unified model for tabular datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="unified",
        choices=["unified"],
        help="Dataset to train on (only unified supported)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lgbm",
        choices=["rf", "lgbm", "xgb", "lr", "mlp", "ensemble", "baseline"],
        help="Model type",
    )
    parser.add_argument(
        "--cv",
        type=int,
        default=5,
        help="Number of cross-validation folds",
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Disable cross-validation",
    )
    parser.add_argument(
        "--hypersearch",
        action="store_true",
        help="(Disabled) Hyperparameter search",
    )
    parser.add_argument(
        "--trials",
        type=int,
        default=50,
        help="Number of hyperparameter search trials",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save trained models",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )

    args = parser.parse_args()

    # Check model dependencies
    if args.model == "lgbm" and not HAS_LIGHTGBM:
        print("LightGBM not available. Using RandomForest instead.")
        args.model = "rf"
    elif args.model == "xgb" and not HAS_XGBOOST:
        print("XGBoost not available. Using RandomForest instead.")
        args.model = "rf"
    elif args.model == "baseline" and not HAS_TABPFN:
        print("TabPFN baseline requires tabpfn/torch. Install with: pip install tabpfn torch")
        sys.exit(1)

    verbose = not args.quiet

    if args.hypersearch:
        print("Hyperparameter search not supported in unified-only mode.")
        return

    _ = train_unified_model(
        model_type=args.model,
        use_cv=not args.no_cv,
        save_model=not args.no_save,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
