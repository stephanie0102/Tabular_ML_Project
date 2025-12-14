"""
TabPFN baseline runner (per-dataset training).

TabPFN supports up to 10 classes, so we train three separate models
for COVTYPE (7 classes), HELOC (2), and HIGGS (2), then sum training time.
"""

import os
import sys
import argparse
import time
import pickle
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from data_utils import get_data_loader
from models_tabular import TabPFNModel, HAS_TABPFN
from predict import combine_submissions

# Default to TabPFN v2.5 unless user overrides.
os.environ.setdefault("TABPFN_MODEL_VERSION", "v2.5")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "models")
SUBMISSION_DIR = os.path.join(BASE_DIR, "submissions")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(SUBMISSION_DIR, exist_ok=True)


def train_tabpfn_dataset(dataset_name: str, use_cv: bool = True, verbose: bool = True):
    loader = get_data_loader(dataset_name)
    X, y, _ = loader.load_train_data()

    model = TabPFNModel()

    cv_scores = None
    if use_cv:
        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model.model, X, y, cv=skf, scoring="accuracy")
        if verbose:
            print(f"{dataset_name}: CV mean={cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=0.15, stratify=y, random_state=42
    )

    start = time.time()
    model.fit(X_tr, y_tr)
    train_time = time.time() - start

    val_pred = model.predict(X_val)
    val_acc = accuracy_score(y_val, val_pred)

    if verbose:
        print(f"{dataset_name}: Val accuracy={val_acc:.4f}")
        print(classification_report(y_val, val_pred))
        print(f"Train time: {train_time:.2f}s")

    # Save model
    model_path = os.path.join(MODEL_DIR, f"{dataset_name}_tabpfn_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    if verbose:
        print(f"Saved: {model_path}")

    return {
        "val_accuracy": val_acc,
        "cv_scores": cv_scores.tolist() if cv_scores is not None else None,
        "train_time": train_time,
        "model_path": model_path,
    }


def predict_tabpfn_dataset(dataset_name: str, save: bool = True, prefix: str = "baseline_", verbose: bool = True):
    model_path = os.path.join(MODEL_DIR, f"{dataset_name}_tabpfn_model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}. Train first.")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    loader = get_data_loader(dataset_name)
    # ensure feature ordering
    loader.load_train_data()
    X_test, _ = loader.load_test_data()
    preds = model.predict(X_test)

    # Post-process labels for submissions
    if dataset_name == "covtype":
        preds = preds.astype(int) + 1  # 1-7
    else:
        preds = preds.astype(int)

    ids = np.arange(loader.id_start, loader.id_start + len(preds))
    import pandas as pd

    submission = pd.DataFrame({"ID": ids, "Prediction": preds})
    if save:
        fname = f"{prefix}{dataset_name}_test_submission.csv"
        out_path = os.path.join(SUBMISSION_DIR, fname)
        submission.to_csv(out_path, index=False)
        if verbose:
            print(f"Saved submission: {out_path}")
    return submission


def main():
    parser = argparse.ArgumentParser(
        description="TabPFN baseline (per-dataset training due to class limit)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--no-cv", action="store_true", help="Skip cross-validation")
    parser.add_argument("--train-only", action="store_true", help="Only train models")
    parser.add_argument("--predict-only", action="store_true", help="Only run prediction")
    parser.add_argument("--no-save", action="store_true", help="Do not save per-dataset submissions")
    parser.add_argument("--no-combined", action="store_true", help="Do not save combined submission")
    parser.add_argument("--combined-only", action="store_true", help="Only save combined submission")
    parser.add_argument("--combined-filename", type=str, default="combined_submission.csv")
    parser.add_argument("--submission-prefix", type=str, default="baseline_")
    parser.add_argument("--quiet", action="store_true", help="Minimal output")
    args = parser.parse_args()

    if not HAS_TABPFN:
        print("TabPFN not installed. Install with: pip install tabpfn torch")
        sys.exit(1)

    verbose = not args.quiet
    save_individual = (not args.no_save) and (not args.combined_only)
    save_combined = (not args.no_combined) or args.combined_only
    prefix = args.submission_prefix or "baseline_"

    datasets = ["covtype", "heloc", "higgs"]

    if not args.predict_only:
        total_time = 0.0
        results = {}
        for ds in datasets:
            if verbose:
                print(f"\n=== Training {ds.upper()} (TabPFN) ===")
            res = train_tabpfn_dataset(ds, use_cv=not args.no_cv, verbose=verbose)
            total_time += res["train_time"]
            results[ds] = res
        print(f"\nTotal TabPFN training time (sum of 3 datasets): {total_time:.2f}s")
        print("Validation accuracy per dataset:")
        for ds, res in results.items():
            print(f"  {ds.upper()}: {res['val_accuracy']:.4f}")
        if args.train_only:
            return

    if not args.train_only:
        submissions = {}
        for ds in datasets:
            if verbose:
                print(f"\n=== Predicting {ds.upper()} ===")
            submissions[ds] = predict_tabpfn_dataset(
                ds, save=save_individual, prefix=prefix, verbose=verbose
            )

        if save_combined and submissions:
            combined_path = combine_submissions(
                submissions,
                filename=args.combined_filename,
                verbose=verbose,
            )
            if verbose:
                print(f"Combined submission saved to: {combined_path}")


if __name__ == "__main__":
    main()
