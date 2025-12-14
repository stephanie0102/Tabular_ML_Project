"""
Unified pipeline entrypoint: train + predict with a single model across all datasets.
"""

import os
import sys
import argparse

# Mitigate OpenMP library conflicts (libomp vs libiomp) common on macOS/conda.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from train import train_unified_model
from predict import predict_unified_model


def run_full_pipeline(
    model_type: str = "lgbm",
    use_cv: bool = True,
    save_individual: bool = True,
    save_combined: bool = True,
    combined_filename: str = "combined_submission.csv",
    submission_prefix: str = None,
    verbose: bool = True,
):
    """
    Train unified model and generate submissions.
    """
    print("TABULAR ML BENCHMARK (Unified)")
    print(f"Model: {model_type}")
    print(f"Cross-validation: {use_cv}")

    result = train_unified_model(
        model_type=model_type,
        use_cv=use_cv,
        save_model=True,
        verbose=verbose,
    )

    per_ds = result.get("per_dataset_accuracy", {})
    print("\nValidation (unified hold-out):")
    print(f"  Overall: {result['val_accuracy']:.4f}")
    for name, acc in per_ds.items():
        if acc is None:
            continue
        print(f"  {name.upper()}: {acc:.4f}")

    print("\nGenerating predictions...")
    predict_unified_model(
        model_type=model_type,
        save_submissions=save_individual,
        save_combined=save_combined,
        combined_filename=combined_filename,
        submission_prefix=submission_prefix or f"{model_type}_",
        verbose=verbose,
    )


def main():
    parser = argparse.ArgumentParser(description="Unified training + prediction")
    parser.add_argument(
        "--model",
        type=str,
        default="lgbm",
        choices=["rf", "lgbm", "xgb", "lr", "mlp", "ensemble", "baseline"],
        help="Model type (default: lgbm)",
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip cross-validation",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train, do not predict",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Only predict (requires trained model)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save per-dataset submission files",
    )
    parser.add_argument(
        "--submission-prefix",
        type=str,
        default=None,
        help="Prefix for saved submission files",
    )
    parser.add_argument(
        "--no-combined",
        action="store_true",
        help="Do not save combined submission file",
    )
    parser.add_argument(
        "--combined-filename",
        type=str,
        default="combined_submission.csv",
        help="Filename for combined submission",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )

    args = parser.parse_args()
    verbose = not args.quiet
    save_individual = not args.no_save
    save_combined = not args.no_combined
    submission_prefix = args.submission_prefix or f"{args.model}_"

    if args.train_only:
        res = train_unified_model(
            model_type=args.model,
            use_cv=not args.no_cv,
            save_model=True,
            verbose=verbose,
        )
        per_ds = res.get("per_dataset_accuracy", {})
        print(f"Overall val accuracy: {res['val_accuracy']:.4f}")
        for name, acc in per_ds.items():
            if acc is not None:
                print(f"  {name.upper()}: {acc:.4f}")
        return

    if args.predict_only:
        predict_unified_model(
            model_type=args.model,
            save_submissions=save_individual,
            save_combined=save_combined,
            combined_filename=args.combined_filename,
            submission_prefix=submission_prefix,
            verbose=verbose,
        )
        return

    run_full_pipeline(
        model_type=args.model,
        use_cv=not args.no_cv,
        save_individual=save_individual,
        save_combined=save_combined,
        combined_filename=args.combined_filename,
        submission_prefix=submission_prefix,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
