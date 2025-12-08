"""
Main entry script
Run the full training + prediction pipeline with a single command.
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

from train import train_all_datasets, train_single_dataset
from predict import predict_all_datasets
from data_utils import get_data_loader


def run_full_pipeline(
    model_type="lgbm",
    use_cv=True,
    submission_prefix=None,
    save_individual=True,
    save_combined=True,
    combined_filename="combined_submission.csv",
    verbose=True,
):
    """
    Run the full training and prediction pipeline.

    Steps:
      1. Basic data overview
      2. Train models for all three datasets
      3. Generate test-set predictions (separate CSV per dataset)
    """
    print("  TABULAR ML BENCHMARK")
    print(f"\nModel: {model_type}")
    print(f"Cross-validation: {use_cv}")

    # Step 1: Data overview
    print("STEP 1: Data Overview")

    for name in ["covtype", "heloc", "higgs"]:
        loader = get_data_loader(name)
        X, y, cols = loader.load_train_data()
        X_test, _ = loader.load_test_data()
        print(f"\n{name.upper()}:")
        print(f"  Train: {X.shape[0]:,} samples, {X.shape[1]} features")
        print(f"  Test:  {X_test.shape[0]:,} samples")
        print(f"  Classes: {len(set(y))}")

    # Step 2: Training
    print("\n STEP 2: Training Models")

    results = train_all_datasets(
        model_type=model_type,
        use_cv=use_cv,
        save_models=True,
        verbose=verbose,
    )

    # Step 3: Prediction
    print("\n STEP 3: Generating Predictions")

    predict_all_datasets(
        model_type=model_type,
        save_submissions=save_individual,
        save_individual=save_individual,
        submission_prefix=submission_prefix,
        save_combined=save_combined,
        combined_filename=combined_filename,
        verbose=verbose,
    )

    # Final summary

    print("\nResults Summary:")
    total_acc = 0.0
    for name, result in results.items():
        acc = result["val_accuracy"]
        total_acc += acc
        print(f"  {name.upper()}: {acc:.4f}")

    print(f"\n  Average Validation Accuracy: {total_acc / 3:.4f}")
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Tabular ML Benchmark - Main Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run.py                         # Run full pipeline with default settings
  python run.py --model rf              # Use Random Forest
  python run.py --model ensemble        # Use ensemble of multiple models
  python run.py --train-only            # Only train, don't predict
  python run.py --predict-only          # Only predict (requires trained models)
        """,
    )

    parser.add_argument(
        "--model",
        type=str,
        default="lgbm",
        choices=["rf", "lgbm", "xgb", "lr", "mlp", "ensemble", "baseline"],
        help="Model type (default: lgbm)",
    )
    parser.add_argument(
        "--compare-with",
        type=str,
        choices=["rf", "lgbm", "xgb", "lr", "mlp", "ensemble", "baseline"],
        help="Train baseline and this model side-by-side (validation accuracy only; skips prediction)",
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip cross-validation",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train models",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Only generate predictions (requires trained models)",
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
        help="Prefix for saved submission files (defaults to 'baseline_' when model=baseline)",
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
        "--quiet",
        action="store_true",
        help="Minimal output",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    submission_prefix = args.submission_prefix or (
        "baseline_" if args.model == "baseline" else ""
    )
    save_individual = (not args.no_save) and (not args.combined_only)
    save_combined = (not args.no_combined) or args.combined_only

    # Comparison mode: always train-only and print a table.
    if args.compare_with:
        models = ["baseline", args.compare_with]
        comparison_results = {}
        for m in models:
            print(f"\n=== Training {m} ===")
            comparison_results[m] = train_all_datasets(
                model_type=m,
                use_cv=not args.no_cv,
                save_models=True,
                verbose=verbose,
            )

        print("\nValidation Accuracy Comparison:")
        datasets = ["covtype", "heloc", "higgs"]
        header = "Dataset  " + "  ".join([f"{m:>12}" for m in models])
        print(header)
        print("-" * len(header))
        for ds in datasets:
            row = [f"{ds:<8}"]
            for m in models:
                acc = comparison_results[m][ds]["val_accuracy"]
                row.append(f"{acc:12.4f}")
            print("  ".join(row))
        return

    if args.train_only:
        # Train only
        train_all_datasets(
            model_type=args.model,
            use_cv=not args.no_cv,
            save_models=True,
            verbose=verbose,
        )
    elif args.predict_only:
        # Predict only (assumes models are already trained)
        predict_all_datasets(
            model_type=args.model,
            save_submissions=save_individual,
            save_individual=save_individual,
            submission_prefix=submission_prefix,
            save_combined=save_combined,
            combined_filename=args.combined_filename,
            verbose=verbose,
        )
    else:
        # Full pipeline: data overview + training + prediction
        run_full_pipeline(
            model_type=args.model,
            use_cv=not args.no_cv,
            submission_prefix=submission_prefix,
            save_individual=save_individual,
            save_combined=save_combined,
            combined_filename=args.combined_filename,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()
