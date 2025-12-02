"""
Main entry script
Run the full training + prediction pipeline with a single command.
(no combined submission file)
"""

import os
import sys
import argparse

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from train import train_all_datasets, train_single_dataset
from predict import predict_all_datasets
from data_utils import get_data_loader


def run_full_pipeline(model_type="lgbm", use_cv=True, verbose=True):
    """
    Run the full training and prediction pipeline.

    Steps:
      1. Basic data overview
      2. Train models for all three datasets
      3. Generate test-set predictions (separate CSV per dataset)
    """
    print("  TABULAR ML BENCHMARK - Full Pipeline")
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
        save_submissions=True,
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
        choices=["rf", "lgbm", "xgb", "lr", "mlp", "ensemble"],
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
        help="Only train models",
    )
    parser.add_argument(
        "--predict-only",
        action="store_true",
        help="Only generate predictions (requires trained models)",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )

    args = parser.parse_args()
    verbose = not args.quiet

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
            save_submissions=True,
            verbose=verbose,
        )
    else:
        # Full pipeline: data overview + training + prediction
        run_full_pipeline(
            model_type=args.model,
            use_cv=not args.no_cv,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()
