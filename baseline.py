"""
Baseline-only runner using the pretrained TabPFN model.

Provides a simple CLI to train and/or predict with the baseline and
optionally generate a combined submission file.
"""

import os
import sys
import argparse

# Add src directory to Python path
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from train import train_all_datasets
from predict import predict_all_datasets


def main():
    parser = argparse.ArgumentParser(
        description="Run the TabPFN baseline (train/predict/combined submission)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Skip cross-validation (recommended for speed)",
    )
    parser.add_argument(
        "--train-only",
        action="store_true",
        help="Only train baseline models",
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
        default="baseline_",
        help="Prefix for per-dataset submission files",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )

    args = parser.parse_args()
    verbose = not args.quiet
    submission_prefix = args.submission_prefix or "baseline_"
    save_individual = (not args.no_save) and (not args.combined_only)
    save_combined = (not args.no_combined) or args.combined_only

    # Train only
    if args.train_only:
        train_all_datasets(
            model_type="tabpfn",
            use_cv=not args.no_cv,
            save_models=True,
            verbose=verbose,
        )
        return

    # Predict only
    if args.predict_only:
        predict_all_datasets(
            model_type="tabpfn",
            save_submissions=save_individual,
            save_individual=save_individual,
            save_combined=save_combined,
            combined_filename=args.combined_filename,
            submission_prefix=submission_prefix,
            verbose=verbose,
        )
        return

    # Full pipeline
    train_all_datasets(
        model_type="tabpfn",
        use_cv=not args.no_cv,
        save_models=True,
        verbose=verbose,
    )
    predict_all_datasets(
        model_type="tabpfn",
        save_submissions=save_individual,
        save_individual=save_individual,
        save_combined=save_combined,
        combined_filename=args.combined_filename,
        submission_prefix=submission_prefix,
        verbose=verbose,
    )


if __name__ == "__main__":
    main()
