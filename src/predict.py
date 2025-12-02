"""
Prediction script
Generate test-set predictions and create submission files (per dataset).
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
import warnings

warnings.filterwarnings("ignore")

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import get_data_loader

# Directory configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
SUBMISSION_DIR = os.path.join(BASE_DIR, "submissions")
os.makedirs(SUBMISSION_DIR, exist_ok=True)


def load_model(dataset_name, model_type="lgbm"):
    """Load a trained model from disk."""
    model_path = os.path.join(MODEL_DIR, f"{dataset_name}_{model_type}_model.pkl")

    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model not found: {model_path}\n"
            f"Please train the model first using: "
            f"python train.py --dataset {dataset_name} --model {model_type}"
        )

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    return model


def predict_single_dataset(
    dataset_name,
    model_type="lgbm",
    save_submission=True,
    verbose=True,
):
    """
    Run prediction on a single dataset.

    Args
    ----
    dataset_name : str
        Name of the dataset ('covtype', 'heloc', or 'higgs').
    model_type : str
        Model type (e.g. 'lgbm', 'rf', 'xgb', ...).
    save_submission : bool
        Whether to save the submission CSV file.
    verbose : bool
        Whether to print detailed logs.

    Returns
    -------
    submission : pandas.DataFrame
        DataFrame with columns ['ID', 'Prediction'].
    """
    if verbose:
        print(f"Predicting {dataset_name.upper()}")

    # Load model
    model = load_model(dataset_name, model_type)
    if verbose:
        model_name = getattr(model, "name", str(model))
        print(f"Model loaded: {model_name}")

    # Load data – important: load train first to fix feature columns
    loader = get_data_loader(dataset_name)

    # Load training data to determine feature columns (we ignore the actual data)
    _, _, train_feature_cols = loader.load_train_data()

    # Then load test data (will use the same feature columns as training)
    X_test, feature_cols = loader.load_test_data()

    if verbose:
        print(f"Test samples:  {X_test.shape[0]}")
        print(f"Test features: {X_test.shape[1]}")

    # Predict labels
    predictions = model.predict(X_test)

    # Post-process labels if needed (here mostly just casting to int)
    if dataset_name == "covtype":
        # CoverType: labels are 1–7
        predictions = predictions.astype(int)
    elif dataset_name == "heloc":
        # HELOC: 0=Bad, 1=Good (we keep numeric as in sample submission)
        predictions = predictions.astype(int)
    elif dataset_name == "higgs":
        # HIGGS: 0=background, 1=signal
        predictions = predictions.astype(int)

    # Create submission DataFrame
    id_start = loader.id_start
    submission = pd.DataFrame(
        {
            "ID": range(id_start, id_start + len(predictions)),
            "Prediction": predictions,
        }
    )

    if verbose:
        print(f"Predictions shape: {submission.shape}")
        print(f"ID range: {submission['ID'].min()} - {submission['ID'].max()}")
        print(
            "Prediction distribution:\n"
            f"{pd.Series(predictions).value_counts().sort_index()}"
        )

    # Save submission file
    if save_submission:
        submission_path = os.path.join(
            SUBMISSION_DIR, f"{dataset_name}_test_submission.csv"
        )
        submission.to_csv(submission_path, index=False)
        if verbose:
            print(f"Submission saved to: {submission_path}")

    return submission


def predict_all_datasets(
    model_type="lgbm",
    save_submissions=True,
    verbose=True,
):
    """
    Run prediction for all three datasets.

    Returns
    -------
    all_submissions : dict
        Mapping from dataset name to its submission DataFrame.
    """
    all_submissions = {}

    for dataset_name in ["covtype", "heloc", "higgs"]:
        try:
            submission = predict_single_dataset(
                dataset_name=dataset_name,
                model_type=model_type,
                save_submission=save_submissions,
                verbose=verbose,
            )
            all_submissions[dataset_name] = submission
        except FileNotFoundError as e:
            print(f"Warning: {e}")
            continue

    return all_submissions


def main():
    parser = argparse.ArgumentParser(
        description="Generate predictions for tabular datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["covtype", "heloc", "higgs", "all"],
        help="Dataset to predict",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lgbm",
        choices=["rf", "lgbm", "xgb", "lr", "mlp", "ensemble"],
        help="Model type",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not save individual submission files",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Minimal output",
    )

    args = parser.parse_args()
    verbose = not args.quiet

    # Predict for one or all datasets
    if args.dataset == "all":
        predict_all_datasets(
            model_type=args.model,
            save_submissions=not args.no_save,
            verbose=verbose,
        )
    else:
        predict_single_dataset(
            dataset_name=args.dataset,
            model_type=args.model,
            save_submission=not args.no_save,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()
