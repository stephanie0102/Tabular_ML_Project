"""
Training script

Supports:
- training a single dataset or all datasets
- cross-validation
- hyperparameter optimization with Optuna
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import pickle
import json
from datetime import datetime
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import warnings

warnings.filterwarnings("ignore")

# Add src directory to Python path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_utils import get_data_loader
from models_tabular import (
    get_model,
    get_default_models,
    get_best_params_per_dataset,
    RandomForestModel,
    LightGBMModel,
    XGBoostModel,
    EnsembleModel,
    HAS_LIGHTGBM,
    HAS_XGBOOST,
    HAS_TABPFN,
)

# Directory configuration
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


def train_single_dataset(
    dataset_name,
    model_type="lgbm",
    use_cv=True,
    cv_folds=5,
    save_model=True,
    verbose=True,
):
    """
    Train a single dataset.

    Args
    ----
    dataset_name : str
        Name of the dataset ('covtype', 'heloc', 'higgs').
    model_type : str
        Model type ('rf', 'lgbm', 'xgb', 'ensemble', ...).
    use_cv : bool
        Whether to run cross-validation.
    cv_folds : int
        Number of cross-validation folds.
    save_model : bool
        Whether to save the trained model to disk.
    verbose : bool
        Whether to print training logs.

    Returns
    -------
    result : dict
        Dictionary containing the model and evaluation metrics.
    """
    if verbose:
        print(f"Training on {dataset_name.upper()} dataset")
        print(f"Model: {model_type}")

    # Load data
    loader = get_data_loader(dataset_name)
    X_train, y_train, feature_cols = loader.load_train_data()

    if verbose:
        print(f"Training samples: {X_train.shape[0]}")
        print(f"Features:        {X_train.shape[1]}")
        print(f"Classes:         {len(np.unique(y_train))}")

    # Optional downsampling for TabPFN baseline to respect 50k pretraining limit.
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

    # Load pre-defined best parameters
    best_params = get_best_params_per_dataset()

    # Build model
    if model_type == "ensemble":
        models = []
        if HAS_LIGHTGBM:
            params = best_params.get(dataset_name, {}).get("lgbm", {})
            models.append(LightGBMModel(**params))
        if HAS_XGBOOST:
            params = best_params.get(dataset_name, {}).get("xgb", {})
            models.append(XGBoostModel(**params))
        params = best_params.get(dataset_name, {}).get("rf", {})
        models.append(RandomForestModel(**params))

        model = EnsembleModel(models, voting="soft")
    else:
        params = best_params.get(dataset_name, {}).get(model_type, {})
        model = get_model(model_type, **params)

    cv_scores = None

    # Cross-validation (for base models, not ensemble wrapper)
    if use_cv and not isinstance(model, EnsembleModel):
        if verbose:
            print(f"\nPerforming {cv_folds}-fold cross-validation...")

        skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
        cv_scores = cross_val_score(
            model.model, X_train, y_train, cv=skf, scoring="accuracy"
        )

        if verbose:
            print(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
            print(f"Fold scores: {[f'{s:.4f}' for s in cv_scores]}")

    # Train/validation split for final reporting
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.15, random_state=42, stratify=y_train
    )

    # Train final model on full training data
    if verbose:
        print("\nTraining final model on full training data...")

    model.fit(X_train, y_train)

    # Evaluate on validation split (for reporting only)
    val_pred = model.predict(X_val)
    val_accuracy = accuracy_score(y_val, val_pred)

    if verbose:
        print(f"Validation Accuracy: {val_accuracy:.4f}")
        print("\nClassification Report:")
        print(classification_report(y_val, val_pred))

    # Save model
    if save_model:
        model_path = os.path.join(MODEL_DIR, f"{dataset_name}_{model_type}_model.pkl")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)
        if verbose:
            print(f"Model saved to: {model_path}")

    return {
        "model": model,
        "dataset": dataset_name,
        "val_accuracy": val_accuracy,
        "cv_scores": cv_scores.tolist() if cv_scores is not None else None,
    }


def train_all_datasets(
    model_type="lgbm",
    use_cv=True,
    save_models=True,
    verbose=True,
):
    """
    Train the same model type on all three datasets.

    Returns
    -------
    results : dict
        Mapping from dataset name to result dict (same structure as train_single_dataset).
    """
    results = {}

    for dataset_name in ["covtype", "heloc", "higgs"]:
        result = train_single_dataset(
            dataset_name=dataset_name,
            model_type=model_type,
            use_cv=use_cv,
            save_model=save_models,
            verbose=verbose,
        )
        results[dataset_name] = result

    # Print summary
    if verbose:
        print("TRAINING SUMMARY")
        total_acc = 0.0
        for name, result in results.items():
            print(f"{name.upper()}: Validation Accuracy = {result['val_accuracy']:.4f}")
            total_acc += result["val_accuracy"]
        print(f"\nAverage Accuracy: {total_acc / 3:.4f}")

    return results


def hyperparameter_search(
    dataset_name,
    model_type="lgbm",
    n_trials=50,
    verbose=True,
):
    """
    Run hyperparameter search with Optuna for a given dataset and model type.
    """
    try:
        import optuna
        from optuna.samplers import TPESampler
    except ImportError:
        print("Optuna not installed. Run: pip install optuna")
        return None

    if verbose:
        print(f"\nHyperparameter search for {dataset_name} using {model_type}")

    # Load data
    loader = get_data_loader(dataset_name)
    X_train, y_train, _ = loader.load_train_data()

    # Train/validation split for objective evaluation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    def objective(trial):
        if model_type == "lgbm" and HAS_LIGHTGBM:
            import lightgbm as lgb

            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "num_leaves": trial.suggest_int("num_leaves", 16, 128),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.5, 1.0
                ),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
                "random_state": 42,
                "n_jobs": -1,
                "verbose": -1,
            }
            model = lgb.LGBMClassifier(**params)

        elif model_type == "xgb" and HAS_XGBOOST:
            import xgboost as xgb

            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "learning_rate": trial.suggest_float(
                    "learning_rate", 0.01, 0.3, log=True
                ),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "min_child_weight": trial.suggest_int(
                    "min_child_weight", 1, 20
                ),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float(
                    "colsample_bytree", 0.5, 1.0
                ),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 10, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 1e-4, 10, log=True),
                "random_state": 42,
                "n_jobs": -1,
                "verbosity": 0,
                "use_label_encoder": False,
                "eval_metric": "logloss",
            }
            model = xgb.XGBClassifier(**params)

        else:
            # Fall back to RandomForest search
            from sklearn.ensemble import RandomForestClassifier

            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 5, 30),
                "min_samples_split": trial.suggest_int(
                    "min_samples_split", 2, 20
                ),
                "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 10),
                "random_state": 42,
                "n_jobs": -1,
            }
            model = RandomForestClassifier(**params)

        model.fit(X_tr, y_tr)
        val_pred = model.predict(X_val)
        return accuracy_score(y_val, val_pred)

    # Run optimization
    sampler = TPESampler(seed=42)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, show_progress_bar=verbose)

    if verbose:
        print(f"\nBest trial accuracy: {study.best_trial.value:.4f}")
        print(f"Best parameters: {study.best_trial.params}")

    return study.best_trial.params


def main():
    parser = argparse.ArgumentParser(
        description="Train models for tabular datasets"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="all",
        choices=["covtype", "heloc", "higgs", "all"],
        help="Dataset to train on",
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
        help="Perform hyperparameter search with Optuna",
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

    # Hyperparameter search mode
    if args.hypersearch:
        if args.dataset == "all":
            for ds in ["covtype", "heloc", "higgs"]:
                best_params = hyperparameter_search(
                    ds, args.model, args.trials, verbose
                )
                if best_params:
                    params_path = os.path.join(
                        MODEL_DIR, f"{ds}_{args.model}_best_params.json"
                    )
                    with open(params_path, "w") as f:
                        json.dump(best_params, f, indent=2)
        else:
            best_params = hyperparameter_search(
                args.dataset, args.model, args.trials, verbose
            )
            if best_params:
                params_path = os.path.join(
                    MODEL_DIR, f"{args.dataset}_{args.model}_best_params.json"
                )
                with open(params_path, "w") as f:
                    json.dump(best_params, f, indent=2)
        return

    # Normal training mode
    if args.dataset == "all":
        _ = train_all_datasets(
            model_type=args.model,
            use_cv=not args.no_cv,
            save_models=not args.no_save,
            verbose=verbose,
        )
    else:
        _ = train_single_dataset(
            dataset_name=args.dataset,
            model_type=args.model,
            use_cv=not args.no_cv,
            cv_folds=args.cv,
            save_model=not args.no_save,
            verbose=verbose,
        )


if __name__ == "__main__":
    main()
