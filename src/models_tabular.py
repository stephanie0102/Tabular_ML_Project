"""
Tabular model definitions.

Contains multiple classification models:
Random Forest, LightGBM, XGBoost, Logistic Regression, MLP, etc.
"""

import os
import numpy as np
import torch
from typing import Optional
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Try importing advanced models
try:
    import lightgbm as lgb
    HAS_LIGHTGBM = True
except ImportError:
    HAS_LIGHTGBM = False
    print("Warning: LightGBM not installed. Run: pip install lightgbm")

try:
    import xgboost as xgb
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False
    print("Warning: XGBoost not installed. Run: pip install xgboost")
    
# TabPFN (pretrained HuggingFace weights for tabular data)
try:
    from tabpfn import TabPFNClassifier
    HAS_TABPFN = True
except ImportError:
    HAS_TABPFN = False
    print("Warning: TabPFN not installed. Run: pip install tabpfn torch")


class BaseModel:
    """Base wrapper class for models."""

    def __init__(self, name, model):
        self.name = name
        self.model = model
        self.is_fitted = False

    def fit(self, X, y):
        """Fit the underlying model."""
        self.model.fit(X, y)
        self.is_fitted = True
        return self

    def predict(self, X):
        """Predict class labels."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
        return self.model.predict(X)

    def predict_proba(self, X):
        """Predict class probabilities if supported."""
        if not self.is_fitted:
            raise ValueError("Model not fitted yet!")
        if hasattr(self.model, "predict_proba"):
            return self.model.predict_proba(X)
        return None

    def cross_validate(self, X, y, cv=5):
        """Run stratified k-fold cross validation and return mean/std accuracy."""
        skf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        scores = cross_val_score(self.model, X, y, cv=skf, scoring="accuracy")
        return scores.mean(), scores.std()

class LogisticRegressionModel(BaseModel):
    """
    Logistic Regression with automatic standardization.
    Good as a 'simple' baseline.
    """

    def __init__(
        self,
        C=1.0,
        max_iter=1000,
        solver='lbfgs',
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    ):
        # LR needs scaled data, so we wrap it in a pipeline
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(
                C=C,
                max_iter=max_iter,
                solver=solver,
                class_weight=class_weight,
                random_state=random_state,
                n_jobs=n_jobs if solver != 'liblinear' else 1
            )
        )
        super().__init__("LogisticRegression", model)

class RandomForestModel(BaseModel):
    """Random Forest classifier."""

    def __init__(
        self,
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    ):
        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        super().__init__("RandomForest", model)


class LightGBMModel(BaseModel):
    """LightGBM classifier."""

    def __init__(
        self,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=-1,
        num_leaves=31,
        min_child_samples=20,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        class_weight="balanced",
        n_jobs=1,
        random_state=42,
    ):
        if not HAS_LIGHTGBM:
            raise ImportError("LightGBM not installed!")

        model = lgb.LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            class_weight=class_weight,
            n_jobs=n_jobs,
            random_state=random_state,
            verbose=-1,
        )
        super().__init__("LightGBM", model)


class XGBoostModel(BaseModel):
    """XGBoost classifier."""

    def __init__(
        self,
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=1,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,
        reg_lambda=0.1,
        n_jobs=-1,
        tree_method="hist",
        objective=None,
        num_class=None,
        random_state=42,
    ):
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed!")

        eval_metric = "mlogloss" if (objective and objective.startswith("multi")) else "logloss"

        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_child_weight=min_child_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            n_jobs=n_jobs,
            random_state=random_state,
            verbosity=0,
            use_label_encoder=False,
            eval_metric=eval_metric,
            tree_method=tree_method,
            objective=objective,
            num_class=num_class,
        )
        super().__init__("XGBoost", model)

class TabPFNModel(BaseModel):
   
    def __init__(
        self,
        device: str = "auto",
        n_configurations: int = 32,
        n_estimators: Optional[int] = None,
        model_path: Optional[str] = None,
        model_version: Optional[str] = None,
        ignore_pretraining_limits: bool = True,
    ):
        if not HAS_TABPFN:
            raise ImportError(
                "TabPFN not installed. Install with: pip install tabpfn torch"
            )
        # TabPFN renamed N_ensemble_configurations -> n_estimators; keep old
        # argument for compatibility and map it to the new name.
        env_ensembles = os.environ.get("TABPFN_N_ESTIMATORS")
        if env_ensembles:
            try:
                n_estimators = int(env_ensembles)
            except ValueError:
                pass

        ensemble_size = n_estimators if n_estimators is not None else n_configurations

        # Respect an explicit path, otherwise let TabPFN pick based on the
        # configured model_version (defaults to v2 to avoid gated downloads).
        path_env = os.environ.get("TABPFN_MODEL_PATH")
        resolved_model_path = model_path or path_env

        if resolved_model_path and resolved_model_path.lower() in {"v2", "v2.5"}:
            resolved_model_path = None

        if not resolved_model_path:
            resolved_model_path = "auto"

        version_choice = model_version or os.environ.get("TABPFN_MODEL_VERSION") or "v2"

        try:
            from tabpfn.settings import settings
            from tabpfn.constants import ModelVersion

            settings.tabpfn.model_version = ModelVersion(version_choice)
        except Exception:
            # If anything fails, fall back to whatever TabPFN chooses internally.
            pass

        # default "auto" lets TabPFN pick.
        env_device = os.environ.get("TABPFN_DEVICE")
        if env_device:
            device = env_device
        model = TabPFNClassifier(
            device=device,
            n_estimators=ensemble_size,
            model_path=resolved_model_path,
            ignore_pretraining_limits=ignore_pretraining_limits,
        )

        super().__init__("TabPFN-Baseline", model)

class EnsembleModel:
    """
    Simple ensemble model combining predictions from multiple models.
    Supports hard and soft voting.
    """

    def __init__(self, models, voting="soft", weights=None):
        """
        Args:
            models: list of BaseModel instances
            voting: 'hard' or 'soft'
            weights: list of weights for each model (same length as models)
        """
        self.models = models
        self.voting = voting
        self.weights = weights if weights else [1] * len(models)
        self.name = "Ensemble"
        self.is_fitted = False

    def fit(self, X, y):
        """Fit all base models."""
        for model in self.models:
            print(f"Training {model.name}...")
            model.fit(X, y)
        self.is_fitted = True
        return self

    def _predict_hard(self, X):
        """Internal helper for hard voting."""
        predictions = np.array([model.predict(X) for model in self.models])
        final_pred = []
        for i in range(predictions.shape[1]):
            votes = {}
            for j, model in enumerate(self.models):
                pred = predictions[j, i]
                votes[pred] = votes.get(pred, 0) + self.weights[j]
            final_pred.append(max(votes, key=votes.get))
        return np.array(final_pred)

    def predict(self, X):
        """Ensemble prediction."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet!")

        if self.voting == "hard":
            # Hard voting: weighted majority vote
            return self._predict_hard(X)
        else:
            # Soft voting: average (weighted) probabilities
            probas = []
            for i, model in enumerate(self.models):
                proba = model.predict_proba(X)
                if proba is not None:
                    probas.append(proba * self.weights[i])

            if probas:
                avg_proba = np.sum(probas, axis=0) / sum(self.weights)
                return np.argmax(avg_proba, axis=1)
            else:
                # Fallback to hard voting if no probabilities are available
                return self._predict_hard(X)

    def predict_proba(self, X):
        """Predict ensemble probabilities."""
        if not self.is_fitted:
            raise ValueError("Ensemble not fitted yet!")

        probas = []
        for i, model in enumerate(self.models):
            proba = model.predict_proba(X)
            if proba is not None:
                probas.append(proba * self.weights[i])

        if probas:
            return np.sum(probas, axis=0) / sum(self.weights)
        return None

def get_model(model_name, **kwargs):
    """Return a model instance given a short name."""
    models = {
        "rf": RandomForestModel,
        "random_forest": RandomForestModel,
        "lgbm": LightGBMModel,
        "lightgbm": LightGBMModel,
        "xgb": XGBoostModel,
        "xgboost": XGBoostModel,
        "baseline": TabPFNModel,
        "tabpfn": TabPFNModel,
        "baseline": TabPFNModel,
        "tabpfn": TabPFNModel,
        "logistic_regression": LogisticRegressionModel,
    }

    model_name = model_name.lower()
    if model_name not in models:
        raise ValueError(f"Unknown model: {model_name}. Choose from {list(models.keys())}")

    return models[model_name](**kwargs)


def get_default_models():
    """Return a list of default baseline models (for quick benchmarking)."""
    models = [RandomForestModel()]

    if HAS_LIGHTGBM:
        models.append(LightGBMModel())

    if HAS_XGBOOST:
        models.append(XGBoostModel())

    return models


def get_best_params_per_dataset():
    """
    Return a dict of tuned hyperparameters per dataset.

    These can be used as strong default configurations for each dataset.
    """
    params = {
        "covtype": {
            "lgbm": {
                "n_estimators": 800,
                "learning_rate": 0.03,
                "num_leaves": 63,
                "max_depth": 12,
                "min_child_samples": 10,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
                "reg_alpha": 0.05,
                "reg_lambda": 0.05,
            },
            "xgb": {
                "n_estimators": 800,
                "learning_rate": 0.03,
                "max_depth": 10,
                "min_child_weight": 3,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            "rf": {
                "n_estimators": 500,
                "max_depth": 30,
                "min_samples_split": 5,
                "min_samples_leaf": 2,
            },
            "lr": {"C": 0.1, "max_iter": 500},
        },
        "heloc": {
            "lgbm": {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "num_leaves": 31,
                "max_depth": 8,
                "min_child_samples": 20,
                "subsample": 0.8,
                "colsample_bytree": 0.8,
            },
            "xgb": {
                "n_estimators": 500,
                "learning_rate": 0.05,
                "max_depth": 6,
                "min_child_weight": 5,
            },
            "rf": {
                "n_estimators": 300,
                "max_depth": 15,
                
            },
            "lr": {"C": 0.01},
        },
        "higgs": {
            "lgbm": {
                "n_estimators": 1000,
                "learning_rate": 0.02,
                "num_leaves": 127,
                "max_depth": 15,
                "min_child_samples": 50,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
            },
            "xgb": {
                "n_estimators": 800,
                "learning_rate": 0.02,
                "max_depth": 12,
                "min_child_weight": 10,
                "subsample": 0.7,
                "colsample_bytree": 0.7,
            },
            "rf": {
                "n_estimators": 300,
                "max_depth": 20,
                "min_samples_split": 10,
            },
            "lr": {"C": 0.001},
        },
    }
    return params


if __name__ == "__main__":
    # Simple smoke test for the models
    from sklearn.datasets import make_classification

    X, y = make_classification(
        n_samples=1000,
        n_features=20,
        n_classes=3,
        n_informative=10,
        random_state=42,
    )

    print("Testing models...")
    for model_name in ["rf", "lr"]:
        model = get_model(model_name)
        mean_score, std_score = model.cross_validate(X, y, cv=3)
        print(f"{model.name}: {mean_score:.4f} (+/- {std_score:.4f})")

    if HAS_LIGHTGBM:
        model = get_model("lgbm")
        mean_score, std_score = model.cross_validate(X, y, cv=3)
        print(f"{model.name}: {mean_score:.4f} (+/- {std_score:.4f})")
