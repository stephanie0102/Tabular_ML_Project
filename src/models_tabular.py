"""
Tabular model definitions.

Contains multiple classification models:
Random Forest, LightGBM, XGBoost, Logistic Regression, MLP, etc.
"""

import os
import warnings
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
import warnings

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

try:
    from catboost import CatBoostClassifier
    HAS_CATBOOST = True
except ImportError:
    HAS_CATBOOST = False

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
        random_state=42,
    ):
        if not HAS_XGBOOST:
            raise ImportError("XGBoost not installed!")

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
            eval_metric="logloss",
        )
        super().__init__("XGBoost", model)


class LogisticRegressionModel(BaseModel):
    """Logistic Regression classifier."""

    def __init__(
        self,
        C=1.0,
        max_iter=1000,
        class_weight="balanced",
        multi_class="multinomial",
        solver="lbfgs",
        n_jobs=-1,
        random_state=42,
    ):
        model = LogisticRegression(
            C=C,
            max_iter=max_iter,
            class_weight=class_weight,
            multi_class=multi_class,
            solver=solver,
            n_jobs=n_jobs,
            random_state=random_state,
        )
        super().__init__("LogisticRegression", model)


class MLPModel(BaseModel):
    """Multi-layer Perceptron classifier."""

    def __init__(
        self,
        hidden_layer_sizes=(256, 128, 64),
        activation="relu",
        solver="adam",
        alpha=0.001,
        batch_size="auto",
        learning_rate_init=0.001,
        max_iter=500,
        random_state=42,
    ):
        model = MLPClassifier(
            hidden_layer_sizes=hidden_layer_sizes,
            activation=activation,
            solver=solver,
            alpha=alpha,
            batch_size=batch_size,
            learning_rate_init=learning_rate_init,
            max_iter=max_iter,
            random_state=random_state,
            early_stopping=True,
            validation_fraction=0.1,
        )
        super().__init__("MLP", model)


class TabPFNModel(BaseModel):
    """
    Pretrained TabPFN classifier (HuggingFace-hosted weights).

    Serves as the required baseline model.
    """

    def __init__(
        self,
        device="auto",
        n_configurations=32,
        n_estimators=None,
        model_path=None,
        model_version=None,
        ignore_pretraining_limits=True,
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
            # Version strings are handled via model_version; treat as no explicit path.
            resolved_model_path = None
        if not resolved_model_path:
            resolved_model_path = "auto"

        # Set desired model version for this process (env or parameter wins; default v2).
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


class GradientBoostingModel(BaseModel):
    """Gradient Boosting classifier."""

    def __init__(
        self,
        n_estimators=200,
        learning_rate=0.1,
        max_depth=5,
        min_samples_split=2,
        min_samples_leaf=1,
        subsample=0.8,
        random_state=42,
    ):
        model = GradientBoostingClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            subsample=subsample,
            random_state=random_state,
        )
        super().__init__("GradientBoosting", model)


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
        "lr": LogisticRegressionModel,
        "logistic": LogisticRegressionModel,
        "mlp": MLPModel,
        "gb": GradientBoostingModel,
        "gradient_boosting": GradientBoostingModel,
        "baseline": TabPFNModel,
        "tabpfn": TabPFNModel,
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
