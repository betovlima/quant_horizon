"""Classification-model factory used by the investment pipeline.

Every model exposes ``fit``, ``predict_proba``, and ``classes_``. LightGBM is
the default, while requests or ``QUANT_HORIZON_MODEL`` may select another one.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from typing import Any, Literal

import numpy as np


ModelName = Literal[
    "lightgbm",
    "catboost",
    "xgboost",
    "logistic",
    "random_forest",
    "ensemble",
]

SUPPORTED_MODELS: tuple[ModelName, ...] = (
    "lightgbm",
    "catboost",
    "xgboost",
    "logistic",
    "random_forest",
    "ensemble",
)

MODEL_ALIASES = {
    "lgbm": "lightgbm",
    "light_gbm": "lightgbm",
    "cat": "catboost",
    "xgb": "xgboost",
    "logistic": "logistic",
    "logistic_regression": "logistic",
    "randomforest": "random_forest",
    "rf": "random_forest",
    "voting": "ensemble",
}


def normalize_model_name(model_name: str | None) -> ModelName:
    name = (model_name or os.environ.get("QUANT_HORIZON_MODEL", "lightgbm")).strip().lower()
    name = MODEL_ALIASES.get(name, name)
    if name not in SUPPORTED_MODELS:
        options = ", ".join(SUPPORTED_MODELS)
        raise ValueError(f"Invalid model: {name!r}. Choose one of: {options}.")
    return name  # type: ignore[return-value]


def available_models() -> tuple[str, ...]:
    return SUPPORTED_MODELS


def _dependency_error(model_name: str, package_name: str, exc: ImportError) -> RuntimeError:
    return RuntimeError(
        f"Model {model_name} requires package {package_name}. "
        "Install dependencies with: pip install -r requirements.txt"
    )


def _create_lightgbm(random_state: int):
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise _dependency_error("lightgbm", "lightgbm", exc) from exc

    # Keep the parameters used by the original LightGBM implementation.
    return lgb.LGBMClassifier(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=4,
        num_leaves=15,
        min_child_samples=40,
        subsample=0.80,
        subsample_freq=1,
        colsample_bytree=0.80,
        reg_alpha=0.20,
        reg_lambda=1.00,
        random_state=random_state,
        n_jobs=-1,
        verbosity=-1,
        importance_type="gain",
    )


def _create_catboost(random_state: int):
    try:
        from catboost import CatBoostClassifier
    except ImportError as exc:
        raise _dependency_error("catboost", "catboost", exc) from exc

    return CatBoostClassifier(
        iterations=400,
        learning_rate=0.03,
        depth=4,
        loss_function="MultiClass",
        l2_leaf_reg=3.0,
        random_strength=1.0,
        random_seed=random_state,
        has_time=True,
        allow_writing_files=False,
        thread_count=-1,
        verbose=False,
    )


class MappedClassifier:
    """Adapt estimators that require zero-based contiguous integer classes."""

    def __init__(self, factory: Callable[[], Any]):
        self._factory = factory
        self.estimator_: Any | None = None
        self.classes_: np.ndarray = np.array([], dtype=int)

    def fit(self, X, y):
        self.classes_ = np.asarray(sorted(np.unique(np.asarray(y))), dtype=int)
        if len(self.classes_) < 2:
            raise ValueError("Training requires at least two classes.")
        mapping = {class_value: index_value for index_value, class_value in enumerate(self.classes_)}
        mapped_y = np.asarray(
            [mapping[int(class_value)] for class_value in y], dtype=int
        )
        self.estimator_ = self._factory()
        if hasattr(self.estimator_, "set_params"):
            self.estimator_.set_params(num_class=len(self.classes_))
        self.estimator_.fit(X, mapped_y)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if self.estimator_ is None:
            raise RuntimeError("The model must be trained before prediction.")
        return np.asarray(self.estimator_.predict_proba(X), dtype=float)

    @property
    def feature_importances_(self) -> np.ndarray:
        if self.estimator_ is None or not hasattr(
            self.estimator_, "feature_importances_"
        ):
            raise AttributeError("The estimator does not provide feature_importances_.")
        return np.asarray(self.estimator_.feature_importances_, dtype=float)


def _create_xgboost(random_state: int):
    try:
        from xgboost import XGBClassifier
    except ImportError as exc:
        raise _dependency_error("xgboost", "xgboost", exc) from exc

    def factory():
        return XGBClassifier(
            n_estimators=400,
            learning_rate=0.03,
            max_depth=4,
            min_child_weight=5.0,
            subsample=0.80,
            colsample_bytree=0.80,
            reg_alpha=0.20,
            reg_lambda=1.00,
            objective="multi:softprob",
            eval_metric="mlogloss",
            random_state=random_state,
            n_jobs=-1,
            verbosity=0,
        )

    return MappedClassifier(factory)


class LogisticRegressionWithImportance:
    """Expose absolute coefficient importance for sklearn logistic regression."""

    def __init__(self, random_state: int):
        from sklearn.impute import SimpleImputer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        self.pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=0.50,
                        max_iter=2_000,
                        class_weight="balanced",
                        random_state=random_state,
                    ),
                ),
            ]
        )
        self.classes_: np.ndarray = np.array([], dtype=int)

    def fit(self, X, y):
        self.pipeline.fit(X, y)
        self.classes_ = np.asarray(self.pipeline.classes_, dtype=int)
        return self

    def predict_proba(self, X) -> np.ndarray:
        return np.asarray(self.pipeline.predict_proba(X), dtype=float)

    @property
    def feature_importances_(self) -> np.ndarray:
        coefficients = np.asarray(
            self.pipeline.named_steps["model"].coef_,
            dtype=float,
        )
        return np.mean(np.abs(coefficients), axis=0)


def _create_logistic(random_state: int):
    return LogisticRegressionWithImportance(random_state)


def _create_random_forest(random_state: int):
    from sklearn.ensemble import RandomForestClassifier

    return RandomForestClassifier(
        n_estimators=500,
        max_depth=6,
        min_samples_leaf=20,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=random_state,
        n_jobs=-1,
    )


class ProbabilityEnsemble:
    """Train independent classifiers and combine their probabilities."""

    def __init__(
        self,
        factories: Sequence[Callable[[], Any]],
        weights: Sequence[float] | None = None,
    ):
        if not factories:
            raise ValueError("The ensemble requires at least one model.")
        self._factories = tuple(factories)
        self._weights = None if weights is None else np.asarray(weights, dtype=float)
        if self._weights is not None:
            if len(self._weights) != len(self._factories) or np.any(self._weights <= 0):
                raise ValueError("Weights must be positive and match the model count.")
        self.models_: list[Any] = []
        self.classes_: np.ndarray = np.array([], dtype=int)

    def fit(self, X, y):
        self.classes_ = np.asarray(sorted(np.unique(np.asarray(y))), dtype=int)
        self.models_ = []
        for factory in self._factories:
            model = factory()
            model.fit(X, y)
            self.models_.append(model)
        return self

    def predict_proba(self, X) -> np.ndarray:
        if not self.models_:
            raise RuntimeError("The ensemble must be trained before prediction.")

        probabilities = []
        for model in self.models_:
            model_probabilities = np.asarray(model.predict_proba(X), dtype=float)
            aligned = np.zeros((len(model_probabilities), len(self.classes_)), dtype=float)
            mapping = {int(class_value): index_value for index_value, class_value in enumerate(model.classes_)}
            for destination, class_value in enumerate(self.classes_):
                if int(class_value) in mapping:
                    aligned[:, destination] = model_probabilities[:, mapping[int(class_value)]]
            probabilities.append(aligned)

        return np.average(
            np.stack(probabilities, axis=0),
            axis=0,
            weights=self._weights,
        )

    @property
    def feature_importances_(self) -> np.ndarray:
        importances = [
            np.asarray(model.feature_importances_, dtype=float)
            for model in self.models_
            if hasattr(model, "feature_importances_")
        ]
        if not importances:
            raise AttributeError("The ensemble models do not provide importances.")
        return np.mean(np.stack(importances, axis=0), axis=0)


def _create_ensemble(random_state: int):
    return ProbabilityEnsemble(
        factories=(
            lambda: _create_lightgbm(random_state),
            lambda: _create_catboost(random_state),
            lambda: _create_xgboost(random_state),
        )
    )


def create_model(
    random_state: int = 42,
    model_name: str | None = None,
):
    """Create a classifier compatible with the investment pipeline."""
    name = normalize_model_name(model_name)
    factories: dict[str, Callable[[int], Any]] = {
        "lightgbm": _create_lightgbm,
        "catboost": _create_catboost,
        "xgboost": _create_xgboost,
        "logistic": _create_logistic,
        "random_forest": _create_random_forest,
        "ensemble": _create_ensemble,
    }
    return factories[name](random_state)


__all__ = [
    "LogisticRegressionWithImportance",
    "MappedClassifier",
    "ModelName",
    "ProbabilityEnsemble",
    "SUPPORTED_MODELS",
    "available_models",
    "create_model",
    "normalize_model_name",
]
