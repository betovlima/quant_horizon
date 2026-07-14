"""Tests for the investment model factory and probability ensemble."""

from __future__ import annotations

import os
import unittest
from datetime import date
from unittest.mock import patch

import numpy as np
from pydantic import ValidationError

from quant_horizon.dtos import DailyAnalysisRequest, DailyForecastRequest, PeriodBacktestRequest
from quant_horizon.investment_models import (
    MappedClassifier,
    ProbabilityEnsemble,
    available_models,
    create_model,
    normalize_model_name,
)


class ModelFactoryTests(unittest.TestCase):
    def test_lists_supported_models(self) -> None:
        self.assertEqual(
            available_models(),
            (
                "lightgbm",
                "catboost",
                "xgboost",
                "logistic",
                "random_forest",
                "ensemble",
            ),
        )

    def test_normalizes_alias_and_environment_variable(self) -> None:
        self.assertEqual(normalize_model_name("XGB"), "xgboost")
        self.assertEqual(normalize_model_name("logistic_regression"), "logistic")
        with patch.dict(os.environ, {"QUANT_HORIZON_MODEL": "rf"}):
            self.assertEqual(normalize_model_name(None), "random_forest")

    def test_rejects_unknown_model(self) -> None:
        with self.assertRaisesRegex(ValueError, "Invalid model"):
            normalize_model_name("neural_network")

    def test_logistic_regression_matches_interface(self) -> None:
        X = np.array(
            [
                [-2.0, 0.0],
                [-1.0, 0.2],
                [-0.2, 0.1],
                [0.0, 0.0],
                [0.2, -0.1],
                [1.0, -0.2],
                [2.0, 0.0],
                [2.5, 0.2],
                [-2.5, -0.2],
            ]
        )
        y = np.array([-1, -1, 0, 0, 0, 1, 1, 1, -1])
        model = create_model(model_name="logistic", random_state=42)

        model.fit(X, y)
        probabilities = model.predict_proba(X[:2])

        np.testing.assert_array_equal(model.classes_, np.array([-1, 0, 1]))
        self.assertEqual(probabilities.shape, (2, 3))
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))
        self.assertEqual(model.feature_importances_.shape, (2,))


class MappedEstimatorTests(unittest.TestCase):
    def test_maps_negative_classes_to_contiguous_indices(self) -> None:
        class FakeEstimator:
            def set_params(self, **parameters):
                self.parameters = parameters
                return self

            def fit(self, X, y):
                self.training_y = np.asarray(y)
                return self

            def predict_proba(self, X):
                return np.tile([0.2, 0.3, 0.5], (len(X), 1))

        adapter = MappedClassifier(FakeEstimator)
        adapter.fit(np.zeros((3, 1)), np.array([-1, 0, 1]))

        np.testing.assert_array_equal(adapter.classes_, np.array([-1, 0, 1]))
        np.testing.assert_array_equal(adapter.estimator_.training_y, np.array([0, 1, 2]))
        self.assertEqual(adapter.estimator_.parameters["num_class"], 3)


class EnsembleTests(unittest.TestCase):
    def test_aligns_and_averages_probabilities(self) -> None:
        class FixedModel:
            def __init__(self, classes, probabilities):
                self.classes_ = np.asarray(classes)
                self.probabilities = np.asarray(probabilities, dtype=float)

            def fit(self, X, y):
                return self

            def predict_proba(self, X):
                return np.tile(self.probabilities, (len(X), 1))

        ensemble = ProbabilityEnsemble(
            factories=(
                lambda: FixedModel([-1, 0, 1], [0.6, 0.3, 0.1]),
                lambda: FixedModel([1, 0, -1], [0.5, 0.3, 0.2]),
            )
        )
        X = np.zeros((2, 1))
        ensemble.fit(X, np.array([-1, 0, 1]))

        probabilities = ensemble.predict_proba(X)

        np.testing.assert_allclose(probabilities[0], np.array([0.4, 0.3, 0.3]))
        np.testing.assert_allclose(probabilities.sum(axis=1), np.ones(2))


class APIModelContractTests(unittest.TestCase):
    def test_daily_forecast_and_backtest_accept_model_per_request(self) -> None:
        forecast = DailyForecastRequest(
            ticker="AAPL",
            model="catboost",
            start_date=date(2026, 7, 13),
            end_date=date(2026, 7, 17),
        )
        backtest = PeriodBacktestRequest(
            ticker="AAPL",
            model="ensemble",
            start_date=date(2025, 1, 1),
            end_date=date(2025, 12, 31),
        )

        self.assertEqual(forecast.model, "catboost")
        self.assertEqual(backtest.model, "ensemble")

    def test_daily_analysis_accepts_past_and_near_future(self) -> None:
        analysis = DailyAnalysisRequest(
            ticker="AAPL",
            model="lightgbm",
            start_date=date(2026, 7, 1),
            end_date=date(2026, 7, 17),
            horizon_trading_days=5,
        )

        self.assertEqual(analysis.end_date, date(2026, 7, 17))
        self.assertEqual(analysis.horizon_trading_days, 5)

    def test_dto_rejects_unsupported_model(self) -> None:
        with self.assertRaises(ValidationError):
            DailyForecastRequest(
                ticker="AAPL",
                model="neural_network",
                start_date=date(2026, 7, 13),
                end_date=date(2026, 7, 17),
            )


if __name__ == "__main__":
    unittest.main()
