"""Tests for daily analysis, independently of the financial backtest."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from quant_horizon import pipeline


class FixedModel:
    classes_ = np.array([-1, 0, 1])

    def predict_proba(self, X):
        return np.tile([0.10, 0.20, 0.70], (len(X), 1))


class DailyAnalysisPipelineTests(unittest.TestCase):
    def test_produces_one_signal_per_session_with_five_day_horizon(self) -> None:
        index = pd.bdate_range("2024-01-02", periods=400)
        features = pd.DataFrame(
            {"return_value": np.linspace(-1, 1, len(index))}, index=index
        )
        X = features.copy()
        events = pd.DataFrame(
            {
                "label": np.resize(np.array([-1, 0, 1]), len(index)),
                "gross_return": np.linspace(-0.02, 0.03, len(index)),
                "entry_price": 100.0,
                "exit_price": 101.0,
                "entry_date": index + pd.offsets.BDay(1),
                "exit_date": index + pd.offsets.BDay(5),
            },
            index=index,
        )
        targets = events.copy()
        test_dates = index[320:330]

        with patch.object(
            pipeline,
            "train_with_internal_threshold",
            return_value=(FixedModel(), 0.55),
        ):
            signals, retrainings = pipeline.predict_daily_historical_signals(
                features=features,
                X_history=X,
                historical_events=events,
                full_target=targets,
                start_date=test_dates[0],
                end_date=test_dates[-1],
                horizon=5,
                side_cost=0.0005,
                annual_short_cost=0.0,
                training_window_days=0,
                position_mode="long_short",
                threshold=0.55,
                optimize_threshold=True,
                retrain_frequency_days=4,
                model_name="lightgbm",
            )

        self.assertEqual(len(signals), 10)
        self.assertEqual(list(signals["signal_date"]), list(test_dates))
        self.assertTrue((signals["action"] == "BUY").all())
        self.assertEqual(len(retrainings), 3)


if __name__ == "__main__":
    unittest.main()
