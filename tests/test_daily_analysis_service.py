"""Integrated test for the daily analysis service contract."""

from __future__ import annotations

import sys
import unittest
from datetime import date
from types import ModuleType
from unittest.mock import patch

import pandas as pd

from quant_horizon.dtos import DailyAnalysisRequest, DailyAnalysisResponse

try:
    from quant_horizon.services import generate_daily_analysis
except ModuleNotFoundError as exc:
    if exc.name != "fastapi":
        raise
    fake_fastapi = ModuleType("fastapi")

    class FakeHTTPException(Exception):
        def __init__(self, status_code: int, detail: str):
            self.status_code = status_code
            self.detail = detail

    fake_fastapi.HTTPException = FakeHTTPException
    sys.modules["fastapi"] = fake_fastapi
    from quant_horizon.services import generate_daily_analysis


class DailyAnalysisServiceTests(unittest.TestCase):
    @patch("quant_horizon.services.next_trading_days")
    @patch("quant_horizon.services.horizon_between_close_and_target")
    @patch("quant_horizon.services.sessions_with_previous_close")
    @patch("quant_horizon.services.pipeline.predict_latest_close")
    @patch("quant_horizon.services.pipeline.predict_daily_historical_signals")
    @patch("quant_horizon.services.pipeline.build_dataset")
    @patch("quant_horizon.services.pipeline.build_targets")
    @patch("quant_horizon.services.pipeline.build_features")
    @patch("quant_horizon.services.pipeline.download_data")
    def test_combines_historical_result_and_future_projection(
        self,
        download_data,
        build_features,
        build_targets,
        build_dataset,
        predict_historical,
        predict_current,
        sessions,
        calculate_horizon,
        next_sessions,
    ) -> None:
        index = pd.DatetimeIndex(["2026-07-09", "2026-07-10"])
        data = pd.DataFrame(
            {"open": [312.0, 314.0], "close": [313.0, 315.0]}, index=index
        )
        features = pd.DataFrame({"return_value": [0.01, 0.02]}, index=index)
        X = features.copy()
        events = pd.DataFrame(
            {
                "label": [1, 1],
                "gross_return": [0.01, 0.02],
                "entry_price": [312.0, 314.0],
                "exit_price": [313.0, 315.0],
                "entry_date": index,
                "exit_date": index,
            },
            index=index,
        )
        signals = pd.DataFrame(
            [
                {
                    "signal_date": pd.Timestamp("2026-07-10"),
                    "probability_down": 0.1,
                    "probability_neutral": 0.2,
                    "probability_up": 0.7,
                    "threshold": 0.55,
                    "action": "BUY",
                    "predicted_class": 1,
                    "result_available": True,
                    "entry_date": pd.Timestamp("2026-07-13"),
                    "entry_price": 316.0,
                    "exit_date": pd.Timestamp("2026-07-17"),
                    "exit_price": 320.0,
                    "observed_return": 0.0126,
                    "actual_class": 1,
                    "is_correct": True,
                    "training_samples": 800,
                }
            ]
        )
        download_data.return_value = data
        build_features.return_value = features
        build_targets.return_value = events
        build_dataset.return_value = (X, events)
        predict_historical.return_value = (
            signals,
            pd.DataFrame([{"retrained_at": index[-1]}]),
        )
        sessions.return_value = (
            "United States (NYSE/Nasdaq)",
            [
                (date(2026, 7, 10), date(2026, 7, 9)),
                (date(2026, 7, 13), date(2026, 7, 10)),
            ],
        )
        calculate_horizon.return_value = 1
        next_sessions.return_value = (
            "United States (NYSE/Nasdaq)",
            [date(2026, 7, 13)],
        )
        predict_current.return_value = {
            "probability_down": 0.15,
            "probability_neutral": 0.20,
            "probability_up": 0.65,
            "threshold": 0.55,
            "action": "BUY",
            "reference_price": 315.0,
            "training_samples": 801,
        }

        response = generate_daily_analysis(
            DailyAnalysisRequest(
                ticker="AAPL",
                model="lightgbm",
                start_date=date(2026, 7, 10),
                end_date=date(2026, 7, 13),
                horizon_trading_days=5,
            )
        )
        contract = DailyAnalysisResponse.model_validate(response)

        self.assertEqual(contract.total_trading_days, 2)
        self.assertEqual(contract.total_evaluated, 1)
        self.assertEqual(contract.total_preliminary, 1)
        self.assertEqual(contract.accuracy_rate, 1.0)
        self.assertEqual(contract.analyses[0].forecast_type, "HISTORICAL")
        self.assertEqual(contract.analyses[1].forecast_type, "PRELIMINARY")


if __name__ == "__main__":
    unittest.main()
