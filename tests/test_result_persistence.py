"""Tests for forecast and backtest persistence in the operational database."""

from __future__ import annotations

import sqlite3
import tempfile
import unittest
from contextlib import closing
from datetime import date, datetime, timezone
from pathlib import Path
from unittest.mock import patch

from quant_horizon import persistence


class ResultPersistenceTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.database = Path(self.temporary_directory.name) / "state.sqlite3"
        self.state_db_patch = patch.object(persistence, "STATE_DB", self.database)
        self.state_db_patch.start()

    def tearDown(self) -> None:
        self.state_db_patch.stop()
        self.temporary_directory.cleanup()

    def test_forecast_and_backtest_are_saved_without_auxiliary_files(self) -> None:
        forecast = {
            "ticker": "AAPL",
            "signal_date": datetime(2026, 7, 10, 20, tzinfo=timezone.utc),
            "action": "BUY",
            "forecast_trading_dates": [date(2026, 7, 13)],
        }
        persistence.save_current_forecast("AAPL", "lightgbm", forecast)
        backtest_id = persistence.save_backtest(
            ticker="AAPL",
            model_used="lightgbm",
            parameters={"start_date": date(2025, 1, 1), "horizon": 5},
            metrics={"final_capital": 110.5},
            trades=[{"position": 1, "return_value": 0.105}],
            retrainings=[{"date": date(2025, 1, 1), "threshold": 0.55}],
        )

        data = persistence.list_persisted_data("AAPL")

        self.assertGreater(backtest_id, 0)
        self.assertEqual(data["current_forecast"]["action"], "BUY")
        self.assertEqual(data["backtests"][0]["id"], backtest_id)
        self.assertEqual(data["backtests"][0]["total_trades"], 1)
        self.assertEqual(data["backtests"][0]["total_retrainings"], 1)

        with closing(sqlite3.connect(self.database)) as connection:
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM current_forecasts").fetchone()[0],
                1,
            )
            self.assertEqual(
                connection.execute("SELECT COUNT(*) FROM backtests").fetchone()[0],
                1,
            )


if __name__ == "__main__":
    unittest.main()
