"""Tests for financial backtest business rules."""

from __future__ import annotations

import unittest
from datetime import date

import pandas as pd
from fastapi.testclient import TestClient

from quant_horizon.backtest_validation import (
    validate_backtest_date_range,
    validate_backtest_market_data,
)
from quant_horizon.dtos import PeriodBacktestRequest
from quant_horizon.exceptions import BusinessRuleError
from quant_horizon.api import app


class BacktestDateRangeTests(unittest.TestCase):
    def test_equal_dates_return_specific_business_error(self) -> None:
        with self.assertRaises(BusinessRuleError) as raised:
            validate_backtest_date_range(
                start_date=date(2026, 7, 17),
                end_date=date(2026, 7, 17),
                horizon_trading_days=5,
                today=date(2026, 7, 17),
            )

        self.assertEqual(raised.exception.code, "INVALID_BACKTEST_DATE_RANGE")
        self.assertEqual(
            raised.exception.context,
            {
                "start_date": "2026-07-17",
                "end_date": "2026-07-17",
            },
        )

    def test_start_after_end_returns_specific_business_error(self) -> None:
        with self.assertRaises(BusinessRuleError) as raised:
            validate_backtest_date_range(
                start_date=date(2026, 7, 18),
                end_date=date(2026, 7, 17),
                horizon_trading_days=5,
                today=date(2026, 7, 18),
            )

        self.assertEqual(raised.exception.code, "INVALID_BACKTEST_DATE_RANGE")

    def test_future_end_date_returns_specific_business_error(self) -> None:
        with self.assertRaises(BusinessRuleError) as raised:
            validate_backtest_date_range(
                start_date=date(2026, 7, 1),
                end_date=date(2026, 7, 18),
                horizon_trading_days=5,
                today=date(2026, 7, 17),
            )

        self.assertEqual(raised.exception.code, "BACKTEST_END_DATE_IN_FUTURE")

    def test_request_dto_leaves_business_rules_to_service(self) -> None:
        request = PeriodBacktestRequest(
            ticker="AAPL",
            start_date=date(2026, 7, 17),
            end_date=date(2026, 7, 17),
        )

        self.assertEqual(request.start_date, request.end_date)


class BacktestMarketDataTests(unittest.TestCase):
    def test_weekend_only_period_has_no_trading_sessions(self) -> None:
        prices = pd.DataFrame(
            {"close": [100.0]},
            index=pd.to_datetime(["2026-07-17"]),
        )

        with self.assertRaises(BusinessRuleError) as raised:
            validate_backtest_market_data(
                prices=prices,
                start_date=date(2026, 7, 18),
                end_date=date(2026, 7, 19),
                horizon_trading_days=1,
            )

        self.assertEqual(raised.exception.code, "BACKTEST_NO_TRADING_SESSIONS")

    def test_period_requires_horizon_plus_entry_session(self) -> None:
        prices = pd.DataFrame(
            {"close": [100.0, 101.0, 102.0]},
            index=pd.to_datetime(
                [
                    "2026-07-13",
                    "2026-07-14",
                    "2026-07-15",
                ]
            ),
        )

        with self.assertRaises(BusinessRuleError) as raised:
            validate_backtest_market_data(
                prices=prices,
                start_date=date(2026, 7, 13),
                end_date=date(2026, 7, 15),
                horizon_trading_days=5,
            )

        self.assertEqual(
            raised.exception.code,
            "BACKTEST_INSUFFICIENT_TRADING_SESSIONS",
        )
        self.assertEqual(
            raised.exception.context["available_trading_sessions"],
            3,
        )
        self.assertEqual(
            raised.exception.context["required_trading_sessions"],
            6,
        )

    def test_sufficient_trading_sessions_are_accepted(self) -> None:
        prices = pd.DataFrame(
            {"close": [100.0] * 6},
            index=pd.to_datetime(
                [
                    "2026-07-10",
                    "2026-07-13",
                    "2026-07-14",
                    "2026-07-15",
                    "2026-07-16",
                    "2026-07-17",
                ]
            ),
        )

        validate_backtest_market_data(
            prices=prices,
            start_date=date(2026, 7, 10),
            end_date=date(2026, 7, 17),
            horizon_trading_days=5,
        )


class BacktestApiErrorTests(unittest.TestCase):
    def test_period_endpoint_returns_structured_date_range_error(self) -> None:
        client = TestClient(app)

        response = client.post(
            "/v1/backtests/period",
            json={
                "ticker": "AAPL",
                "start_date": "2026-07-17",
                "end_date": "2026-07-17",
                "horizon_trading_days": 5,
            },
        )

        self.assertEqual(response.status_code, 422)
        self.assertEqual(
            response.json(),
            {
                "error": {
                    "code": "INVALID_BACKTEST_DATE_RANGE",
                    "message": (
                        "The start date must be earlier than the end date."
                    ),
                    "context": {
                        "start_date": "2026-07-17",
                        "end_date": "2026-07-17",
                    },
                }
            },
        )


if __name__ == "__main__":
    unittest.main()
