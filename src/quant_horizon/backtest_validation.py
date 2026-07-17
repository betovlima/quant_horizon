"""Business rules for financial backtest date ranges and market coverage."""

from __future__ import annotations

from datetime import date

import pandas as pd

from .exceptions import BusinessRuleError


MAX_BACKTEST_CALENDAR_DAYS = 365 * 15


def validate_backtest_date_range(
    *,
    start_date: date,
    end_date: date,
    horizon_trading_days: int,
    today: date | None = None,
) -> None:
    """Validate the requested range before downloading and training models."""

    reference_date = today or date.today()

    if start_date >= end_date:
        raise BusinessRuleError(
            code="INVALID_BACKTEST_DATE_RANGE",
            message="The start date must be earlier than the end date.",
            context={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )

    if start_date > reference_date:
        raise BusinessRuleError(
            code="BACKTEST_START_DATE_IN_FUTURE",
            message="The backtest start date cannot be in the future.",
            context={
                "start_date": start_date.isoformat(),
                "today": reference_date.isoformat(),
            },
        )

    if end_date > reference_date:
        raise BusinessRuleError(
            code="BACKTEST_END_DATE_IN_FUTURE",
            message="The backtest end date cannot be in the future.",
            context={
                "end_date": end_date.isoformat(),
                "today": reference_date.isoformat(),
            },
        )

    calendar_days = (end_date - start_date).days
    if calendar_days > MAX_BACKTEST_CALENDAR_DAYS:
        raise BusinessRuleError(
            code="BACKTEST_DATE_RANGE_TOO_LARGE",
            message="A backtest must not exceed 15 years.",
            context={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "calendar_days": calendar_days,
                "maximum_calendar_days": MAX_BACKTEST_CALENDAR_DAYS,
            },
        )

    if horizon_trading_days < 1:
        raise BusinessRuleError(
            code="INVALID_BACKTEST_HORIZON",
            message="The forecast horizon must contain at least one trading day.",
            context={
                "horizon_trading_days": horizon_trading_days,
            },
        )


def validate_backtest_market_data(
    *,
    prices: pd.DataFrame,
    start_date: date,
    end_date: date,
    horizon_trading_days: int,
) -> None:
    """Ensure the selected period contains enough real trading sessions."""

    if prices.empty:
        raise BusinessRuleError(
            code="BACKTEST_NO_MARKET_DATA",
            message="No market data was found for the selected period.",
            context={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )

    market_dates = pd.DatetimeIndex(pd.to_datetime(prices.index))
    session_dates = sorted(
        {
            timestamp.date()
            for timestamp in market_dates
            if start_date <= timestamp.date() <= end_date
        }
    )

    if not session_dates:
        raise BusinessRuleError(
            code="BACKTEST_NO_TRADING_SESSIONS",
            message="The selected period does not contain any trading sessions.",
            context={
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
            },
        )

    required_sessions = horizon_trading_days + 1
    available_sessions = len(session_dates)

    if available_sessions < required_sessions:
        raise BusinessRuleError(
            code="BACKTEST_INSUFFICIENT_TRADING_SESSIONS",
            message=(
                "The selected period does not contain enough trading sessions "
                "for the requested horizon."
            ),
            context={
                "available_trading_sessions": available_sessions,
                "required_trading_sessions": required_sessions,
                "horizon_trading_days": horizon_trading_days,
                "first_available_date": session_dates[0].isoformat(),
                "last_available_date": session_dates[-1].isoformat(),
            },
        )


__all__ = [
    "MAX_BACKTEST_CALENDAR_DAYS",
    "validate_backtest_date_range",
    "validate_backtest_market_data",
]
