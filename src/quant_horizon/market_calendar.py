"""Trading-session rules for the supported markets."""

from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd


def calendar_for_ticker(ticker: str) -> tuple[str, str]:
    """Return the exchange calendar used to project trading sessions."""
    if ticker.upper().endswith(".SA"):
        return "B3", "B3"
    return "NYSE", "United States (NYSE/Nasdaq)"


def next_trading_days(
    ticker: str,
    signal_date: date | datetime | pd.Timestamp,
    quantity: int,
) -> tuple[str, list[date]]:
    """Calculate future sessions using the asset exchange holiday calendar."""
    import pandas_market_calendars as mcal

    calendar_code, calendar_name = calendar_for_ticker(ticker)
    calendar = mcal.get_calendar(calendar_code)
    start_value = pd.Timestamp(signal_date).normalize() + pd.Timedelta(days=1)
    end_value = start_value + pd.Timedelta(days=max(14, quantity * 3))

    while True:
        sessions = calendar.schedule(start_date=start_value, end_date=end_value).index
        if len(sessions) >= quantity:
            dates = [pd.Timestamp(session).date() for session in sessions[:quantity]]
            return calendar_name, dates
        end_value += pd.Timedelta(days=max(14, quantity * 2))


def sessions_with_previous_close(
    ticker: str,
    start_date: date,
    end_date: date,
) -> tuple[str, list[tuple[date, date]]]:
    import pandas_market_calendars as mcal

    calendar_code, calendar_name = calendar_for_ticker(ticker)
    calendar = mcal.get_calendar(calendar_code)
    search_start = start_date - timedelta(days=40)
    sessions = calendar.schedule(start_date=search_start, end_date=end_date).index
    dates = [pd.Timestamp(session).date() for session in sessions]
    pairs: list[tuple[date, date]] = []
    for index_value, target_date in enumerate(dates):
        if start_date <= target_date <= end_date and index_value > 0:
            pairs.append((target_date, dates[index_value - 1]))
    if not pairs:
        raise ValueError("There are no trading sessions in the requested period.")
    return calendar_name, pairs


def is_trading_day(ticker: str, trade_date: date) -> bool:
    import pandas_market_calendars as mcal

    calendar_code, _ = calendar_for_ticker(ticker)
    calendar = mcal.get_calendar(calendar_code)
    return not calendar.schedule(
        start_date=trade_date,
        end_date=trade_date,
    ).empty


def horizon_between_close_and_target(
    ticker: str,
    latest_close: date,
    target_date: date,
) -> int:
    import pandas_market_calendars as mcal

    calendar_code, _ = calendar_for_ticker(ticker)
    calendar = mcal.get_calendar(calendar_code)
    sessions = calendar.schedule(
        start_date=latest_close + timedelta(days=1),
        end_date=target_date,
    ).index
    dates = [pd.Timestamp(session).date() for session in sessions]
    if target_date not in dates:
        raise ValueError("The projected date is not a trading day for this asset.")
    return dates.index(target_date) + 1


__all__ = [
    "calendar_for_ticker",
    "horizon_between_close_and_target",
    "is_trading_day",
    "next_trading_days",
    "sessions_with_previous_close",
]
