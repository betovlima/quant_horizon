"""SQLite persistence for forecasts, backtests, and simulated trades."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import date, datetime, timezone
from collections.abc import Iterator
from typing import Any

from .market_calendar import is_trading_day
from .config import STATE_DB
from .dtos import TradeAcceptanceRequest
from .entities import PositionState, Trade, TradeType, normalize_ticker


def validate_ticker(ticker: str) -> str:
    try:
        return normalize_ticker(ticker)
    except ValueError as exc:
        raise ValueError("Invalid ticker.") from exc


def _json_value(value: Any) -> Any:
    if isinstance(value, (date, datetime)):
        return value.isoformat()
    if hasattr(value, "item"):
        return value.item()
    raise TypeError(f"Type is not JSON serializable: {type(value).__name__}")


def _encode_json(value: Any) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        default=_json_value,
        separators=(",", ":"),
    )


def connect_state() -> sqlite3.Connection:
    STATE_DB.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(STATE_DB, timeout=10)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys=ON")
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS trades (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            trade_type TEXT NOT NULL CHECK (trade_type IN ('BUY', 'SELL')),
            acceptance_date TEXT NOT NULL,
            acceptance_price REAL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_trades_ticker_data "
        "ON trades (ticker, acceptance_date, id)"
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS daily_forecasts (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            market_calendar TEXT NOT NULL,
            model_used TEXT NOT NULL DEFAULT 'lightgbm',
            target_date TEXT NOT NULL,
            base_close_date TEXT NOT NULL,
            status TEXT NOT NULL,
            position_before TEXT NOT NULL,
            suggested_action TEXT NOT NULL,
            probability_down REAL,
            probability_neutral REAL,
            probability_up REAL,
            threshold REAL,
            reference_price REAL,
            description TEXT NOT NULL,
            forecast_type TEXT,
            horizon_used INTEGER,
            expected_update_date TEXT,
            generated_at TEXT NOT NULL,
            UNIQUE (ticker, target_date)
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_forecasts_ticker_data "
        "ON daily_forecasts (ticker, target_date)"
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS current_forecasts (
            ticker TEXT PRIMARY KEY,
            model_used TEXT NOT NULL,
            signal_date TEXT NOT NULL,
            content_json TEXT NOT NULL,
            generated_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS backtests (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            model_used TEXT NOT NULL,
            parameters_json TEXT NOT NULL,
            metrics_json TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_backtests_ticker_data "
        "ON backtests (ticker, created_at DESC, id DESC)"
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_trades (
            backtest_id INTEGER NOT NULL,
            sequence_index INTEGER NOT NULL,
            content_json TEXT NOT NULL,
            PRIMARY KEY (backtest_id, sequence_index),
            FOREIGN KEY (backtest_id) REFERENCES backtests(id) ON DELETE CASCADE
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS backtest_retrainings (
            backtest_id INTEGER NOT NULL,
            sequence_index INTEGER NOT NULL,
            content_json TEXT NOT NULL,
            PRIMARY KEY (backtest_id, sequence_index),
            FOREIGN KEY (backtest_id) REFERENCES backtests(id) ON DELETE CASCADE
        )
        """
    )
    forecast_columns = {
        row[1] for row in connection.execute("PRAGMA table_info(daily_forecasts)")
    }
    if "forecast_type" not in forecast_columns:
        connection.execute("ALTER TABLE daily_forecasts ADD COLUMN forecast_type TEXT")
    if "horizon_used" not in forecast_columns:
        connection.execute("ALTER TABLE daily_forecasts ADD COLUMN horizon_used INTEGER")
    if "expected_update_date" not in forecast_columns:
        connection.execute(
            "ALTER TABLE daily_forecasts ADD COLUMN expected_update_date TEXT"
        )
    if "model_used" not in forecast_columns:
        connection.execute(
            "ALTER TABLE daily_forecasts ADD COLUMN "
            "model_used TEXT NOT NULL DEFAULT 'lightgbm'"
        )
    connection.commit()
    return connection


@contextmanager
def state_connection() -> Iterator[sqlite3.Connection]:
    """Yield a transactional SQLite connection and always close it."""
    connection = connect_state()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def trade_from_row(row: sqlite3.Row) -> Trade:
    return Trade(
        id=int(row["id"]),
        ticker=str(row["ticker"]),
        trade_type=TradeType(str(row["trade_type"])),
        acceptance_date=date.fromisoformat(str(row["acceptance_date"])),
        acceptance_price=(
            float(row["acceptance_price"])
            if row["acceptance_price"] is not None
            else None
        ),
        created_at=datetime.fromisoformat(str(row["created_at"])),
    )


def list_trades(ticker: str) -> list[Trade]:
    with state_connection() as connection:
        rows = connection.execute(
            "SELECT id, ticker, trade_type, acceptance_date, acceptance_price, created_at "
            "FROM trades WHERE ticker = ? ORDER BY acceptance_date, id",
            (ticker,),
        ).fetchall()
    return [trade_from_row(row) for row in rows]


def get_position_state(
    ticker: str,
    trades: list[Trade] | None = None,
) -> PositionState:
    history = trades if trades is not None else list_trades(ticker)
    return PositionState.from_trades(ticker, history)


def register_trade(
    ticker: str,
    acceptance: TradeAcceptanceRequest,
) -> Trade:
    ticker = validate_ticker(ticker)
    if not is_trading_day(ticker, acceptance.acceptance_date):
        raise ValueError("The acceptance date must be a trading day for the asset.")
    with state_connection() as connection:
        latest_row = connection.execute(
            "SELECT id, ticker, trade_type, acceptance_date, acceptance_price, created_at "
            "FROM trades WHERE ticker = ? ORDER BY acceptance_date DESC, id DESC LIMIT 1",
            (ticker,),
        ).fetchone()
        latest = trade_from_row(latest_row) if latest_row else None

        if latest and acceptance.acceptance_date <= latest.acceptance_date:
            raise ValueError("The acceptance must be later than the latest recorded trade.")
        if acceptance.trade_type == "BUY" and latest and latest.trade_type is TradeType.BUY:
            raise ValueError("The simulated position is already long.")
        if acceptance.trade_type == "SELL" and (
            not latest or latest.trade_type is not TradeType.BUY
        ):
            raise ValueError("There is no open simulated purchase to mark as sold.")

        created_at = datetime.now(timezone.utc).isoformat()
        cursor = connection.execute(
            "INSERT INTO trades "
            "(ticker, trade_type, acceptance_date, acceptance_price, created_at) VALUES (?, ?, ?, ?, ?)",
            (
                ticker,
                acceptance.trade_type,
                acceptance.acceptance_date.isoformat(),
                acceptance.acceptance_price,
                created_at,
            ),
        )
        connection.commit()
        row = connection.execute(
            "SELECT id, ticker, trade_type, acceptance_date, acceptance_price, created_at "
            "FROM trades WHERE id = ?",
            (cursor.lastrowid,),
        ).fetchone()
    return trade_from_row(row)


def reset_trades(ticker: str) -> PositionState:
    """Remove all simulated trades for a ticker and return a flat position."""
    ticker = validate_ticker(ticker)
    with state_connection() as connection:
        connection.execute("DELETE FROM trades WHERE ticker = ?", (ticker,))
        connection.commit()
    return get_position_state(ticker, trades=[])


def save_daily_forecasts(
    ticker: str,
    market_calendar: str,
    model_used: str,
    forecasts: list[dict[str, Any]],
) -> None:
    generated_at = datetime.now(timezone.utc).isoformat()
    with state_connection() as connection:
        for item in forecasts:
            connection.execute(
                """
                INSERT INTO daily_forecasts (
                    ticker, market_calendar, model_used,
                    target_date, base_close_date,
                    status, position_before, suggested_action, probability_down, probability_neutral,
                    probability_up, threshold, reference_price, description, forecast_type,
                    horizon_used, expected_update_date, generated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(ticker, target_date) DO UPDATE SET
                    market_calendar = excluded.market_calendar,
                    model_used = excluded.model_used,
                    base_close_date = excluded.base_close_date,
                    status = excluded.status,
                    position_before = excluded.position_before,
                    suggested_action = excluded.suggested_action,
                    probability_down = excluded.probability_down,
                    probability_neutral = excluded.probability_neutral,
                    probability_up = excluded.probability_up,
                    threshold = excluded.threshold,
                    reference_price = excluded.reference_price,
                    description = excluded.description,
                    forecast_type = excluded.forecast_type,
                    horizon_used = excluded.horizon_used,
                    expected_update_date = excluded.expected_update_date,
                    generated_at = excluded.generated_at
                """,
                (
                    ticker,
                    market_calendar,
                    model_used,
                    item["target_date"].isoformat(),
                    item["base_close_date"].isoformat(),
                    item["status"],
                    item["position_before"],
                    item["suggested_action"],
                    item["probability_down"],
                    item["probability_neutral"],
                    item["probability_up"],
                    item["threshold"],
                    item["reference_price"],
                    item["description"],
                    item["forecast_type"],
                    item["horizon_used"],
                    (
                        item["expected_update_date"].isoformat()
                        if item["expected_update_date"]
                        else None
                    ),
                    generated_at,
                ),
            )
        connection.commit()


def save_current_forecast(
    ticker: str,
    model_used: str,
    forecast: dict[str, Any],
) -> None:
    """Replace the latest forecast for a ticker."""
    ticker = validate_ticker(ticker)
    signal_date = forecast.get("signal_date")
    if signal_date is None:
        raise ValueError("The current forecast does not contain signal_date.")
    signal_date_text = (
        signal_date.isoformat()
        if isinstance(signal_date, (date, datetime))
        else str(signal_date)
    )
    with state_connection() as connection:
        connection.execute(
            """
            INSERT INTO current_forecasts (
                ticker, model_used, signal_date, content_json, generated_at
            ) VALUES (?, ?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                model_used = excluded.model_used,
                signal_date = excluded.signal_date,
                content_json = excluded.content_json,
                generated_at = excluded.generated_at
            """,
            (
                ticker,
                model_used,
                signal_date_text,
                _encode_json(forecast),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        connection.commit()


def save_backtest(
    ticker: str,
    model_used: str,
    parameters: dict[str, Any],
    metrics: dict[str, Any],
    trades: list[dict[str, Any]],
    retrainings: list[dict[str, Any]],
) -> int:
    """Persist a complete backtest run and return its identifier."""
    ticker = validate_ticker(ticker)
    with state_connection() as connection:
        cursor = connection.execute(
            "INSERT INTO backtests "
            "(ticker, model_used, parameters_json, metrics_json, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                ticker,
                model_used,
                _encode_json(parameters),
                _encode_json(metrics),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        backtest_id = int(cursor.lastrowid)
        connection.executemany(
            "INSERT INTO backtest_trades "
            "(backtest_id, sequence_index, content_json) VALUES (?, ?, ?)",
            [
                (backtest_id, sequence_index, _encode_json(item))
                for sequence_index, item in enumerate(trades)
            ],
        )
        connection.executemany(
            "INSERT INTO backtest_retrainings "
            "(backtest_id, sequence_index, content_json) VALUES (?, ?, ?)",
            [
                (backtest_id, sequence_index, _encode_json(item))
                for sequence_index, item in enumerate(retrainings)
            ],
        )
        connection.commit()
    return backtest_id


def list_persisted_data(ticker: str) -> dict[str, Any]:
    ticker = validate_ticker(ticker)
    with state_connection() as connection:
        rows = connection.execute(
            "SELECT * FROM daily_forecasts WHERE ticker = ? ORDER BY target_date",
            (ticker,),
        ).fetchall()
        current_forecast_row = connection.execute(
            "SELECT content_json, generated_at FROM current_forecasts WHERE ticker = ?",
            (ticker,),
        ).fetchone()
        backtest_rows = connection.execute(
            """
            SELECT id, model_used, parameters_json, metrics_json, created_at,
                   (SELECT COUNT(*) FROM backtest_trades o
                    WHERE o.backtest_id = b.id) AS total_trades,
                   (SELECT COUNT(*) FROM backtest_retrainings r
                    WHERE r.backtest_id = b.id) AS total_retrainings
            FROM backtests b
            WHERE ticker = ?
            ORDER BY created_at DESC, id DESC
            """,
            (ticker,),
        ).fetchall()
    forecasts = []
    for row in rows:
        item = dict(row)
        item["target_date"] = date.fromisoformat(item["target_date"])
        item["base_close_date"] = date.fromisoformat(item["base_close_date"])
        if item.get("expected_update_date"):
            item["expected_update_date"] = date.fromisoformat(
                item["expected_update_date"]
            )
        item["generated_at"] = datetime.fromisoformat(item["generated_at"])
        forecasts.append(item)
    current_forecast_value = None
    if current_forecast_row:
        current_forecast_value = json.loads(current_forecast_row["content_json"])
        current_forecast_value["generated_at"] = datetime.fromisoformat(
            current_forecast_row["generated_at"]
        )
    backtests = [
        {
            "id": int(row["id"]),
            "model_used": str(row["model_used"]),
            "parameters": json.loads(row["parameters_json"]),
            "metrics": json.loads(row["metrics_json"]),
            "total_trades": int(row["total_trades"]),
            "total_retrainings": int(row["total_retrainings"]),
            "created_at": datetime.fromisoformat(row["created_at"]),
        }
        for row in backtest_rows
    ]
    return {
        "ticker": ticker,
        "position": get_position_state(ticker).to_dict(),
        "current_forecast": current_forecast_value,
        "daily_forecasts": forecasts,
        "backtests": backtests,
    }

__all__ = [
    "get_position_state",
    "list_persisted_data",
    "list_trades",
    "register_trade",
    "reset_trades",
    "save_backtest",
    "save_current_forecast",
    "save_daily_forecasts",
    "validate_ticker",
]
