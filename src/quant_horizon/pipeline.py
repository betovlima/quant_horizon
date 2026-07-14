"""
Walk-forward pipeline for market-direction forecasting and investment research.

Core guarantees:
  1. A signal at t uses only information available at the close of t.
  2. Entry occurs at the open of t+1 and exit at the close of t+h.
  3. Up, down, and neutral samples remain in the three-class dataset.
  4. Temporal purging prevents training labels from reaching the test period.
  5. The trade grid is global and never restarts after model retraining.
  6. Entry and exit costs are charged for every position.
  7. Buy-and-hold is calculated directly from entry and exit prices.
  8. The confidence threshold is selected using training data only.
  9. Current forecasts use the latest close without requiring a future label.

Dependencies:
    poetry install

Exemplo:
    poetry run python -m quant_horizon.pipeline \
        --ticker PETR4.SA \
        --start 2010-01-01 \
        --simulate_from 2015-01-01 \
        --horizon 5 \
        --initial_capital 100

Latest forecast:
    poetry run python -m quant_horizon.pipeline \
        --ticker AAPL \
        --start 2010-01-01 \
        --current_forecast_only

Warning: this is a research tool. It does not provide financial advice and does
not replace independent validation, taxes, or actual brokerage execution rules.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import sqlite3
from contextlib import contextmanager
from collections.abc import Iterator
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from .investment_models import create_model, normalize_model_name


CLASSES = np.array([-1, 0, 1], dtype=int)  # down, neutral, up
HORIZON_FACTORS = {"daily": 1, "weekly": 5}
CACHE_VERSION = "3"


def _cache_enabled() -> bool:
    return os.environ.get("QUANT_HORIZON_CACHE_ENABLED", "1").strip().lower() not in {
        "0",
        "false",
        "no",
    }


def _cache_db_path() -> Path:
    return Path(
        os.environ.get("QUANT_HORIZON_CACHE_DB", "db/quant_horizon_cache.sqlite3")
    )


def currency_symbol(ticker: str) -> str:
    return "R$" if ticker.upper().endswith(".SA") else "US$"


def horizon_in_trading_days(quantity: int, unit: str = "daily") -> int:
    """Convert a daily or weekly horizon into daily trading bars."""
    if not isinstance(quantity, (int, np.integer)) or quantity < 1:
        raise ValueError("The horizon quantity must be a positive integer.")
    if unit not in HORIZON_FACTORS:
        options = ", ".join(HORIZON_FACTORS)
        raise ValueError(f"Invalid horizon unit. Choose one of: {options}.")
    return int(quantity) * HORIZON_FACTORS[unit]


# -----------------------------------------------------------------------------
# 1. MARKET DATA
# -----------------------------------------------------------------------------
def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    """Normalize remote responses and values recovered from cache."""
    columns_list = ["open", "high", "low", "close", "volume"]
    if df.empty:
        return pd.DataFrame(columns=columns_list)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df = df.rename(columns=str.lower)
    missing = [c for c in columns_list if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in downloaded data: {missing}")
    normalized = df[columns_list].copy()
    normalized.index = pd.to_datetime(normalized.index).tz_localize(None)
    normalized = normalized[~normalized.index.duplicated(keep="last")].sort_index()
    return normalized.replace([np.inf, -np.inf], np.nan).dropna()


def _download_remote_data(
    ticker: str,
    start: str,
    end: str | None,
) -> pd.DataFrame:
    import yfinance as yf

    return _normalize_ohlcv(
        yf.download(
            ticker,
            start=start,
            end=end,
            auto_adjust=True,
            progress=False,
            actions=False,
        )
    )


def _connect_price_cache() -> sqlite3.Connection:
    path = _cache_db_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(path, timeout=20)
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS ohlcv_prices (
            ticker TEXT NOT NULL,
            date TEXT NOT NULL,
            open REAL NOT NULL,
            high REAL NOT NULL,
            low REAL NOT NULL,
            close REAL NOT NULL,
            volume REAL NOT NULL,
            PRIMARY KEY (ticker, date)
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS price_cache_state (
            ticker TEXT PRIMARY KEY,
            exclusive_end_coverage TEXT,
            last_query TEXT,
            last_full_refresh TEXT
        )
        """
    )
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS calculation_cache (
            ticker TEXT NOT NULL,
            category TEXT NOT NULL,
            version TEXT NOT NULL,
            signature TEXT NOT NULL,
            content BLOB NOT NULL,
            created_at TEXT NOT NULL,
            last_access TEXT NOT NULL,
            PRIMARY KEY (ticker, category, version, signature)
        )
        """
    )
    connection.execute(
        "CREATE INDEX IF NOT EXISTS idx_calculation_cache_access "
        "ON calculation_cache (ticker, category, version, last_access DESC)"
    )
    connection.commit()
    return connection


@contextmanager
def _price_cache_connection() -> Iterator[sqlite3.Connection]:
    """Yield a transactional cache connection and always close it."""
    connection = _connect_price_cache()
    try:
        with connection:
            yield connection
    finally:
        connection.close()


def _read_cached_prices(
    ticker: str,
    start_value: date,
    exclusive_end: date | None,
) -> pd.DataFrame:
    query = (
        "SELECT date, open, high, low, close, volume FROM ohlcv_prices "
        "WHERE ticker = ? AND date >= ?"
    )
    parameters: list[object] = [ticker, start_value.isoformat()]
    if exclusive_end is not None:
        query += " AND date < ?"
        parameters.append(exclusive_end.isoformat())
    query += " ORDER BY date"
    with _price_cache_connection() as connection:
        rows = connection.execute(query, parameters).fetchall()
    if not rows:
        return _normalize_ohlcv(pd.DataFrame())
    df = pd.DataFrame(
        rows,
        columns=["date", "open", "high", "low", "close", "volume"],
    ).set_index("date")
    return _normalize_ohlcv(df)


def _save_cached_prices(ticker: str, df: pd.DataFrame) -> None:
    if df.empty:
        return
    rows = [
        (
            ticker,
            pd.Timestamp(index_value).date().isoformat(),
            float(row.open),
            float(row.high),
            float(row.low),
            float(row.close),
            float(row.volume),
        )
        for index_value, row in df.iterrows()
    ]
    with _price_cache_connection() as connection:
        connection.executemany(
            """
            INSERT INTO ohlcv_prices
                (ticker, date, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, date) DO UPDATE SET
                open = excluded.open,
                high = excluded.high,
                low = excluded.low,
                close = excluded.close,
                volume = excluded.volume
            """,
            rows,
        )
        connection.commit()


def _price_cache_state(ticker: str) -> dict[str, date | None]:
    with _price_cache_connection() as connection:
        row = connection.execute(
            "SELECT exclusive_end_coverage, last_query, "
            "last_full_refresh FROM price_cache_state WHERE ticker = ?",
            (ticker,),
        ).fetchone()
    if not row:
        return {"coverage": None, "query": None, "full_refresh": None}
    return {
        "coverage": date.fromisoformat(row[0]) if row[0] else None,
        "query": date.fromisoformat(row[1]) if row[1] else None,
        "full_refresh": date.fromisoformat(row[2]) if row[2] else None,
    }


def _save_price_cache_state(
    ticker: str,
    coverage: date,
    query: date,
    full_refresh: date | None = None,
) -> None:
    previous = _price_cache_state(ticker)
    previous_coverage = previous["coverage"]
    final_coverage = max(coverage, previous_coverage) if previous_coverage else coverage
    final_full_refresh = full_refresh or previous["full_refresh"]
    with _price_cache_connection() as connection:
        connection.execute(
            """
            INSERT INTO price_cache_state
                (ticker, exclusive_end_coverage, last_query,
                 last_full_refresh)
            VALUES (?, ?, ?, ?)
            ON CONFLICT(ticker) DO UPDATE SET
                exclusive_end_coverage = excluded.exclusive_end_coverage,
                last_query = excluded.last_query,
                last_full_refresh = excluded.last_full_refresh
            """,
            (
                ticker,
                final_coverage.isoformat(),
                query.isoformat(),
                final_full_refresh.isoformat() if final_full_refresh else None,
            ),
        )
        connection.commit()


def download_data(ticker: str, start: str, end: str | None = None) -> pd.DataFrame:
    """Return adjusted OHLCV data while updating the local SQLite cache."""
    ticker = ticker.strip().upper()
    start_value = pd.Timestamp(start).date()
    exclusive_end = pd.Timestamp(end).date() if end else None
    if exclusive_end is not None and exclusive_end <= start_value:
        raise ValueError("end must be later than start.")

    if not _cache_enabled():
        df = _download_remote_data(ticker, start_value.isoformat(), end)
        if df.empty:
            raise ValueError("No data was downloaded. Check the ticker and dates.")
        return df

    today_date = date.today()
    desired_coverage = exclusive_end or (today_date + timedelta(days=1))
    existing = _read_cached_prices(ticker, start_value, exclusive_end)
    state = _price_cache_state(ticker)
    last_full_refresh = state["full_refresh"]
    empty_cache = existing.empty
    full_refresh_expired = (
        exclusive_end is None
        and last_full_refresh is not None
        and (today_date - last_full_refresh).days >= 7
    )

    if empty_cache or full_refresh_expired:
        remote = _download_remote_data(ticker, start_value.isoformat(), end)
        _save_cached_prices(ticker, remote)
        _save_price_cache_state(
            ticker,
            coverage=desired_coverage,
            query=today_date,
            full_refresh=today_date,
        )
    else:
        earliest_date = pd.Timestamp(existing.index.min()).date()
        latest_date = pd.Timestamp(existing.index.max()).date()

        if earliest_date > start_value:
            previous = _download_remote_data(
                ticker,
                start_value.isoformat(),
                earliest_date.isoformat(),
            )
            _save_cached_prices(ticker, previous)

        needs_end_update = (
            state["coverage"] is None
            or state["coverage"] < desired_coverage
        )
        if needs_end_update:
            # Overlap refreshes recent revisions published by the data provider.
            incremental_start = max(start_value, latest_date - timedelta(days=7))
            recent = _download_remote_data(
                ticker,
                incremental_start.isoformat(),
                end,
            )
            _save_cached_prices(ticker, recent)
            _save_price_cache_state(
                ticker,
                coverage=desired_coverage,
                query=today_date,
            )

    df = _read_cached_prices(ticker, start_value, exclusive_end)
    if df.empty:
        raise ValueError("No data was downloaded. Check the ticker and dates.")
    if (df[["open", "high", "low", "close"]] <= 0).any().any():
        raise ValueError("Prices must be greater than zero.")
    return df


# -----------------------------------------------------------------------------
# 2. FEATURES
# -----------------------------------------------------------------------------
def _dataframe_signature(df: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update(CACHE_VERSION.encode("utf-8"))
    digest.update("|".join(map(str, df.columns)).encode("utf-8"))
    digest.update("|".join(map(str, df.dtypes)).encode("utf-8"))
    digest.update(pd.util.hash_pandas_object(df, index=True).values.tobytes())
    return digest.hexdigest()[:24]


def _read_calculation_cache(
    ticker: str,
    category: str,
    signature: str,
) -> pd.DataFrame | None:
    """Read serialized features or targets from the SQLite cache."""
    ticker = ticker.strip().upper()
    with _price_cache_connection() as connection:
        row = connection.execute(
            "SELECT content FROM calculation_cache "
            "WHERE ticker = ? AND category = ? AND version = ? AND signature = ?",
            (ticker, category, CACHE_VERSION, signature),
        ).fetchone()
        if row is None:
            return None
        connection.execute(
            "UPDATE calculation_cache SET last_access = ? "
            "WHERE ticker = ? AND category = ? AND version = ? AND signature = ?",
            (
                datetime.now(timezone.utc).isoformat(),
                ticker,
                category,
                CACHE_VERSION,
                signature,
            ),
        )
        connection.commit()

    try:
        value = pickle.loads(row[0])
    except (pickle.UnpicklingError, EOFError, AttributeError, TypeError, ValueError):
        return None
    return value if isinstance(value, pd.DataFrame) else None


def _save_calculation_cache(
    ticker: str,
    category: str,
    signature: str,
    value: pd.DataFrame,
) -> None:
    """Store a calculation as a BLOB and retain five versions per category."""
    ticker = ticker.strip().upper()
    timestamp = datetime.now(timezone.utc).isoformat()
    content = sqlite3.Binary(pickle.dumps(value, protocol=pickle.HIGHEST_PROTOCOL))
    with _price_cache_connection() as connection:
        connection.execute(
            """
            INSERT INTO calculation_cache (
                ticker, category, version, signature,
                content, created_at, last_access
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(ticker, category, version, signature) DO UPDATE SET
                content = excluded.content,
                last_access = excluded.last_access
            """,
            (
                ticker,
                category,
                CACHE_VERSION,
                signature,
                content,
                timestamp,
                timestamp,
            ),
        )
        old_entries = connection.execute(
            "SELECT signature FROM calculation_cache "
            "WHERE ticker = ? AND category = ? AND version = ? "
            "ORDER BY last_access DESC LIMIT -1 OFFSET 5",
            (ticker, category, CACHE_VERSION),
        ).fetchall()
        connection.executemany(
            "DELETE FROM calculation_cache "
            "WHERE ticker = ? AND category = ? AND version = ? AND signature = ?",
            [
                (ticker, category, CACHE_VERSION, row[0])
                for row in old_entries
            ],
        )
        connection.commit()


def _rsi_wilder(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    average_gain = gain.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    average_loss = loss.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()
    rs = average_gain / average_loss.replace(0, np.nan)
    rsi = 100 - 100 / (1 + rs)
    return rsi.fillna(100.0).where(average_gain.ne(0) | average_loss.ne(0), 50.0)


def _macd_hist(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    fast_ema = series.ewm(span=fast, adjust=False).mean()
    slow_ema = series.ewm(span=slow, adjust=False).mean()
    row = fast_ema - slow_ema
    return row - row.ewm(span=signal, adjust=False).mean()


def _atr_wilder(df: pd.DataFrame, period: int = 14) -> pd.Series:
    previous_close = df["close"].shift(1)
    true_range = pd.concat(
        [
            df["high"] - df["low"],
            (df["high"] - previous_close).abs(),
            (df["low"] - previous_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return true_range.ewm(alpha=1 / period, adjust=False, min_periods=period).mean()


def _calculate_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create features known at the end of each trading session."""
    close = df["close"]
    volume = df["volume"]
    features = pd.DataFrame(index=df.index)

    features["ret_1"] = close.pct_change(1, fill_method=None)
    features["ret_5"] = close.pct_change(5, fill_method=None)
    features["ret_10"] = close.pct_change(10, fill_method=None)
    features["ret_20"] = close.pct_change(20, fill_method=None)

    for window in (10, 20, 50):
        features[f"dist_ma_{window}"] = close / close.rolling(window).mean() - 1

    features["vol_10"] = features["ret_1"].rolling(10).std()
    features["vol_20"] = features["ret_1"].rolling(20).std()
    features["atr_14"] = _atr_wilder(df, 14) / close
    features["rsi_14"] = _rsi_wilder(close, 14) / 100.0
    features["macd_hist_rel"] = _macd_hist(close) / close

    mean_20 = close.rolling(20).mean()
    deviation_20 = close.rolling(20).std()
    features["bb_pos"] = (close - mean_20) / (2 * deviation_20.replace(0, np.nan))

    features["vol_chg_5"] = volume.pct_change(5, fill_method=None)
    features["vol_rel_20"] = volume / volume.rolling(20).mean() - 1

    # Simple regime features that are also known at t.
    features["tendencia_50_200"] = close.rolling(50).mean() / close.rolling(200).mean() - 1
    features["drawdown_252"] = close / close.rolling(252, min_periods=60).max() - 1

    return features.replace([np.inf, -np.inf], np.nan)


def build_features(
    df: pd.DataFrame,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Build features known at time t and reuse the signature-based cache."""
    if not ticker or not _cache_enabled():
        return _calculate_features(df)
    signature = _dataframe_signature(df)
    cached = _read_calculation_cache(ticker, "features", signature)
    if cached is not None:
        return cached
    calculated = _calculate_features(df)
    _save_calculation_cache(ticker, "features", signature, calculated)
    return calculated


# -----------------------------------------------------------------------------
# 3. EXECUTABLE TARGET: SIGNAL AT t, ENTRY AT t+1, EXIT AT t+h
# -----------------------------------------------------------------------------
def _calculate_targets(
    df: pd.DataFrame,
    horizon: int = 5,
    side_cost: float = 0.0005,
    safety_margin: float = 0.0005,
) -> pd.DataFrame:
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    if side_cost < 0 or safety_margin < 0:
        raise ValueError("Costs and safety margin cannot be negative.")

    target = pd.DataFrame(index=df.index)
    target["entry_price"] = df["open"].shift(-1)
    target["exit_price"] = df["close"].shift(-horizon)
    target["gross_return"] = target["exit_price"] / target["entry_price"] - 1

    dates = pd.Series(df.index, index=df.index)
    target["entry_date"] = dates.shift(-1)
    target["exit_date"] = dates.shift(-horizon)

    # Round-trip cost contributes to the neutral-zone boundary.
    barrier = 2 * side_cost + safety_margin
    valid = target["gross_return"].notna()
    label = pd.Series(pd.NA, index=df.index, dtype="Int64")
    label.loc[valid] = 0
    label.loc[valid & (target["gross_return"] > barrier)] = 1
    label.loc[valid & (target["gross_return"] < -barrier)] = -1
    target["label"] = label
    return target


def build_targets(
    df: pd.DataFrame,
    horizon: int = 5,
    side_cost: float = 0.0005,
    safety_margin: float = 0.0005,
    ticker: str | None = None,
) -> pd.DataFrame:
    """Create executable targets and reuse identical cached results."""
    if not ticker or not _cache_enabled():
        return _calculate_targets(df, horizon, side_cost, safety_margin)
    parameters = f"{horizon}|{side_cost:.12g}|{safety_margin:.12g}"
    signature = hashlib.sha256(
        f"{_dataframe_signature(df)}|{parameters}".encode("utf-8")
    ).hexdigest()[:24]
    cached = _read_calculation_cache(ticker, "targets", signature)
    if cached is not None:
        return cached
    calculated = _calculate_targets(df, horizon, side_cost, safety_margin)
    _save_calculation_cache(ticker, "targets", signature, calculated)
    return calculated


def build_dataset(features: pd.DataFrame, target: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove warm-up and unlabeled tail rows while preserving neutral days."""
    target_columns = [
        "label",
        "gross_return",
        "entry_price",
        "exit_price",
        "entry_date",
        "exit_date",
    ]
    combined = features.join(target[target_columns], how="inner")
    combined = combined.replace([np.inf, -np.inf], np.nan).dropna()

    X = combined[features.columns].astype(float)
    events = combined[target_columns].copy()
    events["label"] = events["label"].astype(int)

    if len(X) < 300:
        raise ValueError(
            f"Only {len(X)} valid rows are available. Use a longer history."
        )
    return X, events


# -----------------------------------------------------------------------------
# 4. MODEL, PROBABILITIES, AND THRESHOLD
# -----------------------------------------------------------------------------
def aligned_probabilities(model, X: pd.DataFrame) -> np.ndarray:
    """Sempre devolve colunas na order [-1, 0, 1]."""
    model_probabilities = model.predict_proba(X)
    proba = np.zeros((len(X), len(CLASSES)), dtype=float)
    mapping = {int(class_value): i for i, class_value in enumerate(model.classes_)}
    for destination, class_value in enumerate(CLASSES):
        if int(class_value) in mapping:
            proba[:, destination] = model_probabilities[:, mapping[int(class_value)]]
    return proba


def positions_by_threshold(
    proba: np.ndarray,
    threshold: float,
    position_mode: str,
) -> np.ndarray:
    if position_mode not in {"long_short", "long_flat"}:
        raise ValueError("position_mode must be long_short or long_flat.")

    probability_down = proba[:, 0]
    probability_up = proba[:, 2]
    position = np.zeros(len(proba), dtype=int)

    buy = (probability_up >= threshold) & (probability_up > probability_down)
    position[buy] = 1

    if position_mode == "long_short":
        sell = (probability_down >= threshold) & (probability_down > probability_up)
        position[sell] = -1

    return position


def _financial_score(
    position: np.ndarray,
    gross_return: np.ndarray,
    side_cost: float,
    annual_short_cost: float,
    horizon: int,
) -> tuple[float, int]:
    active = position != 0
    costs = np.where(active, 2 * side_cost, 0.0)
    borrowing = np.where(
        position < 0,
        annual_short_cost * horizon / 252.0,
        0.0,
    )
    returns = position * gross_return - costs - borrowing
    active_returns = returns[active]
    if len(active_returns) < 10:
        return -np.inf, len(active_returns)
    deviation = active_returns.std(ddof=1)
    if not np.isfinite(deviation) or deviation <= 0:
        return -np.inf, len(active_returns)
    sharpe = active_returns.mean() / deviation * np.sqrt(252 / horizon)
    return float(sharpe), len(active_returns)


def train_with_internal_threshold(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    training_return: pd.Series,
    horizon: int,
    side_cost: float,
    annual_short_cost: float,
    position_mode: str,
    default_threshold: float,
    optimize_threshold: bool,
    model_name: str | None = None,
):
    """Select a threshold on the training tail, then refit on all training data."""
    selected_threshold = default_threshold

    if optimize_threshold and len(X_train) >= 500:
        validation_start = int(len(X_train) * 0.80)
        subtrain_end = validation_start - horizon

        if subtrain_end >= 250 and len(X_train) - validation_start >= 50:
            internal_model = create_model(
                random_state=41,
                model_name=model_name,
            )
            internal_model.fit(
                X_train.iloc[:subtrain_end],
                y_train.iloc[:subtrain_end],
            )
            X_val = X_train.iloc[validation_start:]
            validation_returns = training_return.iloc[validation_start:].to_numpy()
            validation_probabilities = aligned_probabilities(internal_model, X_val)

            # Non-overlapping evaluation within internal validation.
            selection = np.arange(0, len(X_val), horizon)
            candidates = np.arange(0.40, 0.71, 0.05)
            best_score = -np.inf
            for candidate in candidates:
                pos = positions_by_threshold(validation_probabilities, float(candidate), position_mode)
                score, _ = _financial_score(
                    pos[selection],
                    validation_returns[selection],
                    side_cost,
                    annual_short_cost,
                    horizon,
                )
                if score > best_score:
                    best_score = score
                    selected_threshold = float(candidate)

            # Keep the default if the candidate has no positive adjusted return.
            if not np.isfinite(best_score) or best_score <= 0:
                selected_threshold = default_threshold

    model = create_model(random_state=42, model_name=model_name)
    model.fit(X_train, y_train)
    return model, selected_threshold


# -----------------------------------------------------------------------------
# 5. WALK-FORWARD CLASSIFICATION METRICS
# -----------------------------------------------------------------------------
def evaluate_walk_forward(
    X: pd.DataFrame,
    events: pd.DataFrame,
    n_splits: int,
    horizon: int,
    side_cost: float,
    annual_short_cost: float,
    position_mode: str,
    threshold: float,
    optimize_threshold: bool,
    model_name: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series]:
    from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss, roc_auc_score
    from sklearn.model_selection import TimeSeriesSplit

    y = events["label"]
    splitter = TimeSeriesSplit(n_splits=n_splits, gap=horizon)
    results: list[dict] = []
    forecasts: list[pd.DataFrame] = []
    importances: list[pd.Series] = []

    for fold, (train_idx, test_idx) in enumerate(splitter.split(X), start=1):
        X_train = X.iloc[train_idx]
        y_train = y.iloc[train_idx]
        X_test = X.iloc[test_idx]
        y_test = y.iloc[test_idx]

        model, fold_threshold = train_with_internal_threshold(
            X_train=X_train,
            y_train=y_train,
            training_return=events["gross_return"].iloc[train_idx],
            horizon=horizon,
            side_cost=side_cost,
            annual_short_cost=annual_short_cost,
            position_mode=position_mode,
            default_threshold=threshold,
            optimize_threshold=optimize_threshold,
            model_name=model_name,
        )

        proba = aligned_probabilities(model, X_test)
        predicted_class = CLASSES[np.argmax(proba, axis=1)]
        position = positions_by_threshold(proba, fold_threshold, position_mode)
        one_hot = (y_test.to_numpy()[:, None] == CLASSES[None, :]).astype(float)
        multiclass_brier = np.mean(np.sum((proba - one_hot) ** 2, axis=1))

        try:
            auc = roc_auc_score(
                y_test,
                proba,
                labels=CLASSES,
                multi_class="ovr",
                average="macro",
            )
        except ValueError:
            auc = np.nan

        results.append(
            {
                "fold": fold,
                "test_start": X_test.index[0],
                "test_end": X_test.index[-1],
                "n_train": len(X_train),
                "n_test": len(X_test),
                "threshold": fold_threshold,
                "accuracy": accuracy_score(y_test, predicted_class),
                "balanced_accuracy": balanced_accuracy_score(y_test, predicted_class),
                "auc_ovr_macro": auc,
                "log_loss": log_loss(y_test, proba, labels=CLASSES),
                "multiclass_brier": multiclass_brier,
                "position_exposure_ratio": np.mean(position != 0),
            }
        )

        forecasts.append(
            pd.DataFrame(
                {
                    "fold": fold,
                    "label": y_test,
                    "predicted_class": predicted_class,
                    "probability_down": proba[:, 0],
                    "probability_neutral": proba[:, 1],
                    "probability_up": proba[:, 2],
                    "position": position,
                    "gross_return": events["gross_return"].iloc[test_idx],
                },
                index=X_test.index,
            )
        )

        gain = pd.Series(model.feature_importances_, index=X.columns, dtype=float)
        if gain.sum() > 0:
            gain /= gain.sum()
        importances.append(gain)

        r = results[-1]
        print(
            f"[Fold {fold}] train={len(X_train):4d} test={len(X_test):4d} "
            f"threshold={fold_threshold:.2f} | balanced_acc={r['balanced_accuracy']:.3f} "
            f"auc={r['auc_ovr_macro']:.3f} logloss={r['log_loss']:.3f}"
        )

    average_importance = pd.concat(importances, axis=1).mean(axis=1).sort_values(ascending=False)
    return pd.DataFrame(results), pd.concat(forecasts).sort_index(), average_importance


# -----------------------------------------------------------------------------
# 5B. DAILY HISTORICAL SIGNALS WITH PERIODIC RETRAINING
# -----------------------------------------------------------------------------
def predict_daily_historical_signals(
    features: pd.DataFrame,
    X_history: pd.DataFrame,
    historical_events: pd.DataFrame,
    full_target: pd.DataFrame,
    start_date: str | pd.Timestamp,
    end_date: str | pd.Timestamp,
    horizon: int,
    side_cost: float,
    annual_short_cost: float,
    training_window_days: int,
    position_mode: str,
    threshold: float,
    optimize_threshold: bool,
    retrain_frequency_days: int = 21,
    model_name: str | None = None,
    minimum_training_samples: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Predict every close in a range without spacing signals by the horizon.

    Unlike the financial backtest, outcomes may overlap: each trading session
    receives its own forecast and is compared with the return observed over the
    requested horizon. The model is reused within each retraining block and
    uses only labels completed before the first signal in that block.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1.")
    if retrain_frequency_days < 1:
        raise ValueError("retrain_frequency_days must be at least 1.")

    valid_features = features.replace([np.inf, -np.inf], np.nan).dropna()
    start_value = pd.Timestamp(start_date)
    end_value = pd.Timestamp(end_date)
    signal_dates = valid_features.index[
        (valid_features.index >= start_value) & (valid_features.index <= end_value)
    ]
    if len(signal_dates) == 0:
        return pd.DataFrame(), pd.DataFrame()

    rows: list[dict] = []
    retraining_log: list[dict] = []
    block_start = 0
    while block_start < len(signal_dates):
        block_end = min(block_start + retrain_frequency_days, len(signal_dates))
        block_dates = signal_dates[block_start:block_end]
        retraining_date = pd.Timestamp(block_dates[0])

        known = pd.to_datetime(historical_events["exit_date"]) <= retraining_date
        X_train = X_history.loc[known]
        training_events = historical_events.loc[X_train.index]
        if training_window_days > 0:
            X_train = X_train.iloc[-training_window_days:]
            training_events = training_events.loc[X_train.index]

        if len(X_train) < minimum_training_samples:
            raise ValueError(
                f"Only {len(X_train)} known samples exist before "
                f"{retraining_date.date()}. Use older data or analyze a later period."
            )
        if training_events["label"].nunique() < 2:
            raise ValueError("Training history contains fewer than two classes.")

        model, current_threshold = train_with_internal_threshold(
            X_train=X_train,
            y_train=training_events["label"],
            training_return=training_events["gross_return"],
            horizon=horizon,
            side_cost=side_cost,
            annual_short_cost=annual_short_cost,
            position_mode=position_mode,
            default_threshold=threshold,
            optimize_threshold=optimize_threshold,
            model_name=model_name,
        )
        retraining_log.append(
            {
                "retrained_at": retraining_date,
                "training_start": X_train.index[0],
                "training_end": X_train.index[-1],
                "training_samples": len(X_train),
                "threshold": current_threshold,
                "model_used": normalize_model_name(model_name),
            }
        )

        X_block = valid_features.loc[block_dates].reindex(columns=X_history.columns)
        probabilities = aligned_probabilities(model, X_block)
        positions = positions_by_threshold(probabilities, current_threshold, position_mode)
        predicted_classes = CLASSES[np.argmax(probabilities, axis=1)]

        for offset, signal_date in enumerate(block_dates):
            proba = probabilities[offset]
            position = int(positions[offset])
            predicted_class = int(predicted_classes[offset])
            day_target = full_target.loc[signal_date]
            result_available = bool(pd.notna(day_target["label"]))
            actual_class = int(day_target["label"]) if result_available else None

            rows.append(
                {
                    "signal_date": pd.Timestamp(signal_date),
                    "probability_down": float(proba[0]),
                    "probability_neutral": float(proba[1]),
                    "probability_up": float(proba[2]),
                    "threshold": float(current_threshold),
                    "position": position,
                    "action": {1: "BUY", -1: "SELL", 0: "WAIT"}[position],
                    "predicted_class": predicted_class,
                    "result_available": result_available,
                    "entry_date": (
                        pd.Timestamp(day_target["entry_date"])
                        if pd.notna(day_target["entry_date"])
                        else None
                    ),
                    "entry_price": (
                        float(day_target["entry_price"])
                        if pd.notna(day_target["entry_price"])
                        else None
                    ),
                    "exit_date": (
                        pd.Timestamp(day_target["exit_date"])
                        if pd.notna(day_target["exit_date"])
                        else None
                    ),
                    "exit_price": (
                        float(day_target["exit_price"])
                        if pd.notna(day_target["exit_price"])
                        else None
                    ),
                    "observed_return": (
                        float(day_target["gross_return"])
                        if pd.notna(day_target["gross_return"])
                        else None
                    ),
                    "actual_class": actual_class,
                    "is_correct": (
                        predicted_class == actual_class
                        if actual_class is not None
                        else None
                    ),
                    "training_samples": len(X_train),
                }
            )

        block_start = block_end

    return pd.DataFrame(rows), pd.DataFrame(retraining_log)


# -----------------------------------------------------------------------------
# 6. GLOBAL NON-OVERLAPPING SIMULATION
# -----------------------------------------------------------------------------
def simulate_investment(
    X: pd.DataFrame,
    events: pd.DataFrame,
    start_date: str,
    horizon: int,
    side_cost: float,
    annual_short_cost: float,
    initial_capital: float,
    retrain_frequency_days: int,
    training_window_days: int,
    position_mode: str,
    threshold: float,
    optimize_threshold: bool,
    model_name: str | None = None,
    minimum_training_samples: int = 252,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if initial_capital <= 0:
        raise ValueError("initial_capital must be positive.")
    if retrain_frequency_days < 1:
        raise ValueError("retrain_frequency_days must be at least 1.")

    cutoff = pd.Timestamp(start_date)
    start_value = int(X.index.searchsorted(cutoff))
    if start_value >= len(X):
        raise ValueError(f"There is no valid data on or after {start_date}.")

    # One global grid; it does not restart after model retraining.
    signal_indices = np.arange(start_value, len(X), horizon, dtype=int)
    capital = float(initial_capital)
    first_buy_hold_price = float(events["entry_price"].iloc[signal_indices[0]])
    rows: list[dict] = []
    retraining_log: list[dict] = []
    block_start = 0
    while block_start < len(signal_indices):
        retraining_position = int(signal_indices[block_start])

        # At the close at pos, the latest known label is pos-horizon.
        exclusive_training_end = retraining_position - horizon + 1
        if exclusive_training_end < minimum_training_samples:
            raise ValueError(
                f"Only {exclusive_training_end} known samples exist before "
                f"{X.index[retraining_position].date()}. Use older data or simulate from a later date."
            )

        training_start = 0
        if training_window_days > 0:
            training_start = max(0, exclusive_training_end - training_window_days)

        X_train = X.iloc[training_start:exclusive_training_end]
        y_train = events["label"].iloc[training_start:exclusive_training_end]
        training_return = events["gross_return"].iloc[training_start:exclusive_training_end]

        model, current_threshold = train_with_internal_threshold(
            X_train=X_train,
            y_train=y_train,
            training_return=training_return,
            horizon=horizon,
            side_cost=side_cost,
            annual_short_cost=annual_short_cost,
            position_mode=position_mode,
            default_threshold=threshold,
            optimize_threshold=optimize_threshold,
            model_name=model_name,
        )

        retraining_log.append(
            {
                "retrained_at": X.index[retraining_position],
                "training_start": X_train.index[0],
                "training_end": X_train.index[-1],
                "training_samples": len(X_train),
                "threshold": current_threshold,
                "model_used": normalize_model_name(model_name),
            }
        )
        print(
            f"  Retrained on {X.index[retraining_position].date()} with {len(X_train)} rows "
            f"and threshold {current_threshold:.2f}"
        )

        retraining_limit = retraining_position + retrain_frequency_days
        block_end = int(np.searchsorted(signal_indices, retraining_limit, side="left"))
        block_indices = signal_indices[block_start:block_end]

        # A single vectorized call replaces one prediction call per event.
        block_probabilities = aligned_probabilities(
            model,
            X.iloc[block_indices],
        )
        block_positions = positions_by_threshold(
            block_probabilities,
            current_threshold,
            position_mode,
        )

        for offset, pos in enumerate(block_indices):
            pos = int(pos)
            proba = block_probabilities[offset]
            position = int(block_positions[offset])
            gross_return = float(events["gross_return"].iloc[pos])

            total_cost = 2 * side_cost if position != 0 else 0.0
            borrowing_cost = (
                annual_short_cost * horizon / 252.0 if position < 0 else 0.0
            )
            strategy_return = position * gross_return - total_cost - borrowing_cost

            if strategy_return <= -1:
                raise RuntimeError(
                    f"The strategy lost 100% or more on {X.index[pos].date()}. "
                    "Use position sizing and liquidation rules."
                )

            capital *= 1 + strategy_return
            exit_price = float(events["exit_price"].iloc[pos])
            buy_hold_capital = initial_capital * exit_price / first_buy_hold_price

            rows.append(
                {
                    "signal_date": X.index[pos],
                    "entry_date": events["entry_date"].iloc[pos],
                    "exit_date": events["exit_date"].iloc[pos],
                    "entry_price": events["entry_price"].iloc[pos],
                    "exit_price": exit_price,
                    "probability_down": proba[0],
                    "probability_neutral": proba[1],
                    "probability_up": proba[2],
                    "threshold": current_threshold,
                    "position": position,
                    "gross_asset_return": gross_return,
                    "total_cost": total_cost + borrowing_cost,
                    "strategy_return": strategy_return,
                    "strategy_capital": capital,
                    "buy_hold_capital": buy_hold_capital,
                }
            )

        block_start = block_end

    return pd.DataFrame(rows), pd.DataFrame(retraining_log)


# -----------------------------------------------------------------------------
# 7. LATEST FORECAST AND TEXT OUTPUT
# -----------------------------------------------------------------------------
def predict_latest_close(
    ticker: str,
    df: pd.DataFrame,
    features: pd.DataFrame,
    X_history: pd.DataFrame,
    historical_events: pd.DataFrame,
    horizon: int,
    side_cost: float,
    annual_short_cost: float,
    training_window_days: int,
    position_mode: str,
    threshold: float,
    optimize_threshold: bool,
    model_name: str | None = None,
    minimum_training_samples: int = 252,
) -> dict:
    """
    Train only on labels whose outcomes are known, then predict from the latest
    close features. The forecast row does not require a label.

    The signal assumes entry at the next open and use of all available capital,
    preserving the same reinvestment rule used by the backtest.
    """
    available_features = features.replace([np.inf, -np.inf], np.nan).dropna()
    if available_features.empty:
        raise ValueError("No recent row contains every required feature.")

    X_current = available_features.iloc[[-1]].reindex(columns=X_history.columns)
    signal_date = pd.Timestamp(X_current.index[0])

    # Additional safeguard: only labels completed by the analyzed close.
    known = pd.to_datetime(historical_events["exit_date"]) <= signal_date
    X_train = X_history.loc[known]
    training_events = historical_events.loc[X_train.index]

    if training_window_days > 0:
        X_train = X_train.iloc[-training_window_days:]
        training_events = training_events.loc[X_train.index]

    if len(X_train) < minimum_training_samples:
        raise ValueError(
            f"Only {len(X_train)} known samples are available for the current forecast. "
            "Use a longer historical range."
        )
    if training_events["label"].nunique() < 2:
        raise ValueError("Training history contains fewer than two classes.")

    model, current_threshold = train_with_internal_threshold(
        X_train=X_train,
        y_train=training_events["label"],
        training_return=training_events["gross_return"],
        horizon=horizon,
        side_cost=side_cost,
        annual_short_cost=annual_short_cost,
        position_mode=position_mode,
        default_threshold=threshold,
        optimize_threshold=optimize_threshold,
        model_name=model_name,
    )

    proba = aligned_probabilities(model, X_current)
    position = int(positions_by_threshold(proba, current_threshold, position_mode)[0])

    if position == 1:
        action = "BUY"
        description = "Buy at the next open using all available capital."
    elif position == -1:
        action = "SELL_SHORT"
        description = "Open a short position at the next open using all available capital."
    else:
        action = "STAY_OUT"
        description = "Do not open a position; close an existing one under the horizon rule."

    latest_dataframe_date = pd.Timestamp(df.index[-1])
    reference_price = float(df.loc[signal_date, "close"])

    return {
        "ticker": ticker,
        "model_used": normalize_model_name(model_name),
        "currency": currency_symbol(ticker),
        "signal_date": signal_date,
        "latest_price_date": latest_dataframe_date,
        "reference_price": reference_price,
        "probability_down": float(proba[0, 0]),
        "probability_neutral": float(proba[0, 1]),
        "probability_up": float(proba[0, 2]),
        "threshold": float(current_threshold),
        "target_position": position,
        "action": action,
        "description": description,
        "horizon": int(horizon),
        "capital_fraction": 1.0,
        "training_samples": int(len(X_train)),
        "training_start": pd.Timestamp(X_train.index[0]),
        "training_end": pd.Timestamp(X_train.index[-1]),
    }


def format_current_forecast(forecast: dict) -> str:
    icon = {
        "BUY": "🟢",
        "SELL_SHORT": "🔴",
        "STAY_OUT": "⚪",
    }[forecast["action"]]
    action_name = {
        "BUY": "BUY",
        "SELL_SHORT": "SELL / SHORT",
        "STAY_OUT": "STAY OUT",
    }[forecast["action"]]
    return (
        f"{icon} {action_name} SIGNAL — {forecast['ticker']}\n\n"
        f"Analyzed close: {forecast['signal_date']:%Y-%m-%d}\n"
        f"Reference price: {forecast['currency']} {forecast['reference_price']:.2f}\n\n"
        f"Down probability: {forecast['probability_down']:.2%}\n"
        f"Neutral probability: {forecast['probability_neutral']:.2%}\n"
        f"Up probability: {forecast['probability_up']:.2%}\n"
        f"Threshold: {forecast['threshold']:.2%}\n\n"
        f"Decision: {forecast['description']}\n"
        f"Horizon: {forecast['horizon']} trading days.\n"
        "The exact next-open price is not known yet.\n\n"
        "Quantitative monitoring signal; returns are not guaranteed."
    )


def save_current_forecast(forecast: dict, path: Path) -> None:
    serializable = {
        key: value.isoformat() if isinstance(value, pd.Timestamp) else value
        for key, value in forecast.items()
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(serializable, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# -----------------------------------------------------------------------------
# 8. FINANCIAL METRICS
# -----------------------------------------------------------------------------
def calculate_metrics(simulated_events: pd.DataFrame, initial_capital: float, horizon: int) -> pd.Series:
    if simulated_events.empty:
        raise ValueError("The simulation produced no events.")

    returns = simulated_events["strategy_return"].astype(float)
    capital = pd.concat(
        [pd.Series([initial_capital]), simulated_events["strategy_capital"].reset_index(drop=True)],
        ignore_index=True,
    )
    peak = capital.cummax()
    drawdown = capital / peak - 1
    years = max(
        (pd.Timestamp(simulated_events["exit_date"].iloc[-1]) - pd.Timestamp(simulated_events["entry_date"].iloc[0])).days
        / 365.25,
        1 / 365.25,
    )
    periods_per_year = 252 / horizon
    deviation = returns.std(ddof=1)
    downside = np.sqrt(np.mean(np.square(np.minimum(returns, 0))))
    active_trades = simulated_events["position"] != 0
    active_returns = returns[active_trades]

    total_return = simulated_events["strategy_capital"].iloc[-1] / initial_capital - 1
    buy_hold_total_return = simulated_events["buy_hold_capital"].iloc[-1] / initial_capital - 1
    cagr = (1 + total_return) ** (1 / years) - 1 if total_return > -1 else -1.0
    buy_hold_cagr = (1 + buy_hold_total_return) ** (1 / years) - 1 if buy_hold_total_return > -1 else -1.0

    return pd.Series(
        {
            "final_strategy_capital": simulated_events["strategy_capital"].iloc[-1],
            "final_buy_hold_capital": simulated_events["buy_hold_capital"].iloc[-1],
            "total_strategy_return": total_return,
            "total_buy_hold_return": buy_hold_total_return,
            "strategy_cagr": cagr,
            "buy_hold_cagr": buy_hold_cagr,
            "annualized_event_volatility": deviation * np.sqrt(periods_per_year),
            "approximate_sharpe": returns.mean() / deviation * np.sqrt(periods_per_year) if deviation > 0 else np.nan,
            "approximate_sortino": returns.mean() / downside * np.sqrt(periods_per_year) if downside > 0 else np.nan,
            "max_close_drawdown": drawdown.min(),
            "approximate_calmar": cagr / abs(drawdown.min()) if drawdown.min() < 0 else np.nan,
            "evaluated_signals": len(simulated_events),
            "executed_trades": int(active_trades.sum()),
            "exposure_ratio": active_trades.mean(),
            "trade_win_rate": (active_returns > 0).mean() if len(active_returns) else np.nan,
        }
    )


def print_metrics(metrics: pd.Series, currency: str = "R$") -> None:
    print("\n=== Financial result ===")
    print(f"Final strategy capital: {currency} {metrics['final_strategy_capital']:.2f}")
    print(f"Final buy-and-hold capital: {currency} {metrics['final_buy_hold_capital']:.2f}")
    print(f"Total strategy return: {metrics['total_strategy_return']:+.2%}")
    print(f"Total buy-and-hold return: {metrics['total_buy_hold_return']:+.2%}")
    print(f"Strategy CAGR: {metrics['strategy_cagr']:+.2%}")
    print(f"Buy-and-hold CAGR: {metrics['buy_hold_cagr']:+.2%}")
    print(f"Approximate Sharpe: {metrics['approximate_sharpe']:.3f}")
    print(f"Approximate Sortino: {metrics['approximate_sortino']:.3f}")
    print(f"Maximum close drawdown: {metrics['max_close_drawdown']:.2%}")
    print(f"Exposure: {metrics['exposure_ratio']:.2%}")
    print(f"Executed trades: {int(metrics['executed_trades'])}")
    print(f"Trade win rate: {metrics['trade_win_rate']:.2%}")


# -----------------------------------------------------------------------------
# 9. MAIN
# -----------------------------------------------------------------------------
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Walk-forward pipeline for market-direction research"
    )
    parser.add_argument("--ticker", default="PETR4.SA")
    parser.add_argument(
        "--start", default="2010-01-01", help="Start before the simulation date"
    )
    parser.add_argument("--end", default=None, help="Exclusive yfinance end date")
    parser.add_argument("--horizon", type=int, default=5)
    parser.add_argument(
        "--side_cost",
        type=float,
        default=0.0005,
        help="Cost per side, including fees, spread, and slippage",
    )
    parser.add_argument("--safety_margin", type=float, default=0.0005)
    parser.add_argument("--annual_short_cost", type=float, default=0.0)
    parser.add_argument("--n_splits", type=int, default=6)
    parser.add_argument("--simulate_from", default=None)
    parser.add_argument("--initial_capital", type=float, default=100.0)
    parser.add_argument("--retrain_frequency_days", type=int, default=252)
    parser.add_argument(
        "--training_window_days",
        type=int,
        default=0,
        help="0 uses expanding history; another value uses a rolling window",
    )
    parser.add_argument(
        "--position_mode",
        choices=["long_short", "long_flat"],
        default="long_flat",
    )
    parser.add_argument("--threshold", type=float, default=0.55)
    parser.add_argument(
        "--optimize_threshold",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Select the threshold through internal training validation",
    )
    parser.add_argument(
        "--current_forecast",
        action="store_true",
        help="Also forecast from the latest available close",
    )
    parser.add_argument(
        "--current_forecast_only",
        action="store_true",
        help="Skip walk-forward and simulation; run only the latest forecast",
    )
    parser.add_argument("--output_dir", default="model_outputs")
    args = parser.parse_args()

    if not 0 < args.threshold < 1:
        parser.error("--threshold must be between 0 and 1.")
    if args.n_splits < 2:
        parser.error("--n_splits must be at least 2.")

    print(f"Downloading {args.ticker}...")
    df = download_data(args.ticker, args.start, args.end)
    print(
        f"Downloaded {len(df)} trading days: "
        f"{df.index[0].date()} to {df.index[-1].date()}"
    )

    features = build_features(df, ticker=args.ticker)
    target = build_targets(
        df,
        horizon=args.horizon,
        side_cost=args.side_cost,
        safety_margin=args.safety_margin,
        ticker=args.ticker,
    )
    X, events = build_dataset(features, target)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"{len(X)} valid samples, including the neutral class.")
    print("\nClass distribution:")
    print(events["label"].value_counts(normalize=True).sort_index().rename(index={-1: "down", 0: "neutral", 1: "up"}))

    if not args.current_forecast_only:
        print("\n=== Purged walk-forward ===")
        results, oos_forecasts, importance = evaluate_walk_forward(
            X=X,
            events=events,
            n_splits=args.n_splits,
            horizon=args.horizon,
            side_cost=args.side_cost,
            annual_short_cost=args.annual_short_cost,
            position_mode=args.position_mode,
            threshold=args.threshold,
            optimize_threshold=args.optimize_threshold,
        )

        print("\nFold averages:")
        print(
            results[
                ["accuracy", "balanced_accuracy", "auc_ovr_macro", "log_loss", "multiclass_brier"]
            ].mean()
        )
        print("\nAverage gain importance:")
        print(importance)

        results.to_csv(output_dir / "walk_forward_metrics.csv", index=False)
        oos_forecasts.to_csv(output_dir / "walk_forward_forecasts.csv", index_label="data")
        importance.rename("average_importance").to_csv(
            output_dir / "feature_importance.csv"
        )

    if args.simulate_from and not args.current_forecast_only:
        print(f"\n=== Simulation from {args.simulate_from} ===")
        simulation, retraining_log = simulate_investment(
            X=X,
            events=events,
            start_date=args.simulate_from,
            horizon=args.horizon,
            side_cost=args.side_cost,
            annual_short_cost=args.annual_short_cost,
            initial_capital=args.initial_capital,
            retrain_frequency_days=args.retrain_frequency_days,
            training_window_days=args.training_window_days,
            position_mode=args.position_mode,
            threshold=args.threshold,
            optimize_threshold=args.optimize_threshold,
        )
        metrics = calculate_metrics(simulation, args.initial_capital, args.horizon)
        print_metrics(metrics, currency=currency_symbol(args.ticker))

        simulation.to_csv(output_dir / "investment_simulation.csv", index=False)
        retraining_log.to_csv(output_dir / "simulation_retraining_log.csv", index=False)
        metrics.rename("value").to_csv(output_dir / "simulation_metrics.csv")

    run_current_forecast = args.predict_current or args.current_forecast_only
    if run_current_forecast:
        print("\n=== Forecast from the latest available close ===")
        forecast = predict_latest_close(
            ticker=args.ticker,
            df=df,
            features=features,
            X_history=X,
            historical_events=events,
            horizon=args.horizon,
            side_cost=args.side_cost,
            annual_short_cost=args.annual_short_cost,
            training_window_days=args.training_window_days,
            position_mode=args.position_mode,
            threshold=args.threshold,
            optimize_threshold=args.optimize_threshold,
        )
        message = format_current_forecast(forecast)
        print(message)
        save_current_forecast(forecast, output_dir / "current_forecast.json")

    print(f"\nFiles written to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
