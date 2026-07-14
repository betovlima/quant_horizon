"""Application services for forecasts, simulated positions, and backtests."""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import date, datetime, timedelta, timezone
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import HTTPException

from . import pipeline
from .market_calendar import (
    horizon_between_close_and_target,
    next_trading_days,
    sessions_with_previous_close,
)
from .config import API_VERSION, MODEL_LOCK
from .dtos import (
    TradeAcceptanceRequest,
    DailyAnalysisRequest,
    DailyForecastRequest,
    PeriodBacktestRequest,
    BacktestRequest,
    ModelParameters,
    CurrentForecastRequest,
    HistoricalForecastRequest,
)
from .entities import TradeType
from .investment_models import normalize_model_name
from .persistence import (
    list_persisted_data,
    list_trades,
    get_position_state,
    register_trade,
    reset_trades,
    save_backtest,
    save_current_forecast,
    save_daily_forecasts,
    validate_ticker,
)


logger = logging.getLogger("services")


def normalize_json(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): normalize_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [normalize_json(v) for v in value]
    if isinstance(value, pd.Timestamp):
        return value.to_pydatetime()
    if isinstance(value, np.generic):
        return normalize_json(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def class_name(class_value: int) -> str:
    return {-1: "down", 0: "neutral", 1: "up"}[int(class_value)]

def build_data(parameters: ModelParameters):
    horizon_trading_days = parameters.horizon_trading_days
    df = pipeline.download_data(
        ticker=parameters.ticker,
        start=parameters.start.isoformat(),
        end=parameters.end.isoformat() if parameters.end else None,
    )
    features = pipeline.build_features(df, ticker=parameters.ticker)
    target = pipeline.build_targets(
        df,
        horizon=horizon_trading_days,
        side_cost=parameters.side_cost,
        safety_margin=parameters.safety_margin,
        ticker=parameters.ticker,
    )
    X, events = pipeline.build_dataset(features, target)
    return df, features, X, events


# -----------------------------------------------------------------------------
# SYNCHRONOUS SERVICES (RUN OUTSIDE THE EVENT LOOP)
# -----------------------------------------------------------------------------
def generate_forecast(parameters: CurrentForecastRequest) -> dict[str, Any]:
    horizon_trading_days = parameters.horizon_trading_days
    model_used = normalize_model_name(parameters.model)
    df, features, X, events = build_data(parameters)
    forecast = pipeline.predict_latest_close(
        ticker=parameters.ticker,
        df=df,
        features=features,
        X_history=X,
        historical_events=events,
        horizon=horizon_trading_days,
        side_cost=parameters.side_cost,
        annual_short_cost=parameters.annual_short_cost,
        training_window_days=parameters.training_window_days,
        position_mode=parameters.position_mode,
        threshold=parameters.threshold,
        optimize_threshold=parameters.optimize_threshold,
        model_name=model_used,
    )

    response = normalize_json(forecast)
    response["requested_horizon"] = parameters.horizon
    response["horizon_unit"] = parameters.horizon_unit
    response["horizon_trading_days"] = horizon_trading_days
    market_calendar, trading_dates = next_trading_days(
        ticker=parameters.ticker,
        signal_date=response["signal_date"],
        quantity=horizon_trading_days,
    )
    response["market_calendar"] = market_calendar
    response["forecast_trading_dates"] = trading_dates
    response["expected_entry_date"] = trading_dates[0]
    response["expected_exit_date"] = trading_dates[-1]
    save_current_forecast(
        ticker=parameters.ticker,
        model_used=model_used,
        forecast=response,
    )
    return response


def generate_daily_forecasts(parameters: DailyForecastRequest) -> dict[str, Any]:
    """Generate updated or preliminary daily signals without future data."""
    model_used = normalize_model_name(parameters.model)
    df = pipeline.download_data(
        ticker=parameters.ticker,
        start=parameters.start.isoformat(),
    )
    features = pipeline.build_features(df, ticker=parameters.ticker)
    market_calendar, sessions = sessions_with_previous_close(
        ticker=parameters.ticker,
        start_date=parameters.start_date,
        end_date=parameters.end_date,
    )

    dataframe_index = pd.DatetimeIndex(df.index)
    dataframe_dates = {pd.Timestamp(value).date() for value in dataframe_index}
    latest_close = pd.Timestamp(dataframe_index[-1]).date()
    trades = list_trades(parameters.ticker)
    forecasts: list[dict[str, Any]] = []
    datasets: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}

    def dataset_for_horizon(horizon: int) -> tuple[pd.DataFrame, pd.DataFrame]:
        if horizon not in datasets:
            target = pipeline.build_targets(
                df,
                horizon=horizon,
                side_cost=parameters.side_cost,
                safety_margin=parameters.safety_margin,
                ticker=parameters.ticker,
            )
            datasets[horizon] = pipeline.build_dataset(features, target)
        return datasets[horizon]

    for target_date, required_close in sessions:
        previous_entries = [op for op in trades if op.acceptance_date < target_date]
        previous_latest = previous_entries[-1] if previous_entries else None
        position_before = (
            "LONG"
            if previous_latest and previous_latest.trade_type is TradeType.BUY
            else "NO_POSITION"
        )
        day_acceptances = [op for op in trades if op.acceptance_date == target_date]
        day_acceptance = day_acceptances[-1] if day_acceptances else None

        item: dict[str, Any] = {
            "target_date": target_date,
            "base_close_date": required_close,
            "position_before": position_before,
            "registered_acceptance": day_acceptance.to_dict() if day_acceptance else None,
            "forecast_type": None,
            "horizon_used": None,
            "expected_update_date": None,
            "probability_down": None,
            "probability_neutral": None,
            "probability_up": None,
            "threshold": None,
            "reference_price": None,
        }

        if required_close <= latest_close and required_close not in dataframe_dates:
            item.update(
                {
                    "status": "NO_QUOTE",
                    "suggested_action": "WAITING_FOR_DATA",
                    "description": "No quote is available for the previous close.",
                }
            )
            forecasts.append(item)
            continue

        if required_close <= latest_close:
            base_date = required_close
            horizon_used = 1
            forecast_type = "UPDATED"
            expected_update_date = None
        else:
            base_date = latest_close
            horizon_used = horizon_between_close_and_target(
                ticker=parameters.ticker,
                latest_close=latest_close,
                target_date=target_date,
            )
            forecast_type = "PRELIMINARY"
            expected_update_date = required_close

        X, events = dataset_for_horizon(horizon_used)
        cutoff = pd.Timestamp(base_date)
        forecast = pipeline.predict_latest_close(
            ticker=parameters.ticker,
            df=df.loc[:cutoff],
            features=features.loc[:cutoff],
            X_history=X,
            historical_events=events,
            horizon=horizon_used,
            side_cost=parameters.side_cost,
            annual_short_cost=parameters.annual_short_cost,
            training_window_days=parameters.training_window_days,
            position_mode="long_short",
            threshold=parameters.threshold,
            optimize_threshold=parameters.optimize_threshold,
            model_name=model_used,
        )
        model_position = int(forecast["target_position"])
        if position_before == "LONG":
            action = "SELL" if model_position == -1 else "HOLD"
            description = (
                "Strong downside signal: consider placing a sell offer."
                if action == "SELL"
                else "No strong downside signal: keep monitoring the position."
            )
        else:
            action = "BUY" if model_position == 1 else "WAIT"
            description = (
                "Strong upside signal: consider placing a buy offer."
                if action == "BUY"
                else "No strong upside signal: wait without opening a position."
            )

        if forecast_type == "PRELIMINARY":
            description = (
                f"Preliminary {horizon_used}-session forecast based on the "
                f"{base_date:%Y-%m-%d} close. {description} It will be recalculated "
                f"after {required_close:%Y-%m-%d}."
            )

        item.update(
            {
                "status": "AVAILABLE",
                "suggested_action": action,
                "base_close_date": base_date,
                "forecast_type": forecast_type,
                "horizon_used": horizon_used,
                "expected_update_date": expected_update_date,
                "probability_down": float(forecast["probability_down"]),
                "probability_neutral": float(forecast["probability_neutral"]),
                "probability_up": float(forecast["probability_up"]),
                "threshold": float(forecast["threshold"]),
                "reference_price": float(forecast["reference_price"]),
                "description": description,
            }
        )
        forecasts.append(item)

    total_available = sum(item["status"] == "AVAILABLE" for item in forecasts)
    total_preliminary = sum(
        item["forecast_type"] == "PRELIMINARY" for item in forecasts
    )
    current_state = get_position_state(parameters.ticker, trades)
    save_daily_forecasts(
        ticker=parameters.ticker,
        market_calendar=market_calendar,
        model_used=model_used,
        forecasts=forecasts,
    )
    return normalize_json(
        {
            "api_version": API_VERSION,
            "ticker": parameters.ticker,
            "model_used": model_used,
            "currency": pipeline.currency_symbol(parameters.ticker),
            "market_calendar": market_calendar,
            "start_date": parameters.start_date,
            "end_date": parameters.end_date,
            "latest_available_close": latest_close,
            "current_position": current_state.status.value,
            "total_trading_days": len(forecasts),
            "total_available": total_available,
            "total_preliminary": total_preliminary,
            "total_pending": len(forecasts) - total_available,
            "forecasts": forecasts,
        }
    )


def generate_daily_analysis(parameters: DailyAnalysisRequest) -> dict[str, Any]:
    """Combine historical close validation with future projections."""
    model_used = normalize_model_name(parameters.model)
    df = pipeline.download_data(
        ticker=parameters.ticker,
        start=parameters.start.isoformat(),
    )
    features = pipeline.build_features(df, ticker=parameters.ticker)
    latest_close = pd.Timestamp(df.index[-1])
    latest_close_date = latest_close.date()
    market_calendar, sessions = sessions_with_previous_close(
        ticker=parameters.ticker,
        start_date=parameters.start_date,
        end_date=parameters.end_date,
    )

    analyses: list[dict[str, Any]] = []
    total_retrainings = 0
    history_end = min(parameters.end_date, latest_close_date)
    if parameters.start_date <= history_end:
        full_target = pipeline.build_targets(
            df,
            horizon=parameters.horizon_trading_days,
            side_cost=parameters.side_cost,
            safety_margin=parameters.safety_margin,
            ticker=parameters.ticker,
        )
        X_history, historical_events = pipeline.build_dataset(features, full_target)
        signals, retrainings = pipeline.predict_daily_historical_signals(
            features=features,
            X_history=X_history,
            historical_events=historical_events,
            full_target=full_target,
            start_date=parameters.start_date,
            end_date=history_end,
            horizon=parameters.horizon_trading_days,
            side_cost=parameters.side_cost,
            annual_short_cost=parameters.annual_short_cost,
            training_window_days=parameters.training_window_days,
            position_mode="long_short",
            threshold=parameters.threshold,
            optimize_threshold=parameters.optimize_threshold,
            retrain_frequency_days=parameters.retrain_frequency_days,
            model_name=model_used,
        )
        total_retrainings += len(retrainings)

        for _, row in signals.iterrows():
            signal_date = pd.Timestamp(row["signal_date"])
            result_available = bool(row["result_available"])
            actual_class = (
                int(row["actual_class"])
                if result_available and pd.notna(row["actual_class"])
                else None
            )
            analyses.append(
                {
                    "reference_date": signal_date.date(),
                    "base_date": signal_date.date(),
                    "forecast_type": "HISTORICAL",
                    "result_status": (
                        "EVALUATED" if result_available else "PENDING"
                    ),
                    "horizon_used": parameters.horizon_trading_days,
                    "reference_price": float(df.loc[signal_date, "close"]),
                    "probability_down": float(row["probability_down"]),
                    "probability_neutral": float(row["probability_neutral"]),
                    "probability_up": float(row["probability_up"]),
                    "threshold": float(row["threshold"]),
                    "action": str(row["action"]),
                    "predicted_class": int(row["predicted_class"]),
                    "predicted_class_name": class_name(int(row["predicted_class"])),
                    "entry_date": (
                        pd.Timestamp(row["entry_date"]).date()
                        if pd.notna(row["entry_date"])
                        else None
                    ),
                    "entry_price": (
                        float(row["entry_price"])
                        if pd.notna(row["entry_price"])
                        else None
                    ),
                    "exit_date": (
                        pd.Timestamp(row["exit_date"]).date()
                        if pd.notna(row["exit_date"])
                        else None
                    ),
                    "exit_price": (
                        float(row["exit_price"])
                        if pd.notna(row["exit_price"])
                        else None
                    ),
                    "observed_return": (
                        float(row["observed_return"])
                        if pd.notna(row["observed_return"])
                        else None
                    ),
                    "actual_class": actual_class,
                    "actual_class_name": (
                        class_name(actual_class) if actual_class is not None else None
                    ),
                    "is_correct": (
                        bool(row["is_correct"])
                        if pd.notna(row["is_correct"])
                        else None
                    ),
                    "training_samples": int(row["training_samples"]),
                    "description": (
                        "Forecast replayed with the data available at that close "
                        "and compared with the observed result."
                        if result_available
                        else "Forecast replayed, but the horizon does not yet have "
                        "a complete result in the available data."
                    ),
                }
            )

    future_datasets: dict[int, tuple[pd.DataFrame, pd.DataFrame]] = {}
    future_dates = [target_date for target_date, _ in sessions if target_date > latest_close_date]
    for target_date in future_dates:
        horizon_used = horizon_between_close_and_target(
            ticker=parameters.ticker,
            latest_close=latest_close_date,
            target_date=target_date,
        )
        if horizon_used not in future_datasets:
            target = pipeline.build_targets(
                df,
                horizon=horizon_used,
                side_cost=parameters.side_cost,
                safety_margin=parameters.safety_margin,
                ticker=parameters.ticker,
            )
            future_datasets[horizon_used] = pipeline.build_dataset(features, target)
        X_history, historical_events = future_datasets[horizon_used]
        forecast = pipeline.predict_latest_close(
            ticker=parameters.ticker,
            df=df,
            features=features,
            X_history=X_history,
            historical_events=historical_events,
            horizon=horizon_used,
            side_cost=parameters.side_cost,
            annual_short_cost=parameters.annual_short_cost,
            training_window_days=parameters.training_window_days,
            position_mode="long_short",
            threshold=parameters.threshold,
            optimize_threshold=parameters.optimize_threshold,
            model_name=model_used,
        )
        total_retrainings += 1
        predicted_class = int(
            pipeline.CLASSES[
                int(
                    np.argmax(
                        [forecast["probability_down"], forecast["probability_neutral"], forecast["probability_up"]]
                    )
                )
            ]
        )
        action = {
            "BUY": "BUY",
            "SELL_SHORT": "SELL",
            "STAY_OUT": "WAIT",
        }[str(forecast["action"])]
        _, projection_dates = next_trading_days(
            ticker=parameters.ticker,
            signal_date=latest_close_date,
            quantity=horizon_used,
        )
        analyses.append(
            {
                "reference_date": target_date,
                "base_date": latest_close_date,
                "forecast_type": "PRELIMINARY",
                "result_status": "PENDING",
                "horizon_used": horizon_used,
                "reference_price": float(forecast["reference_price"]),
                "probability_down": float(forecast["probability_down"]),
                "probability_neutral": float(forecast["probability_neutral"]),
                "probability_up": float(forecast["probability_up"]),
                "threshold": float(forecast["threshold"]),
                "action": action,
                "predicted_class": predicted_class,
                "predicted_class_name": class_name(predicted_class),
                "entry_date": projection_dates[0],
                "entry_price": None,
                "exit_date": target_date,
                "exit_price": None,
                "observed_return": None,
                "actual_class": None,
                "actual_class_name": None,
                "is_correct": None,
                "training_samples": int(forecast["training_samples"]),
                "description": (
                    f"Preliminary projection calculated from the "
                    f"{latest_close_date:%Y-%m-%d} close. It will be recalculated "
                    "when new closes become available."
                ),
            }
        )

    analyses.sort(key=lambda item: item["reference_date"])
    evaluated = [item for item in analyses if item["result_status"] == "EVALUATED"]
    total_correct = sum(item["is_correct"] is True for item in evaluated)
    total_historical = sum(item["forecast_type"] == "HISTORICAL" for item in analyses)
    total_preliminary = len(analyses) - total_historical
    return normalize_json(
        {
            "api_version": API_VERSION,
            "ticker": parameters.ticker,
            "model_used": model_used,
            "currency": pipeline.currency_symbol(parameters.ticker),
            "market_calendar": market_calendar,
            "start_date": parameters.start_date,
            "end_date": parameters.end_date,
            "latest_available_close": latest_close_date,
            "historical_horizon_trading_days": parameters.horizon_trading_days,
            "retraining_frequency_trading_days": parameters.retrain_frequency_days,
            "ignored_non_trading_days": (
                (parameters.end_date - parameters.start_date).days + 1 - len(sessions)
            ),
            "total_trading_days": len(analyses),
            "total_historical": total_historical,
            "total_preliminary": total_preliminary,
            "total_evaluated": len(evaluated),
            "total_pending": len(analyses) - len(evaluated),
            "total_correct": total_correct,
            "accuracy_rate": total_correct / len(evaluated) if evaluated else None,
            "total_retrainings": total_retrainings,
            "analyses": analyses,
        }
    )


def reproduce_historical_forecasts(
    parameters: HistoricalForecastRequest,
) -> dict[str, Any]:
    """Replay each signal using only information known on that date."""
    horizon_trading_days = parameters.horizon_trading_days
    model_used = normalize_model_name(parameters.model)
    df, features, X, events = build_data(parameters)
    full_target = pipeline.build_targets(
        df,
        horizon=horizon_trading_days,
        side_cost=parameters.side_cost,
        safety_margin=parameters.safety_margin,
        ticker=parameters.ticker,
    )

    valid_features = features.replace([np.inf, -np.inf], np.nan).dropna()
    start_value = pd.Timestamp(parameters.start_date)
    end_value = pd.Timestamp(parameters.end_date)
    signal_dates = valid_features.index[
        (valid_features.index >= start_value) & (valid_features.index <= end_value)
    ]
    if len(signal_dates) == 0:
        raise ValueError("No trading sessions contain complete data in the requested period.")

    # Price history is the source of truth for the calendar. This avoids
    # maintaining separate holiday lists for Nasdaq, B3, and other markets.
    price_index = pd.DatetimeIndex(df.index)
    if price_index.tz is not None:
        price_index = price_index.tz_localize(None)
    trading_dates = set(price_index.normalize())
    requested_days = pd.date_range(start_value.normalize(), end_value.normalize(), freq="D")
    ignored_non_trading_days = sum(
        data not in trading_dates for data in requested_days
    )

    forecasts: list[dict[str, Any]] = []
    for signal_date in signal_dates:
        forecast = pipeline.predict_latest_close(
            ticker=parameters.ticker,
            df=df.loc[:signal_date],
            features=features.loc[:signal_date],
            X_history=X,
            historical_events=events,
            horizon=horizon_trading_days,
            side_cost=parameters.side_cost,
            annual_short_cost=parameters.annual_short_cost,
            training_window_days=parameters.training_window_days,
            position_mode=parameters.position_mode,
            threshold=parameters.threshold,
            optimize_threshold=parameters.optimize_threshold,
            model_name=model_used,
        )

        probabilities = np.array(
            [forecast["probability_down"], forecast["probability_neutral"], forecast["probability_up"]],
            dtype=float,
        )
        predicted_class = int(pipeline.CLASSES[int(np.argmax(probabilities))])
        day_target = full_target.loc[signal_date]
        result_available = pd.notna(day_target["label"])

        entry_date = (
            pd.Timestamp(day_target["entry_date"])
            if pd.notna(day_target["entry_date"])
            else None
        )
        entry_price = (
            float(day_target["entry_price"])
            if pd.notna(day_target["entry_price"])
            else None
        )

        actual_class: int | None = None
        realized_return: float | None = None
        exit_date: pd.Timestamp | None = None
        exit_price: float | None = None
        is_correct: bool | None = None
        if result_available:
            actual_class = int(day_target["label"])
            realized_return = float(day_target["gross_return"])
            exit_date = pd.Timestamp(day_target["exit_date"])
            exit_price = float(day_target["exit_price"])
            is_correct = predicted_class == actual_class

        forecasts.append(
            {
                "signal_date": pd.Timestamp(signal_date),
                "reference_price": float(forecast["reference_price"]),
                "entry_date": entry_date,
                "entry_price": entry_price,
                "probability_down": float(forecast["probability_down"]),
                "probability_neutral": float(forecast["probability_neutral"]),
                "probability_up": float(forecast["probability_up"]),
                "threshold": float(forecast["threshold"]),
                "action": forecast["action"],
                "description": forecast["description"],
                "predicted_class": predicted_class,
                "predicted_class_name": class_name(predicted_class),
                "result_available": result_available,
                "exit_date": exit_date,
                "exit_price": exit_price,
                "realized_return": realized_return,
                "actual_class": actual_class,
                "actual_class_name": class_name(actual_class) if actual_class is not None else None,
                "is_correct": is_correct,
            }
        )

    evaluated = [item for item in forecasts if item["result_available"]]
    total_correct = sum(bool(item["is_correct"]) for item in evaluated)
    accuracy_rate = total_correct / len(evaluated) if evaluated else None
    return normalize_json(
        {
            "ticker": parameters.ticker,
            "model_used": model_used,
            "currency": pipeline.currency_symbol(parameters.ticker),
            "start_date": parameters.start_date,
            "end_date": parameters.end_date,
            "effective_start_date": pd.Timestamp(signal_dates[0]).date(),
            "effective_end_date": pd.Timestamp(signal_dates[-1]).date(),
            "ignored_non_trading_days": ignored_non_trading_days,
            "horizon_trading_days": horizon_trading_days,
            "total_signals": len(forecasts),
            "total_evaluated": len(evaluated),
            "total_correct": total_correct,
            "accuracy_rate": accuracy_rate,
            "forecasts": forecasts,
        }
    )


def simulate_and_save_backtest(
    parameters: BacktestRequest,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, int]:
    horizon_trading_days = parameters.horizon_trading_days
    model_used = normalize_model_name(parameters.model)
    _, _, X, events = build_data(parameters)
    simulation, retraining_log = pipeline.simulate_investment(
        X=X,
        events=events,
        start_date=parameters.simulate_from.isoformat(),
        horizon=horizon_trading_days,
        side_cost=parameters.side_cost,
        annual_short_cost=parameters.annual_short_cost,
        initial_capital=parameters.initial_capital,
        retrain_frequency_days=parameters.retrain_frequency_days,
        training_window_days=parameters.training_window_days,
        position_mode=parameters.position_mode,
        threshold=parameters.threshold,
        optimize_threshold=parameters.optimize_threshold,
        model_name=model_used,
    )
    metrics = pipeline.calculate_metrics(
        simulation,
        initial_capital=parameters.initial_capital,
        horizon=horizon_trading_days,
    )

    backtest_id = save_backtest(
        ticker=parameters.ticker,
        model_used=model_used,
        parameters=parameters.model_dump(mode="json"),
        metrics=normalize_json(metrics.to_dict()),
        trades=normalize_json(simulation.to_dict(orient="records")),
        retrainings=normalize_json(retraining_log.to_dict(orient="records")),
    )
    return simulation, retraining_log, metrics, backtest_id


def run_backtest(parameters: BacktestRequest) -> dict[str, Any]:
    simulation, retraining_log, metrics, backtest_id = simulate_and_save_backtest(
        parameters
    )
    horizon_trading_days = parameters.horizon_trading_days

    return {
        "ticker": parameters.ticker,
        "backtest_id": backtest_id,
        "model_used": normalize_model_name(parameters.model),
        "currency": pipeline.currency_symbol(parameters.ticker),
        "requested_start": parameters.simulate_from,
        "first_signal": pd.Timestamp(simulation["signal_date"].iloc[0]).to_pydatetime(),
        "last_exit": pd.Timestamp(simulation["exit_date"].iloc[-1]).to_pydatetime(),
        "horizon": horizon_trading_days,
        "requested_horizon": parameters.horizon,
        "horizon_unit": parameters.horizon_unit,
        "horizon_trading_days": horizon_trading_days,
        "position_mode": parameters.position_mode,
        "metrics": normalize_json(metrics.to_dict()),
        "total_retrainings": int(len(retraining_log)),
    }


def run_period_backtest(
    parameters: PeriodBacktestRequest,
) -> dict[str, Any]:
    training_start = parameters.start_date - timedelta(
        days=365 * parameters.training_history_years
    )
    configuration = BacktestRequest(
        ticker=parameters.ticker,
        model=parameters.model,
        start=training_start,
        end=parameters.end_date + timedelta(days=1),
        horizon=parameters.horizon_trading_days,
        horizon_unit="daily",
        side_cost=parameters.side_cost,
        annual_short_cost=parameters.annual_short_cost,
        training_window_days=parameters.training_window_days,
        position_mode=parameters.position_mode,
        threshold=parameters.threshold,
        optimize_threshold=parameters.optimize_threshold,
        simulate_from=parameters.start_date,
        initial_capital=parameters.initial_capital,
        retrain_frequency_days=parameters.retrain_frequency_days,
    )
    simulation, retraining_log, metrics, backtest_id = simulate_and_save_backtest(
        configuration
    )

    trades: list[dict[str, Any]] = []
    action_names = {1: "BUY", 0: "STAY_OUT", -1: "SELL_SHORT"}
    for _, row in simulation.iterrows():
        position = int(row["position"])
        trades.append(
            {
                "signal_date": pd.Timestamp(row["signal_date"]),
                "entry_date": pd.Timestamp(row["entry_date"]),
                "exit_date": pd.Timestamp(row["exit_date"]),
                "entry_price": float(row["entry_price"]),
                "exit_price": float(row["exit_price"]),
                "probability_down": float(row["probability_down"]),
                "probability_neutral": float(row["probability_neutral"]),
                "probability_up": float(row["probability_up"]),
                "threshold": float(row["threshold"]),
                "position": position,
                "action": action_names[position],
                "asset_return": float(row["gross_asset_return"]),
                "strategy_return": float(row["strategy_return"]),
                "strategy_capital": float(row["strategy_capital"]),
                "buy_hold_capital": float(row["buy_hold_capital"]),
            }
        )

    return normalize_json(
        {
            "api_version": API_VERSION,
            "ticker": parameters.ticker,
            "backtest_id": backtest_id,
            "model_used": normalize_model_name(parameters.model),
            "currency": pipeline.currency_symbol(parameters.ticker),
            "start_date": parameters.start_date,
            "end_date": parameters.end_date,
            "first_signal": pd.Timestamp(simulation["signal_date"].iloc[0]),
            "last_exit": pd.Timestamp(simulation["exit_date"].iloc[-1]),
            "initial_capital": parameters.initial_capital,
            "horizon_trading_days": parameters.horizon_trading_days,
            "position_mode": parameters.position_mode,
            "metrics": metrics.to_dict(),
            "total_retrainings": int(len(retraining_log)),
            "total_events": int(len(simulation)),
            "trades": trades,
        }
    )


async def run_with_error_handling(function, parameters):
    # One heavy run per process avoids local CPU and memory contention.
    async with MODEL_LOCK:
        try:
            return await asyncio.to_thread(function, parameters)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except RuntimeError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        except ModuleNotFoundError as exc:
            raise HTTPException(
                status_code=500,
                detail=f"Missing server dependency: {exc.name}",
            ) from exc
        except Exception as exc:
            logger.exception("Unexpected pipeline failure")
            raise HTTPException(
                status_code=500,
                detail="Internal pipeline failure. Check the server log.",
            ) from exc


def service_information() -> dict[str, str]:
    return {
        "service": "Quant Horizon API",
        "version": API_VERSION,
        "documentation": "/docs",
        "health": "/health",
    }


def health_status() -> dict[str, str]:
    return {
        "status": "ok",
        "version": API_VERSION,
        "utc": datetime.now(timezone.utc).isoformat(),
    }


async def get_simulated_position(ticker: str) -> dict[str, Any]:
    try:
        normalized_ticker = validate_ticker(ticker)
        state = await asyncio.to_thread(get_position_state, normalized_ticker)
        return state.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def reset_simulated_position(ticker: str) -> dict[str, Any]:
    try:
        normalized_ticker = validate_ticker(ticker)
        state = await asyncio.to_thread(reset_trades, normalized_ticker)
        return state.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def register_simulated_acceptance(
    ticker: str,
    acceptance: TradeAcceptanceRequest,
) -> dict[str, Any]:
    try:
        normalized_ticker = validate_ticker(ticker)
        await asyncio.to_thread(register_trade, normalized_ticker, acceptance)
        state = await asyncio.to_thread(get_position_state, normalized_ticker)
        return state.to_dict()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def get_saved_data(ticker: str) -> dict[str, Any]:
    try:
        return await asyncio.to_thread(list_persisted_data, ticker)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


async def generate_simple_forecast(
    ticker: str,
    model: str | None,
    start: date,
    horizon: int,
    horizon_unit: Literal["daily", "weekly"],
    position_mode: Literal["long_flat", "long_short"],
    threshold: float,
    optimize_threshold: bool,
) -> dict[str, Any]:
    try:
        parameters = CurrentForecastRequest(
            ticker=ticker,
            model=model,
            start=start,
            horizon=horizon,
            horizon_unit=horizon_unit,
            position_mode=position_mode,
            threshold=threshold,
            optimize_threshold=optimize_threshold,
        )
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    return await run_with_error_handling(generate_forecast, parameters)


__all__ = [
    "generate_daily_analysis",
    "generate_daily_forecasts",
    "generate_forecast",
    "generate_simple_forecast",
    "get_saved_data",
    "get_simulated_position",
    "health_status",
    "register_simulated_acceptance",
    "reproduce_historical_forecasts",
    "reset_simulated_position",
    "run_backtest",
    "run_period_backtest",
    "run_with_error_handling",
    "service_information",
]
