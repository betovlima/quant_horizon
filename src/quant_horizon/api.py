"""HTTP endpoints for the local Quant Horizon API.

Recommended command:
    uvicorn api:app --host 127.0.0.1 --port 8000

Configuration, security, persistence, and business rules live in separate
modules. This file contains only public routes.
"""

from __future__ import annotations

from datetime import date
from typing import Any, Literal

from fastapi import Depends, Query

from .app import app
from .dtos import (
    BacktestRequest,
    BacktestResponse,
    CurrentForecastRequest,
    CurrentForecastResponse,
    DailyAnalysisRequest,
    DailyAnalysisResponse,
    DailyForecastRequest,
    DailyForecastResponse,
    HistoricalForecastRequest,
    HistoricalForecastResponse,
    PeriodBacktestRequest,
    PeriodBacktestResponse,
    PositionStateResponse,
    TradeAcceptanceRequest,
)
from .investment_models import ModelName
from .security import validate_api_key
from .services import (
    generate_daily_analysis,
    generate_daily_forecasts,
    generate_forecast,
    generate_simple_forecast,
    get_saved_data,
    get_simulated_position,
    health_status,
    register_simulated_acceptance,
    reproduce_historical_forecasts,
    reset_simulated_position,
    run_backtest,
    run_period_backtest,
    run_with_error_handling,
    service_information,
)


@app.get("/", tags=["Service"])
async def root() -> dict[str, str]:
    return service_information()


@app.get("/health", tags=["Service"])
async def health() -> dict[str, str]:
    return health_status()


@app.post(
    "/v1/forecasts/current",
    response_model=CurrentForecastResponse,
    tags=["Forecasts"],
    dependencies=[Depends(validate_api_key)],
)
async def current_forecast(parameters: CurrentForecastRequest) -> dict[str, Any]:
    """Forecast the next horizon from the latest available close."""
    return await run_with_error_handling(generate_forecast, parameters)


@app.post(
    "/v1/forecasts/historical",
    response_model=HistoricalForecastResponse,
    tags=["Forecasts"],
    dependencies=[Depends(validate_api_key)],
)
async def historical_forecasts(
    parameters: HistoricalForecastRequest,
) -> dict[str, Any]:
    """Replay one forecast per session and compare it with the outcome."""
    return await run_with_error_handling(reproduce_historical_forecasts, parameters)


@app.post(
    "/v1/analyses/daily",
    response_model=DailyAnalysisResponse,
    tags=["Analyses"],
    dependencies=[Depends(validate_api_key)],
)
async def daily_analyses(parameters: DailyAnalysisRequest) -> dict[str, Any]:
    """Analyze past sessions and project future sessions in a date range."""
    return await run_with_error_handling(generate_daily_analysis, parameters)


@app.post(
    "/v1/forecasts/daily",
    response_model=DailyForecastResponse,
    tags=["Forecasts"],
    dependencies=[Depends(validate_api_key)],
)
async def daily_forecasts(parameters: DailyForecastRequest) -> dict[str, Any]:
    """Generate one progressive signal per session from the previous close."""
    return await run_with_error_handling(generate_daily_forecasts, parameters)


@app.get(
    "/v1/positions/{ticker}",
    response_model=PositionStateResponse,
    tags=["Simulated position"],
    dependencies=[Depends(validate_api_key)],
)
async def get_position(ticker: str) -> dict[str, Any]:
    """Read simulated acceptances without sending brokerage orders."""
    return await get_simulated_position(ticker)


@app.delete(
    "/v1/positions/{ticker}/acceptances",
    response_model=PositionStateResponse,
    tags=["Simulated position"],
    dependencies=[Depends(validate_api_key)],
)
async def reset_acceptances(ticker: str) -> dict[str, Any]:
    """Delete simulated buys and sells without changing forecasts."""
    return await reset_simulated_position(ticker)


@app.post(
    "/v1/positions/{ticker}/acceptances",
    response_model=PositionStateResponse,
    tags=["Simulated position"],
    dependencies=[Depends(validate_api_key)],
)
async def register_acceptance(
    ticker: str,
    acceptance: TradeAcceptanceRequest,
) -> dict[str, Any]:
    """Record a manual buy or sell acceptance without executing a trade."""
    return await register_simulated_acceptance(ticker, acceptance)


@app.get(
    "/v1/data/{ticker}",
    tags=["Persisted data"],
    dependencies=[Depends(validate_api_key)],
)
async def get_persisted_data(ticker: str) -> dict[str, Any]:
    """Read forecasts and simulated acceptances stored in local SQLite."""
    return await get_saved_data(ticker)


@app.get(
    "/v1/forecasts/current/{ticker}",
    response_model=CurrentForecastResponse,
    tags=["Forecasts"],
    dependencies=[Depends(validate_api_key)],
)
async def get_current_forecast(
    ticker: str,
    model: ModelName | None = Query(default=None),
    start: date = Query(default=date(2010, 1, 1)),
    horizon: int = Query(default=5, ge=1, le=60),
    horizon_unit: Literal["daily", "weekly"] = Query(default="daily"),
    position_mode: Literal["long_flat", "long_short"] = Query(default="long_flat"),
    threshold: float = Query(default=0.55, gt=0, lt=1),
    optimize_threshold: bool = Query(default=True),
) -> dict[str, Any]:
    """Simplified endpoint for browser testing."""
    return await generate_simple_forecast(
        ticker=ticker,
        model=model,
        start=start,
        horizon=horizon,
        horizon_unit=horizon_unit,
        position_mode=position_mode,
        threshold=threshold,
        optimize_threshold=optimize_threshold,
    )


@app.post(
    "/v1/backtests/period",
    response_model=PeriodBacktestResponse,
    tags=["Backtests"],
    dependencies=[Depends(validate_api_key)],
)
async def period_backtest(parameters: PeriodBacktestRequest) -> dict[str, Any]:
    """Run a complete backtest for the date range selected in the UI."""
    return await run_with_error_handling(run_period_backtest, parameters)


@app.post(
    "/v1/backtests",
    response_model=BacktestResponse,
    tags=["Backtests"],
    dependencies=[Depends(validate_api_key)],
)
async def backtest(parameters: BacktestRequest) -> dict[str, Any]:
    """Run a historical simulation and return aggregate metrics."""
    return await run_with_error_handling(run_backtest, parameters)
