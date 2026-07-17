"""Request and response DTOs for the Quant Horizon API.

HTTP validation stays in this module. Domain entities remain in
``entities.py`` and quantitative calculations remain in ``pipeline.py``.
"""

from __future__ import annotations

from datetime import date, datetime, timedelta
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from . import pipeline
from .entities import normalize_ticker
from .investment_models import ModelName


class ModelParameters(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    ticker: str = Field(default="AAPL", min_length=1, max_length=20)
    model: ModelName | None = Field(
        default=None,
        description="Requested model. Empty uses QUANT_HORIZON_MODEL or LightGBM.",
    )
    start: date = Field(default=date(2010, 1, 1))
    end: date | None = Field(
        default=None,
        description="Exclusive end date. Leave empty to use the latest data.",
    )
    horizon: int = Field(
        default=5,
        ge=1,
        le=60,
        description="Quantity expressed in the selected horizon_unit.",
    )
    horizon_unit: Literal["daily", "weekly"] = "daily"
    side_cost: float = Field(default=0.0005, ge=0, le=0.05)
    safety_margin: float = Field(default=0.0005, ge=0, le=0.10)
    annual_short_cost: float = Field(default=0.0, ge=0, le=1.0)
    training_window_days: int = Field(
        default=0,
        ge=0,
        le=10000,
        description="0 uses an expanding history; another value uses a rolling window.",
    )
    position_mode: Literal["long_flat", "long_short"] = "long_flat"
    threshold: float = Field(default=0.55, gt=0, lt=1)
    optimize_threshold: bool = True

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        return normalize_ticker(value)

    @model_validator(mode="after")
    def validate_dates(self):
        if self.end is not None and self.end <= self.start:
            raise ValueError("end must be later than start.")
        if self.horizon_unit == "weekly" and self.horizon > 12:
            raise ValueError("In weekly mode, horizon must not exceed 12.")
        return self

    @property
    def horizon_trading_days(self) -> int:
        return pipeline.horizon_in_trading_days(
            quantity=self.horizon,
            unit=self.horizon_unit,
        )


class CurrentForecastRequest(ModelParameters):
    pass


class CurrentForecastResponse(BaseModel):
    ticker: str
    model_used: ModelName
    currency: str
    signal_date: datetime
    latest_price_date: datetime
    reference_price: float
    probability_down: float
    probability_neutral: float
    probability_up: float
    threshold: float
    target_position: int
    action: str
    description: str
    horizon: int
    requested_horizon: int
    horizon_unit: Literal["daily", "weekly"]
    horizon_trading_days: int
    market_calendar: str
    forecast_trading_dates: list[date]
    expected_entry_date: date
    expected_exit_date: date
    capital_fraction: float
    training_samples: int
    training_start: datetime
    training_end: datetime


class HistoricalForecastRequest(ModelParameters):
    start_date: date
    end_date: date

    @model_validator(mode="after")
    def validate_historical_period(self):
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date.")
        if (self.end_date - self.start_date).days > 31:
            raise ValueError("The historical period must not exceed 31 calendar days.")
        return self


class HistoricalForecastItem(BaseModel):
    signal_date: datetime
    reference_price: float
    entry_date: datetime | None
    entry_price: float | None
    probability_down: float
    probability_neutral: float
    probability_up: float
    threshold: float
    action: str
    description: str
    predicted_class: int
    predicted_class_name: str
    result_available: bool
    exit_date: datetime | None
    exit_price: float | None
    realized_return: float | None
    actual_class: int | None
    actual_class_name: str | None
    is_correct: bool | None


class HistoricalForecastResponse(BaseModel):
    ticker: str
    model_used: ModelName
    currency: str
    start_date: date
    end_date: date
    effective_start_date: date
    effective_end_date: date
    ignored_non_trading_days: int
    horizon_trading_days: int
    total_signals: int
    total_evaluated: int
    total_correct: int
    accuracy_rate: float | None
    forecasts: list[HistoricalForecastItem]


class DailyAnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    ticker: str = Field(default="AAPL", min_length=1, max_length=20)
    model: ModelName | None = None
    start_date: date
    end_date: date
    start: date = Field(default=date(2010, 1, 1))
    horizon_trading_days: int = Field(default=5, ge=1, le=60)
    side_cost: float = Field(default=0.0005, ge=0, le=0.05)
    safety_margin: float = Field(default=0.0005, ge=0, le=0.10)
    annual_short_cost: float = Field(default=0.0, ge=0, le=1.0)
    training_window_days: int = Field(default=0, ge=0, le=10000)
    threshold: float = Field(default=0.55, gt=0, lt=1)
    optimize_threshold: bool = True
    retrain_frequency_days: int = Field(default=21, ge=1, le=252)

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        try:
            return normalize_ticker(value)
        except ValueError as exc:
            raise ValueError("Invalid ticker.") from exc

    @model_validator(mode="after")
    def validate_period(self):
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date.")
        if (self.end_date - self.start_date).days > 366:
            raise ValueError("Daily analysis must not exceed 366 calendar days.")
        if self.end_date > date.today() + timedelta(days=31):
            raise ValueError("Future projections must end within 31 days.")
        return self


class DailyAnalysisItem(BaseModel):
    reference_date: date
    base_date: date
    forecast_type: Literal["HISTORICAL", "PRELIMINARY"]
    result_status: Literal["EVALUATED", "PENDING"]
    horizon_used: int
    reference_price: float
    probability_down: float
    probability_neutral: float
    probability_up: float
    threshold: float
    action: Literal["BUY", "SELL", "WAIT"]
    predicted_class: int
    predicted_class_name: str
    entry_date: date | None
    entry_price: float | None
    exit_date: date | None
    exit_price: float | None
    observed_return: float | None
    actual_class: int | None
    actual_class_name: str | None
    is_correct: bool | None
    training_samples: int
    description: str


class DailyAnalysisResponse(BaseModel):
    api_version: str
    ticker: str
    model_used: ModelName
    currency: str
    market_calendar: str
    start_date: date
    end_date: date
    latest_available_close: date
    historical_horizon_trading_days: int
    retraining_frequency_trading_days: int
    ignored_non_trading_days: int
    total_trading_days: int
    total_historical: int
    total_preliminary: int
    total_evaluated: int
    total_pending: int
    total_correct: int
    accuracy_rate: float | None
    total_retrainings: int
    analyses: list[DailyAnalysisItem]


class DailyForecastRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    ticker: str = Field(default="AAPL", min_length=1, max_length=20)
    model: ModelName | None = None
    start_date: date
    end_date: date
    start: date = Field(default=date(2010, 1, 1))
    side_cost: float = Field(default=0.0005, ge=0, le=0.05)
    safety_margin: float = Field(default=0.0005, ge=0, le=0.10)
    annual_short_cost: float = Field(default=0.0, ge=0, le=1.0)
    training_window_days: int = Field(default=0, ge=0, le=10000)
    threshold: float = Field(default=0.55, gt=0, lt=1)
    optimize_threshold: bool = True

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        try:
            return normalize_ticker(value)
        except ValueError as exc:
            raise ValueError("Invalid ticker.") from exc

    @model_validator(mode="after")
    def validate_period(self):
        if self.end_date < self.start_date:
            raise ValueError("end_date must be on or after start_date.")
        if (self.end_date - self.start_date).days > 31:
            raise ValueError("The daily forecast range must not exceed 31 calendar days.")
        return self


class TradeAcceptanceRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    trade_type: Literal["BUY", "SELL"]
    acceptance_date: date
    acceptance_price: float | None = Field(default=None, gt=0)

    @model_validator(mode="after")
    def prevent_future_date(self):
        if self.acceptance_date > date.today():
            raise ValueError("The acceptance date cannot be in the future.")
        return self


class RegisteredTrade(BaseModel):
    id: int
    ticker: str
    trade_type: Literal["BUY", "SELL"]
    acceptance_date: date
    acceptance_price: float | None
    created_at: datetime


class PositionStateResponse(BaseModel):
    ticker: str
    status: Literal["NO_POSITION", "LONG"]
    purchase_date: date | None
    purchase_price: float | None
    last_trade: RegisteredTrade | None
    trades: list[RegisteredTrade]


class DailyForecastItem(BaseModel):
    target_date: date
    base_close_date: date
    status: Literal["AVAILABLE", "WAITING_FOR_CLOSE", "NO_QUOTE"]
    position_before: Literal["NO_POSITION", "LONG"]
    suggested_action: Literal[
        "BUY", "WAIT", "HOLD", "SELL", "WAITING_FOR_DATA"
    ]
    forecast_type: Literal["UPDATED", "PRELIMINARY"] | None
    horizon_used: int | None
    expected_update_date: date | None
    probability_down: float | None
    probability_neutral: float | None
    probability_up: float | None
    threshold: float | None
    reference_price: float | None
    description: str
    registered_acceptance: RegisteredTrade | None


class DailyForecastResponse(BaseModel):
    api_version: str
    ticker: str
    model_used: ModelName
    currency: str
    market_calendar: str
    start_date: date
    end_date: date
    latest_available_close: date
    current_position: Literal["NO_POSITION", "LONG"]
    total_trading_days: int
    total_available: int
    total_preliminary: int
    total_pending: int
    forecasts: list[DailyForecastItem]


class BacktestRequest(ModelParameters):
    simulate_from: date = Field(default=date(2015, 1, 1))
    initial_capital: float = Field(default=100.0, gt=0, le=1_000_000_000)
    retrain_frequency_days: int = Field(default=252, ge=1, le=2520)

    @model_validator(mode="after")
    def validate_simulation_start(self):
        if self.simulate_from <= self.start:
            raise ValueError(
                "simulate_from must be later than start so training history exists."
            )
        if self.end is not None and self.simulate_from >= self.end:
            raise ValueError("simulate_from must be earlier than end.")
        return self


class BacktestResponse(BaseModel):
    backtest_id: int
    ticker: str
    model_used: ModelName
    currency: str
    requested_start: date
    first_signal: datetime
    last_exit: datetime
    horizon: int
    requested_horizon: int
    horizon_unit: Literal["daily", "weekly"]
    horizon_trading_days: int
    position_mode: str
    metrics: dict[str, Any]
    total_retrainings: int


class PeriodBacktestRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    ticker: str = Field(default="AAPL", min_length=1, max_length=20)
    model: ModelName | None = None
    start_date: date
    end_date: date
    initial_capital: float = Field(default=100.0, gt=0, le=1_000_000_000)
    horizon_trading_days: int = Field(default=5, ge=1, le=60)
    training_history_years: int = Field(default=8, ge=2, le=20)
    side_cost: float = Field(default=0.0005, ge=0, le=0.05)
    annual_short_cost: float = Field(default=0.0, ge=0, le=1.0)
    training_window_days: int = Field(default=0, ge=0, le=10000)
    position_mode: Literal["long_flat", "long_short"] = "long_flat"
    threshold: float = Field(default=0.55, gt=0, lt=1)
    optimize_threshold: bool = True
    retrain_frequency_days: int = Field(default=252, ge=1, le=2520)

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        try:
            return normalize_ticker(value)
        except ValueError as exc:
            raise ValueError("Invalid ticker.") from exc

    # Date-range business rules are evaluated by the backtest service so the
    # API can return stable error codes and contextual information to clients.


class PeriodBacktestTrade(BaseModel):
    signal_date: datetime
    entry_date: datetime
    exit_date: datetime
    entry_price: float
    exit_price: float
    probability_down: float
    probability_neutral: float
    probability_up: float
    threshold: float
    position: int
    action: str
    asset_return: float
    strategy_return: float
    strategy_capital: float
    buy_hold_capital: float


class PeriodBacktestResponse(BaseModel):
    api_version: str
    backtest_id: int
    ticker: str
    model_used: ModelName
    currency: str
    start_date: date
    end_date: date
    first_signal: datetime
    last_exit: datetime
    initial_capital: float
    horizon_trading_days: int
    position_mode: str
    metrics: dict[str, Any]
    total_retrainings: int
    total_events: int
    trades: list[PeriodBacktestTrade]


__all__ = [
    "BacktestRequest",
    "BacktestResponse",
    "CurrentForecastRequest",
    "CurrentForecastResponse",
    "DailyAnalysisItem",
    "DailyAnalysisRequest",
    "DailyAnalysisResponse",
    "DailyForecastItem",
    "DailyForecastRequest",
    "DailyForecastResponse",
    "HistoricalForecastItem",
    "HistoricalForecastRequest",
    "HistoricalForecastResponse",
    "ModelParameters",
    "PeriodBacktestRequest",
    "PeriodBacktestResponse",
    "PeriodBacktestTrade",
    "PositionStateResponse",
    "RegisteredTrade",
    "TradeAcceptanceRequest",
]
