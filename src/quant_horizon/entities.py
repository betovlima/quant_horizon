"""Framework-independent domain entities used by Quant Horizon."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date, datetime
from enum import Enum
from typing import Any, Iterable


TICKER_PATTERN = re.compile(r"^[A-Z0-9.^=_-]{1,20}$")


class TradeType(str, Enum):
    BUY = "BUY"
    SELL = "SELL"


class PositionStatus(str, Enum):
    NO_POSITION = "NO_POSITION"
    LONG = "LONG"


def normalize_ticker(ticker: str) -> str:
    """Normalize and validate a market ticker."""
    value = ticker.strip().upper()
    if not TICKER_PATTERN.fullmatch(value):
        raise ValueError(
            "Invalid ticker. Use only letters, numbers, and . ^ = _ - characters."
        )
    return value


@dataclass(frozen=True, slots=True)
class Trade:
    """A manually accepted simulated buy or sell action."""

    id: int
    ticker: str
    trade_type: TradeType
    acceptance_date: date
    acceptance_price: float | None
    created_at: datetime

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "ticker": self.ticker,
            "trade_type": self.trade_type.value,
            "acceptance_date": self.acceptance_date,
            "acceptance_price": self.acceptance_price,
            "created_at": self.created_at,
        }


@dataclass(frozen=True, slots=True)
class PositionState:
    """Current simulated position derived from the recorded trades."""

    ticker: str
    status: PositionStatus
    purchase_date: date | None
    purchase_price: float | None
    last_trade: Trade | None
    trades: tuple[Trade, ...]

    @classmethod
    def from_trades(
        cls,
        ticker: str,
        trades: Iterable[Trade],
    ) -> "PositionState":
        history = tuple(trades)
        latest = history[-1] if history else None
        is_long = bool(latest and latest.trade_type is TradeType.BUY)
        purchase = latest if is_long else None
        return cls(
            ticker=ticker,
            status=PositionStatus.LONG if is_long else PositionStatus.NO_POSITION,
            purchase_date=purchase.acceptance_date if purchase else None,
            purchase_price=purchase.acceptance_price if purchase else None,
            last_trade=latest,
            trades=history,
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "ticker": self.ticker,
            "status": self.status.value,
            "purchase_date": self.purchase_date,
            "purchase_price": self.purchase_price,
            "last_trade": self.last_trade.to_dict() if self.last_trade else None,
            "trades": [trade.to_dict() for trade in self.trades],
        }


__all__ = [
    "PositionState",
    "PositionStatus",
    "TICKER_PATTERN",
    "Trade",
    "TradeType",
    "normalize_ticker",
]
