"""Domain and business-rule exceptions for Quant Horizon."""

from __future__ import annotations

from typing import Any


class BusinessRuleError(Exception):
    """Raised when a request violates an application business rule."""

    def __init__(
        self,
        *,
        code: str,
        message: str,
        status_code: int = 422,
        context: dict[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code
        self.context = context or {}


__all__ = ["BusinessRuleError"]
