"""Optional authentication for the local API."""

from __future__ import annotations

import os
import secrets

from fastapi import Header, HTTPException, status


async def validate_api_key(x_api_key: str | None = Header(default=None)) -> None:
    expected = os.environ.get("QUANT_HORIZON_API_KEY", "").strip()
    if not expected:
        return
    if not x_api_key or not secrets.compare_digest(x_api_key, expected):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="The API key is missing or invalid.",
        )


__all__ = ["validate_api_key"]
