"""Shared Quant Horizon API settings."""

from __future__ import annotations

import asyncio
import os
from pathlib import Path


API_VERSION = "2.0.0"
STATE_DB = Path(os.environ.get("QUANT_HORIZON_DB", "db/quant_horizon.sqlite3"))
MODEL_LOCK = asyncio.Lock()


__all__ = ["API_VERSION", "MODEL_LOCK", "STATE_DB"]
