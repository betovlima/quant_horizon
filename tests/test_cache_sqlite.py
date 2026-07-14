"""Tests for SQLite-only price and calculation caching."""

from __future__ import annotations

import os
import sqlite3
import tempfile
import unittest
from contextlib import closing
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from quant_horizon import pipeline


class SQLiteCacheTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary_directory = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary_directory.name)
        self.cache_db = self.root / "cache.sqlite3"
        self.environment = patch.dict(
            os.environ,
            {
                "QUANT_HORIZON_CACHE_ENABLED": "1",
                "QUANT_HORIZON_CACHE_DB": str(self.cache_db),
            },
        )
        self.environment.start()

    def tearDown(self) -> None:
        self.environment.stop()
        self.temporary_directory.cleanup()

    @staticmethod
    def sample_data() -> pd.DataFrame:
        index = pd.bdate_range("2024-01-02", periods=320)
        close = pd.Series(
            100 + np.linspace(0, 30, len(index)) + np.sin(np.arange(len(index))),
            index=index,
        )
        return pd.DataFrame(
            {
                "open": close * 0.999,
                "high": close * 1.01,
                "low": close * 0.99,
                "close": close,
                "volume": 1_000_000 + np.arange(len(index)) * 100,
            },
            index=index,
        )

    def test_features_and_targets_stay_in_sqlite(self) -> None:
        data = self.sample_data()

        with patch.object(
            pipeline,
            "_calculate_features",
            wraps=pipeline._calculate_features,
        ) as calculate_features:
            first = pipeline.build_features(data, ticker="AAPL")
            second = pipeline.build_features(data, ticker="AAPL")

        pipeline.build_targets(data, horizon=5, ticker="AAPL")

        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(calculate_features.call_count, 1)
        self.assertTrue(self.cache_db.exists())
        self.assertFalse((self.root / "investment_cache").exists())

        with closing(sqlite3.connect(self.cache_db)) as connection:
            categories = dict(
                connection.execute(
                    "SELECT category, COUNT(*) FROM calculation_cache GROUP BY category"
                ).fetchall()
            )
        self.assertEqual(categories, {"features": 1, "targets": 1})


if __name__ == "__main__":
    unittest.main()
