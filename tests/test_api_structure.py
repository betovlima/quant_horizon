"""Unit tests for the separation between entities, DTOs, and HTTP routes."""

from __future__ import annotations

import ast
import unittest
from datetime import date, datetime, timezone
from pathlib import Path

from quant_horizon.entities import PositionState, PositionStatus, Trade, TradeType, normalize_ticker


ROOT = Path(__file__).resolve().parents[1] / "src" / "quant_horizon"


class EntityTests(unittest.TestCase):
    def test_state_without_trades(self) -> None:
        state = PositionState.from_trades("AAPL", [])

        self.assertIs(state.status, PositionStatus.NO_POSITION)
        self.assertIsNone(state.last_trade)
        self.assertEqual(state.to_dict()["trades"], [])

    def test_open_buy_defines_long_state(self) -> None:
        purchase = Trade(
            id=1,
            ticker="AAPL",
            trade_type=TradeType.BUY,
            acceptance_date=date(2026, 7, 10),
            acceptance_price=314.20,
            created_at=datetime(2026, 7, 10, 20, tzinfo=timezone.utc),
        )

        state = PositionState.from_trades("AAPL", [purchase])
        serialized = state.to_dict()

        self.assertIs(state.status, PositionStatus.LONG)
        self.assertEqual(serialized["status"], "LONG")
        self.assertEqual(serialized["last_trade"]["trade_type"], "BUY")

    def test_sell_closes_position(self) -> None:
        created_at = datetime(2026, 7, 10, 20, tzinfo=timezone.utc)
        trades = [
            Trade(1, "AAPL", TradeType.BUY, date(2026, 7, 9), 310.0, created_at),
            Trade(2, "AAPL", TradeType.SELL, date(2026, 7, 10), 314.0, created_at),
        ]

        state = PositionState.from_trades("AAPL", trades)

        self.assertIs(state.status, PositionStatus.NO_POSITION)
        self.assertIsNone(state.purchase_date)
        self.assertEqual(len(state.trades), 2)

    def test_normalize_ticker(self) -> None:
        self.assertEqual(normalize_ticker(" aapl "), "AAPL")
        with self.assertRaises(ValueError):
            normalize_ticker("AAPL;DROP TABLE")


class ArchitectureTests(unittest.TestCase):
    def test_api_imports_dtos_without_redeclaring_them(self) -> None:
        api_tree = ast.parse((ROOT / "api.py").read_text(encoding="utf-8"))
        api_classes = {
            node.name for node in ast.walk(api_tree) if isinstance(node, ast.ClassDef)
        }
        dto_imports = {
            alias.name
            for node in ast.walk(api_tree)
            if isinstance(node, ast.ImportFrom) and node.module == "dtos"
            for alias in node.names
        }

        self.assertFalse(api_classes)
        self.assertIn("CurrentForecastRequest", dto_imports)
        self.assertIn("DailyAnalysisResponse", dto_imports)
        self.assertIn("DailyForecastResponse", dto_imports)
        self.assertIn("PeriodBacktestResponse", dto_imports)

    def test_api_contains_only_endpoint_functions(self) -> None:
        api_tree = ast.parse((ROOT / "api.py").read_text(encoding="utf-8"))
        functions = [
            node
            for node in api_tree.body
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
        ]

        self.assertEqual(len(functions), 13)
        for function in functions:
            app_decorators = [
                decorator
                for decorator in function.decorator_list
                if isinstance(decorator, ast.Call)
                and isinstance(decorator.func, ast.Attribute)
                and isinstance(decorator.func.value, ast.Name)
                and decorator.func.value.id == "app"
            ]
            self.assertTrue(
                app_decorators,
                f"{function.name} is not an endpoint and does not belong in api.py.",
            )

    def test_api_does_not_import_infrastructure_or_pipeline(self) -> None:
        api_tree = ast.parse((ROOT / "api.py").read_text(encoding="utf-8"))
        modules = {
            node.module.split(".")[0]
            for node in ast.walk(api_tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        modules.update(
            alias.name.split(".")[0]
            for node in ast.walk(api_tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )

        forbidden = {"numpy", "pandas", "pipeline", "sqlite3", "yfinance"}
        self.assertTrue(forbidden.isdisjoint(modules))

    def test_entities_do_not_depend_on_web_frameworks(self) -> None:
        tree = ast.parse((ROOT / "entities.py").read_text(encoding="utf-8"))
        modules = {
            node.module.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module
        }
        modules.update(
            alias.name.split(".")[0]
            for node in ast.walk(tree)
            if isinstance(node, ast.Import)
            for alias in node.names
        )

        self.assertNotIn("fastapi", modules)
        self.assertNotIn("pydantic", modules)
        self.assertNotIn("pipeline", modules)


if __name__ == "__main__":
    unittest.main()
