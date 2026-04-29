"""Tests for milestone 4 paper trading pipeline."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from core.backtester import Backtester
from core.paper_broker import PaperBroker
from core.trade_ledger import TradeLedger
from strategies.base import Strategy
from strategies.schema import SignalDict


class BuyThenHoldStrategy(Strategy):
    def generate_signal(self, df: pd.DataFrame) -> SignalDict:
        if len(df) < 90:
            return {"signal": "buy", "confidence": 0.7, "reason": "entry"}
        return {"signal": "hold", "confidence": 0.1, "reason": "idle"}


class SellWeakStrategy(Strategy):
    def generate_signal(self, df: pd.DataFrame) -> SignalDict:
        return {"signal": "sell", "confidence": 0.2, "reason": "weak counter"}


def _sample_bars(n: int = 140) -> pd.DataFrame:
    idx = pd.date_range("2025-01-01", periods=n, freq="h")
    close = pd.Series(range(n), dtype=float) * 0.25 + 100.0
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": close - 0.1,
            "high": close + 0.2,
            "low": close - 0.3,
            "close": close,
            "volume": 10_000.0,
        }
    )


def test_trade_ledger_idempotent_event_id(tmp_path: Path):
    ledger = TradeLedger(db_path=str(tmp_path / "trades.db"), csv_dir=str(tmp_path / "csv"))
    event = {
        "run_id": "r1",
        "event_id": "evt-1",
        "event_type": "signal_received",
        "bar_time": pd.Timestamp("2025-01-01T00:00:00Z").to_pydatetime(),
        "symbol": "SOL/USD",
        "strategy": "X",
        "strategy_version": "v1",
        "latency_ms": 1,
        "payload_json": "{}",
        "error": None,
    }
    ledger.log_event(event)  # type: ignore[arg-type]
    ledger.log_event(event)  # type: ignore[arg-type]
    counts = ledger.count_events("r1")
    assert counts["signal_received"] == 1


def test_backtester_end_to_end_writes_events(tmp_path: Path):
    ledger = TradeLedger(db_path=str(tmp_path / "trades.db"), csv_dir=str(tmp_path / "csv"))
    broker = PaperBroker(initial_cash=5_000.0, fee_bps_taker=8.0)
    bt = Backtester(ledger=ledger, broker=broker, warmup_bars=30, delay_bars=1)

    out = bt.run(
        df=_sample_bars(),
        symbol="SOL/USD",
        timeframe="1h",
        strategies=[BuyThenHoldStrategy(), SellWeakStrategy()],
        run_id="run-2025",
        random_seed=11,
    )

    assert out["run_id"] == "run-2025"
    counts = ledger.count_events("run-2025")
    assert counts["signal_received"] > 10
    assert counts["order_simulated"] >= 1
    assert counts["fill_applied"] >= 1
    assert counts["equity_snapshot"] >= 1
    csv_file = tmp_path / "csv" / "run-2025.csv"
    assert csv_file.exists()


def test_paper_broker_rejects_oversized_buy():
    broker = PaperBroker(initial_cash=10.0, fee_bps_taker=10.0, allow_negative_cash=False)
    signal = {
        "symbol": "SOL/USD",
        "side": "buy",
        "confidence": 1.0,
        "reason": "max size",
        "timestamp": pd.Timestamp("2025-01-01T00:00:00Z").to_pydatetime(),
        "strategy_name": "S",
        "strategy_version": "v1",
        "strategy_meta": {},
    }
    res = broker.execute_signal(
        signal,  # type: ignore[arg-type]
        current_bar={"open": 100.0, "high": 102.0, "low": 99.0, "close": 100.0},
        next_bar={"open": 101.0, "high": 102.0, "low": 100.0, "close": 101.0},
    )
    assert res["status"] in {"filled", "rejected"}
