"""Milestone 4 validation runner for paper backtesting."""

from __future__ import annotations

from datetime import UTC

import numpy as np
import pandas as pd

from core.backtester import Backtester
from core.paper_broker import PaperBroker
from core.trade_ledger import TradeLedger
from strategies.kronos import KronosStrategy
from strategies.momentum import MomentumStrategy
from strategies.value_edge import ValueEdgeStrategy


def build_2025_hourly_bars(seed: int = 42, start_price: float = 100.0) -> pd.DataFrame:
    """Deterministic synthetic OHLCV bars for reproducible validation runs."""
    rng = np.random.default_rng(seed)
    ts = pd.date_range("2025-01-01T00:00:00Z", "2025-12-31T23:00:00Z", freq="1h", tz=UTC)
    noise = rng.normal(loc=0.00005, scale=0.0035, size=len(ts))
    close = start_price * np.exp(np.cumsum(noise))
    open_ = np.concatenate(([start_price], close[:-1]))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0009, 0.0003, len(ts))))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0009, 0.0003, len(ts))))
    volume = rng.uniform(120.0, 2200.0, len(ts))
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
        }
    )


def main() -> None:
    run_id = "milestone4-2025-validation"
    df = build_2025_hourly_bars(seed=42)

    ledger = TradeLedger(db_path="data/trades.db", csv_dir="data/ledger_csv")
    broker = PaperBroker(
        initial_cash=25_000.0,
        fee_bps_taker=9.0,
        fee_bps_maker=4.0,
        slippage_bps_fixed=2.5,
        slippage_vol_multiplier=0.75,
        supported_symbols={"BTCUSDT"},
    )
    backtester = Backtester(ledger=ledger, broker=broker, warmup_bars=96, delay_bars=1)

    # Kronos remains optional; if the heavy model is unavailable it returns hold safely.
    strategies = [MomentumStrategy(), ValueEdgeStrategy(), KronosStrategy()]
    summary = backtester.run(
        df=df,
        symbol="BTCUSDT",
        timeframe="1h",
        strategies=strategies,
        run_id=run_id,
        random_seed=42,
    )
    checks = backtester.validate_run_outputs(run_id)
    event_counts = ledger.count_events(run_id)

    print("Run Summary:", summary)
    print("Acceptance Checks:", checks)
    print("Event Counts:", event_counts)
    ledger.close()

    if not checks["ok"]:
        raise SystemExit(f"Validation failed: {checks['violations']}")


if __name__ == "__main__":
    main()

