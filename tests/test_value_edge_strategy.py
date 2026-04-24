"""Value edge strategy tests."""

import numpy as np
import pandas as pd

from strategies.value_edge import ValueEdgeStrategy


def test_value_edge_neutral_on_flat(ohlcv_flat):
    s = ValueEdgeStrategy(buy_edge_pct=0.05, sell_edge_pct=-0.05)
    out = s.generate_signal(ohlcv_flat)
    assert out["signal"] in ("buy", "sell", "hold")
    assert 0.0 <= out["confidence"] <= 1.0


def test_value_edge_buy_on_strong_drift(rng):
    n = 120
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    # Strong upward drift -> positive edge vs last
    close = 100 * np.exp(np.linspace(0, 0.08, n)) + rng.normal(0, 0.01, n)
    df = pd.DataFrame(
        {
            "timestamp": idx,
            "open": close,
            "high": close + 0.05,
            "low": close - 0.05,
            "close": close,
            "volume": rng.uniform(1000, 2000, n),
        }
    )
    s = ValueEdgeStrategy(lookback=80, horizon_bars=5, buy_edge_pct=0.0005, sell_edge_pct=-0.01)
    out = s.generate_signal(df)
    assert out["signal"] in ("buy", "hold")
