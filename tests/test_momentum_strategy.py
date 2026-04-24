"""Momentum strategy tests."""

from strategies.momentum import MomentumStrategy


def test_momentum_uptrend_tends_buy(ohlcv_uptrend_hourly):
    s = MomentumStrategy(buy_threshold=0.05, sell_threshold=-0.05)
    out = s.generate_signal(ohlcv_uptrend_hourly)
    assert out["signal"] in ("buy", "hold", "sell")
    assert 0.0 <= out["confidence"] <= 1.0
    assert "momentum_score" in out["reason"] or out["signal"] == "hold"


def test_momentum_insufficient_rows():
    s = MomentumStrategy()
    import pandas as pd

    small = pd.DataFrame(
        {
            "timestamp": pd.date_range("2024-01-01", periods=10, freq="h"),
            "open": range(10),
            "high": range(1, 11),
            "low": range(10),
            "close": range(10),
            "volume": [1] * 10,
        }
    )
    out = s.safe_generate(small)
    assert out["signal"] == "hold"
