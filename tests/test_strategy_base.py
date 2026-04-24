"""Tests for base validation and utilities."""

import pandas as pd

from strategies.base import Strategy
from strategies.schema import SignalDict
from strategies.utils import clamp_confidence, hold_signal, validate_ohlcv_df


class DummyStrategy(Strategy):
    def generate_signal(self, df: pd.DataFrame) -> SignalDict:
        return {"signal": "buy", "confidence": 0.5, "reason": "dummy"}


def test_validate_ohlcv_df_missing_column():
    df = pd.DataFrame({"open": [1], "high": [1], "low": [1], "close": [1]})
    err = validate_ohlcv_df(df)
    assert err is not None
    assert "volume" in err


def test_safe_generate_empty():
    s = DummyStrategy()
    out = s.safe_generate(pd.DataFrame())
    assert out["signal"] == "hold"


def test_clamp_confidence():
    assert clamp_confidence(-1) == 0.0
    assert clamp_confidence(2) == 1.0
    assert clamp_confidence(float("nan")) == 0.0


def test_hold_signal():
    h = hold_signal(0.3, "x")
    assert h["signal"] == "hold"
    assert h["confidence"] == 0.3
    assert h["reason"] == "x"
