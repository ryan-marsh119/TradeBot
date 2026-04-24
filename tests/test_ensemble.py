"""Ensemble orchestrator tests."""

from strategies.base import Strategy
from strategies.ensemble import WeightedStrategy, run_ensemble
from strategies.schema import SignalDict
import pandas as pd


class FixedStrategy(Strategy):
    def __init__(self, signal: str, conf: float):
        self._signal = signal
        self._conf = conf

    def generate_signal(self, df: pd.DataFrame) -> SignalDict:
        return {"signal": self._signal, "confidence": self._conf, "reason": "fixed"}


def test_ensemble_buy_majority(ohlcv_flat):
    members = [
        WeightedStrategy(FixedStrategy("buy", 0.9), 1.0),
        WeightedStrategy(FixedStrategy("buy", 0.8), 1.0),
        WeightedStrategy(FixedStrategy("sell", 0.2), 1.0),
    ]
    out = run_ensemble(ohlcv_flat, members)
    assert out["signal"] == "buy"
    assert out["confidence"] > 0


def test_ensemble_hold_when_balanced(ohlcv_flat):
    members = [
        WeightedStrategy(FixedStrategy("buy", 0.5), 1.0),
        WeightedStrategy(FixedStrategy("sell", 0.5), 1.0),
    ]
    out = run_ensemble(ohlcv_flat, members)
    assert out["signal"] == "hold"
