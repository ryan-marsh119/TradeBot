"""Kronos strategy tests (mocked predictor)."""

import pandas as pd

from strategies.kronos import KronosStrategy
from strategies.kronos_model import MockKronosPredictor


def test_kronos_buy_with_mock(ohlcv_uptrend_hourly):
    mock = MockKronosPredictor(bias=0.02)
    s = KronosStrategy(
        lookback=96,
        pred_len=8,
        move_threshold_pct=0.001,
        predict_override=mock.predict,
    )
    out = s.generate_signal(ohlcv_uptrend_hourly)
    assert out["signal"] == "buy"
    assert out["confidence"] > 0


def test_kronos_sell_with_negative_mock(ohlcv_uptrend_hourly):
    mock = MockKronosPredictor(bias=-0.02)

    def predict(**kwargs):
        df = kwargs["df"]
        pred_len = int(kwargs.get("pred_len", 5))
        last = float(df["close"].iloc[-1])
        close = last * (1.0 - 0.02)
        return pd.DataFrame({"close": [close] * pred_len})

    s = KronosStrategy(
        lookback=96,
        pred_len=8,
        move_threshold_pct=0.001,
        predict_override=predict,
    )
    out = s.generate_signal(ohlcv_uptrend_hourly)
    assert out["signal"] == "sell"
