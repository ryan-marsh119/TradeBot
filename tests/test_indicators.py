"""Indicator sanity checks."""

import pandas as pd

from strategies.indicators import bollinger_bands, macd, obv, rsi


def test_rsi_bounds():
    close = pd.Series(range(1, 100))
    r = rsi(close)
    assert r.dropna().between(0, 100).all()


def test_macd_lengths():
    close = pd.Series(range(1, 200))
    line, sig, hist = macd(close)
    assert len(line) == len(close)


def test_bollinger_order():
    close = pd.Series([1, 2, 3, 4, 5] * 10)
    lo, mid, hi = bollinger_bands(close, period=10)
    valid = (~mid.isna()) & (~lo.isna()) & (~hi.isna())
    assert (lo[valid] <= mid[valid]).all() and (mid[valid] <= hi[valid]).all()


def test_obv_finite():
    close = pd.Series([1.0, 1.1, 1.0, 1.2])
    vol = pd.Series([10, 10, 10, 10])
    o = obv(close, vol)
    assert o.notna().all()
