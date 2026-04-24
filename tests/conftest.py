"""Shared pytest fixtures."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(42)


@pytest.fixture
def ohlcv_uptrend_hourly(rng: np.random.Generator) -> pd.DataFrame:
    """~42 days of hourly data with a steady drift up (positive daily streaks)."""
    n = 24 * 42
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    t = np.linspace(0, 3.0, n)
    close = 100.0 * np.exp(0.0008 * np.arange(n) + 0.01 * np.sin(t)) + rng.normal(0, 0.02, n)
    open_ = np.r_[close[0], close[:-1]] + rng.normal(0, 0.01, n)
    high = np.maximum(open_, close) + rng.uniform(0, 0.3, n)
    low = np.minimum(open_, close) - rng.uniform(0, 0.3, n)
    vol = rng.uniform(1000, 5000, n)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "volume": vol,
        }
    )


@pytest.fixture
def ohlcv_flat(rng: np.random.Generator) -> pd.DataFrame:
    n = 200
    idx = pd.date_range("2024-01-01", periods=n, freq="h")
    close = 50.0 + rng.normal(0, 0.05, n)
    return pd.DataFrame(
        {
            "timestamp": idx,
            "open": close,
            "high": close + 0.02,
            "low": close - 0.02,
            "close": close,
            "volume": rng.uniform(500, 800, n),
        }
    )


@pytest.fixture
def ohlcv_no_timestamp_index_only(rng: np.random.Generator) -> pd.DataFrame:
    """OHLCV without timestamp column (uses synthetic timeline in Kronos adapter)."""
    n = 250
    close = 10 + np.cumsum(rng.normal(0.01, 0.05, n))
    return pd.DataFrame(
        {
            "open": close,
            "high": close + 0.02,
            "low": close - 0.02,
            "close": close,
            "volume": rng.uniform(100, 200, n),
        }
    )
