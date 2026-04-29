"""Technical indicators built on pandas (no extra deps)."""

from __future__ import annotations

import pandas as pd


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    """Compute Relative Strength Index (RSI) from close prices.

    Args:
        close (pd.Series): Close price time series.
        period (int): RSI lookback period.

    Returns:
        pd.Series: RSI values in ``[0, 100]`` with NaNs during warmup.
    """
    delta = close.diff()
    gain = delta.clip(lower=0.0)
    loss = (-delta).clip(lower=0.0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    out = 100 - (100 / (1 + rs))
    return out


def macd(close: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute MACD line, signal line, and histogram.

    Args:
        close (pd.Series): Close price time series.
        fast (int): Fast EMA span.
        slow (int): Slow EMA span.
        signal (int): Signal EMA span.

    Returns:
        tuple[pd.Series, pd.Series, pd.Series]: ``(macd_line, signal_line, hist)``.
    """
    ema_fast = close.ewm(span=fast, adjust=False).mean()
    ema_slow = close.ewm(span=slow, adjust=False).mean()
    line = ema_fast - ema_slow
    sig = line.ewm(span=signal, adjust=False).mean()
    hist = line - sig
    return line, sig, hist


def bollinger_bands(close: pd.Series, period: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Compute Bollinger Bands around rolling mean.

    Args:
        close (pd.Series): Close price time series.
        period (int): Rolling window length.
        num_std (float): Standard deviation multiplier.

    Returns:
        tuple[pd.Series, pd.Series, pd.Series]: ``(lower, middle, upper)`` bands.
    """
    mid = close.rolling(period, min_periods=period).mean()
    std = close.rolling(period, min_periods=period).std()
    upper = mid + num_std * std
    lower = mid - num_std * std
    return lower, mid, upper


def obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """Compute On-Balance Volume (OBV) cumulative indicator.

    Args:
        close (pd.Series): Close price series.
        volume (pd.Series): Volume series aligned to close prices.

    Returns:
        pd.Series: Cumulative OBV values.
    """
    direction = close.diff().apply(lambda x: 1 if x > 0 else (-1 if x < 0 else 0))
    return (direction * volume).fillna(0).cumsum()
