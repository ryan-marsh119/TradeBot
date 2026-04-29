"""Shared helpers for strategies."""

from __future__ import annotations

import math
from typing import Any

import pandas as pd

from strategies.schema import REQUIRED_OHLCV_COLUMNS, SignalDict


def clamp_confidence(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    """Clamp a numeric confidence score to a bounded interval.

    Args:
        x (float): Candidate score.
        lo (float): Lower bound.
        hi (float): Upper bound.

    Returns:
        float: Bounded score, or ``lo`` for NaN/inf values.
    """
    if math.isnan(x) or math.isinf(x):
        return lo
    return max(lo, min(hi, float(x)))


def validate_ohlcv_df(df: pd.DataFrame, min_rows: int = 1) -> str | None:
    """Validate OHLCV DataFrame shape and required fields.

    Args:
        df (pd.DataFrame): Candidate input frame.
        min_rows (int): Minimum required number of rows.

    Returns:
        str | None: Error reason when invalid, else ``None``.
    """
    if df is None or not isinstance(df, pd.DataFrame):
        return "Input is not a DataFrame"
    if df.empty:
        return "DataFrame is empty"
    if len(df) < min_rows:
        return f"Need at least {min_rows} rows, got {len(df)}"
    cols = set(df.columns)
    missing = REQUIRED_OHLCV_COLUMNS - cols
    if missing:
        return f"Missing columns: {sorted(missing)}"
    return None


def hold_signal(confidence: float, reason: str) -> SignalDict:
    """Build a normalized hold signal payload.

    Args:
        confidence (float): Confidence score to clamp.
        reason (str): Explanation for non-trading decision.

    Returns:
        SignalDict: Hold signal dictionary.
    """
    return {
        "signal": "hold",
        "confidence": clamp_confidence(confidence),
        "reason": reason,
    }


def format_reason(parts: list[tuple[str, Any]]) -> str:
    """Format key-value evidence pairs into one reason string.

    Args:
        parts (list[tuple[str, Any]]): Ordered key-value tuples.

    Returns:
        str: Semicolon-delimited human-readable summary.
    """
    return "; ".join(f"{k}={v}" for k, v in parts if v is not None)
