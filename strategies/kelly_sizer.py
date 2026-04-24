"""Kelly-style sizing with hard caps (usable with strategy confidence)."""

from __future__ import annotations

import math
from typing import Literal

from strategies.utils import clamp_confidence


def _clamp(x: float, lo: float, hi: float) -> float:
    if math.isnan(x) or math.isinf(x):
        return lo
    return max(lo, min(hi, float(x)))


def kelly_fraction(
    win_prob: float,
    win_loss_ratio: float,
    *,
    max_fraction: float = 0.25,
    min_fraction: float = 0.0,
    half_kelly: bool = True,
) -> float:
    """
    Classic binary-outcome Kelly: f* = (p(b+1) - 1) / b with b = win/loss size ratio.

    Here `win_loss_ratio` is payoff per unit risk (reward/risk), e.g. 1.5 means +1.5 on win, -1 on loss.
    """
    if not (0.0 <= win_prob <= 1.0) or math.isnan(win_prob):
        return min_fraction
    if win_loss_ratio <= 0 or math.isnan(win_loss_ratio):
        return min_fraction

    b = float(win_loss_ratio)
    p = float(win_prob)
    q = 1.0 - p
    raw = (b * p - q) / b if b else 0.0
    if half_kelly:
        raw *= 0.5
    if raw < 0:
        raw = 0.0
    return _clamp(raw, min_fraction, max_fraction)


def size_from_confidence(
    confidence: float,
    *,
    signal: Literal["buy", "sell", "hold"] = "buy",
    max_fraction: float = 0.2,
    min_fraction: float = 0.0,
) -> float:
    """Map strategy confidence [0,1] to a suggested position fraction (direction-agnostic magnitude)."""
    if signal == "hold":
        return min_fraction
    c = clamp_confidence(confidence)
    return clamp_confidence(c * max_fraction, lo=min_fraction, hi=max_fraction)


def combined_size(
    win_prob: float,
    win_loss_ratio: float,
    confidence: float,
    *,
    signal: Literal["buy", "sell", "hold"] = "buy",
    max_fraction: float = 0.2,
) -> float:
    """Blend Kelly (from prob/odds) with confidence cap."""
    if signal == "hold":
        return 0.0
    k = kelly_fraction(win_prob, win_loss_ratio, max_fraction=max_fraction, half_kelly=True)
    c = clamp_confidence(confidence)
    return clamp_confidence(k * c, lo=0.0, hi=max_fraction)
