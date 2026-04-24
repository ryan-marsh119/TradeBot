"""Run multiple strategies and aggregate confidence-weighted votes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import pandas as pd

from strategies.base import Strategy
from strategies.schema import SignalDict, SignalLiteral
from strategies.utils import clamp_confidence, format_reason, hold_signal


@dataclass
class WeightedStrategy:
    strategy: Strategy
    weight: float = 1.0


def _score_for_signal(signal: SignalLiteral) -> float:
    if signal == "buy":
        return 1.0
    if signal == "sell":
        return -1.0
    return 0.0


def run_ensemble(df: pd.DataFrame, members: Iterable[WeightedStrategy]) -> SignalDict:
    """Average signed vote using confidence * weight; ties become hold."""
    items = list(members)
    if not items:
        return hold_signal(0.0, "ensemble has no strategies")

    total_w = sum(max(m.weight, 0.0) for m in items) or 1.0
    acc = 0.0
    parts: list[tuple[str, object]] = []

    for m in items:
        out = m.strategy.safe_generate(df)
        direction = _score_for_signal(out["signal"])
        contrib = direction * clamp_confidence(out["confidence"]) * m.weight
        acc += contrib
        parts.append((m.strategy.__class__.__name__, f"{out['signal']}:{out['confidence']:.2f}"))

    score = acc / total_w
    conf = clamp_confidence(abs(score))

    if score > 0.15:
        signal: SignalLiteral = "buy"
    elif score < -0.15:
        signal = "sell"
    else:
        return hold_signal(conf, format_reason([("ensemble_score", round(score, 4)), ("votes", parts)]))

    return {
        "signal": signal,
        "confidence": conf,
        "reason": format_reason([("ensemble_score", round(score, 4)), ("votes", parts)]),
    }
