"""Unified signal schema for all strategies."""

from typing import Literal, TypedDict

SignalLiteral = Literal["buy", "sell", "hold"]

REQUIRED_OHLCV_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


class SignalDict(TypedDict):
    """Standard strategy output contract shared across orchestration layers.

    Attributes:
        signal (SignalLiteral): Directional action emitted by strategy.
        confidence (float): Confidence score expected in ``[0, 1]``.
        reason (str): Human-readable explanation for audit/debug contexts.
    """

    signal: SignalLiteral
    confidence: float
    reason: str
