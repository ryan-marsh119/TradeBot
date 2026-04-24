"""Unified signal schema for all strategies."""

from typing import Literal, TypedDict

SignalLiteral = Literal["buy", "sell", "hold"]

REQUIRED_OHLCV_COLUMNS = frozenset({"open", "high", "low", "close", "volume"})


class SignalDict(TypedDict):
    """Standard strategy output."""

    signal: SignalLiteral
    confidence: float
    reason: str
