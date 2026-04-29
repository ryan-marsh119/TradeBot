"""Canonical contracts for backtesting execution and ledger events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal, NotRequired, TypedDict

from strategies.schema import SignalLiteral

EventType = Literal[
    "signal_received",
    "order_simulated",
    "fill_applied",
    "position_updated",
    "equity_snapshot",
    "error",
    "run_started",
    "run_finished",
]

OrderSide = Literal["buy", "sell"]
ExecutionStatus = Literal["filled", "rejected", "ignored"]


class SignalInput(TypedDict):
    """Standardized strategy signal payload for broker execution."""

    symbol: str
    side: SignalLiteral
    confidence: float
    reason: str
    timestamp: datetime
    strategy_name: str
    strategy_version: str
    strategy_meta: NotRequired[dict[str, Any]]


class ExecutionResult(TypedDict):
    """Broker output for simulated execution."""

    status: ExecutionStatus
    reason: str
    symbol: str
    side: str
    requested_qty: float
    filled_qty: float
    fill_price: float
    benchmark_price: float
    fee: float
    slippage_bps: float
    cash_before: float
    cash_after: float
    position_before: float
    position_after: float
    realized_pnl: float
    unrealized_pnl: float
    total_equity: float


class LedgerEvent(TypedDict):
    """Canonical shape persisted to ledger table and CSV."""

    run_id: str
    event_id: str
    event_type: EventType
    bar_time: datetime
    symbol: str
    strategy: str
    strategy_version: str
    latency_ms: int
    payload_json: str
    error: str | None


@dataclass(frozen=True)
class BacktestRunConfig:
    run_id: str
    symbol: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    random_seed: int
    warmup_bars: int
    initial_cash: float
    allow_negative_cash: bool = False
