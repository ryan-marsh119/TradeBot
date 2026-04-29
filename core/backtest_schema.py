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
    """Canonical signal payload routed from strategy layer to broker.

    Attributes:
        symbol (str): Traded symbol.
        side (SignalLiteral): Desired action: buy, sell, or hold.
        confidence (float): Confidence score in ``[0, 1]``.
        reason (str): Human-readable explanation for auditability.
        timestamp (datetime): Signal timestamp associated with a bar.
        strategy_name (str): Strategy class or identifier.
        strategy_version (str): Version tag for reproducibility.
        strategy_meta (dict[str, Any]): Optional extra metadata.
    """

    symbol: str
    side: SignalLiteral
    confidence: float
    reason: str
    timestamp: datetime
    strategy_name: str
    strategy_version: str
    strategy_meta: NotRequired[dict[str, Any]]


class ExecutionResult(TypedDict):
    """Normalized execution output emitted by paper broker.

    Attributes:
        status (ExecutionStatus): Filled, rejected, or ignored.
        reason (str): Machine-readable reason code.
        symbol (str): Traded symbol.
        side (str): Requested side.
        requested_qty (float): Quantity requested by sizing logic.
        filled_qty (float): Quantity actually filled.
        fill_price (float): Effective fill price after slippage.
        benchmark_price (float): Reference price before slippage.
        fee (float): Fees charged for this execution.
        slippage_bps (float): Applied slippage in basis points.
        cash_before (float): Cash before execution.
        cash_after (float): Cash after execution.
        position_before (float): Position quantity before execution.
        position_after (float): Position quantity after execution.
        realized_pnl (float): Cumulative realized PnL after execution.
        unrealized_pnl (float): Mark-to-market unrealized PnL.
        total_equity (float): Portfolio equity snapshot.
    """

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
    """Canonical event record persisted by trade ledger.

    Attributes:
        run_id (str): Backtest run identifier.
        event_id (str): Stable event identifier unique within run.
        event_type (EventType): Type of lifecycle event.
        bar_time (datetime): Bar timestamp associated with event.
        symbol (str): Symbol context.
        strategy (str): Strategy generating or owning the event.
        strategy_version (str): Strategy version for reproducibility.
        latency_ms (int): Measured processing latency.
        payload_json (str): JSON payload body as serialized string.
        error (str | None): Optional error message.
    """

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
    """Immutable metadata snapshot describing a single backtest run.

    This object documents minimal reproducibility inputs used by the simulator
    and ledger tables.
    """

    run_id: str
    symbol: str
    timeframe: str
    start_time: datetime
    end_time: datetime
    random_seed: int
    warmup_bars: int
    initial_cash: float
    allow_negative_cash: bool = False
