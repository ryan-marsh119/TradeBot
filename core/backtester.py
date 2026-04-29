"""Event-driven backtester that orchestrates strategies and paper broker."""

from __future__ import annotations

import hashlib
import json
import random
import traceback
import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from time import perf_counter
from typing import Any

import pandas as pd
import structlog
from structlog.processors import JSONRenderer
from structlog.stdlib import add_log_level

from core.backtest_schema import LedgerEvent, SignalInput
from core.paper_broker import PaperBroker
from core.trade_ledger import TradeLedger
from strategies.base import Strategy

logger = structlog.get_logger(__name__)


@dataclass
class PendingOrder:
    """Queued order awaiting delayed execution in event loop.

    Attributes:
        execute_at_idx (int): Bar index at which execution should be attempted.
        signal (SignalInput): Canonical signal payload selected for execution.
        signal_payload (dict[str, Any]): Serializable signal snapshot for logs.
    """

    execute_at_idx: int
    signal: SignalInput
    signal_payload: dict[str, Any]


class Backtester:
    """Coordinate strategy evaluation, broker execution, and ledger persistence.

    Responsibility:
        Runs chronological event-driven simulations with optional warmup and
        delayed fills, while capturing rich event telemetry for replay.

    Key Attributes:
        ledger (TradeLedger): Storage sink for run metadata and events.
        broker (PaperBroker): State machine handling fill simulation.
        warmup_bars (int): Bars required before strategy output is tradable.
        delay_bars (int): Bars to delay order execution after signal selection.

    Interactions:
        - Pulls signals from any object implementing ``Strategy``.
        - Routes selected orders through ``PaperBroker``.
        - Emits canonical events via ``TradeLedger`` and structlog.
    """

    def __init__(
        self,
        *,
        ledger: TradeLedger,
        broker: PaperBroker,
        warmup_bars: int = 80,
        delay_bars: int = 1,
    ) -> None:
        """Initialize backtester orchestration dependencies and settings.

        Args:
            ledger (TradeLedger): Event and run metadata persistence backend.
            broker (PaperBroker): Paper execution engine instance.
            warmup_bars (int): Number of bars to skip before trading.
            delay_bars (int): Number of bars to delay fills.

        Returns:
            None: Creates runtime state for one backtester instance.
        """
        self.ledger = ledger
        self.broker = broker
        self.warmup_bars = warmup_bars
        self.delay_bars = max(delay_bars, 1)
        self._event_counter = 0
        self._last_side: str | None = None
        self._configure_structlog()

    @staticmethod
    def _configure_structlog() -> None:
        """Configure JSON structured logging for backtest events.

        Returns:
            None: Applies global structlog processor pipeline.
        """
        structlog.configure(
            processors=[
                add_log_level,
                structlog.processors.TimeStamper(fmt="iso", utc=True),
                JSONRenderer(),
            ]
        )

    def _next_event_id(self) -> str:
        """Generate the next sequential event ID for current run context.

        Returns:
            str: Event ID formatted as ``evt-XXXXXXXXXX``.
        """
        self._event_counter += 1
        return f"evt-{self._event_counter:010d}"

    def _emit(
        self,
        *,
        run_id: str,
        event_type: str,
        bar_time: datetime,
        symbol: str,
        strategy: str,
        strategy_version: str,
        payload: dict[str, Any],
        latency_ms: int = 0,
        error: str | None = None,
    ) -> None:
        """Build and persist one canonical ledger event.

        Args:
            run_id (str): Run identifier.
            event_type (str): Lifecycle event type.
            bar_time (datetime): Event timestamp.
            symbol (str): Symbol context.
            strategy (str): Strategy/system origin.
            strategy_version (str): Strategy version label.
            payload (dict[str, Any]): Event payload body.
            latency_ms (int): Optional measured latency.
            error (str | None): Optional error summary.

        Returns:
            None: Writes event to ledger and emits structlog message.
        """
        event: LedgerEvent = {
            "run_id": run_id,
            "event_id": self._next_event_id(),
            "event_type": event_type,  # type: ignore[typeddict-item]
            "bar_time": bar_time,
            "symbol": symbol,
            "strategy": strategy,
            "strategy_version": strategy_version,
            "latency_ms": latency_ms,
            "payload_json": json.dumps(payload, default=str),
            "error": error,
        }
        self.ledger.log_event(event)
        logger.info(
            "backtest_event",
            run_id=run_id,
            event_type=event_type,
            symbol=symbol,
            strategy=strategy,
            bar_time=bar_time.isoformat(),
        )

    @staticmethod
    def _strategy_version(strategy: Strategy) -> str:
        """Resolve strategy version identifier with safe default.

        Args:
            strategy (Strategy): Strategy instance.

        Returns:
            str: Version string, defaulting to ``v1``.
        """
        return getattr(strategy, "version", "v1")

    @staticmethod
    def _config_hash(payload: dict[str, Any]) -> str:
        """Compute a short deterministic hash for run configuration.

        Args:
            payload (dict[str, Any]): Serializable run configuration.

        Returns:
            str: First 16 hex chars of SHA-256 digest.
        """
        return hashlib.sha256(json.dumps(payload, default=str, sort_keys=True).encode("utf-8")).hexdigest()[:16]

    def _choose_signal(self, candidates: list[SignalInput]) -> SignalInput | None:
        """Select one tradable signal from same-bar strategy outputs.

        Args:
            candidates (list[SignalInput]): Candidate signals from strategies.

        Returns:
            SignalInput | None: Selected buy/sell signal, or ``None`` on tie/no-op.

        Notes:
            Opposing buy/sell signals within 0.05 confidence are treated as
            conflict and suppressed.
        """
        tradable = [s for s in candidates if s["side"] in {"buy", "sell"}]
        if not tradable:
            return None

        buy_best = max((s for s in tradable if s["side"] == "buy"), key=lambda x: x["confidence"], default=None)
        sell_best = max((s for s in tradable if s["side"] == "sell"), key=lambda x: x["confidence"], default=None)
        if buy_best and sell_best:
            if abs(buy_best["confidence"] - sell_best["confidence"]) <= 0.05:
                return None
            return buy_best if buy_best["confidence"] > sell_best["confidence"] else sell_best
        return buy_best or sell_best

    def run(
        self,
        *,
        df: pd.DataFrame,
        symbol: str,
        timeframe: str,
        strategies: list[Strategy],
        run_id: str | None = None,
        random_seed: int = 42,
    ) -> dict[str, Any]:
        """Execute a full backtest run over historical bars.

        Args:
            df (pd.DataFrame): Historical OHLCV frame including ``timestamp``.
            symbol (str): Symbol being simulated.
            timeframe (str): Timeframe label for run metadata.
            strategies (list[Strategy]): Strategy instances to evaluate per bar.
            run_id (str | None): Optional deterministic run identifier.
            random_seed (int): Seed for deterministic run behavior.

        Returns:
            dict[str, Any]: Run summary metrics and identifiers.

        Raises:
            ValueError: If required timestamp column is missing or data is short.

        Notes:
            Continues across strategy failures by logging ``error`` events.
        """
        if "timestamp" not in df.columns:
            raise ValueError("DataFrame must contain timestamp column")
        if len(df) <= self.warmup_bars + self.delay_bars + 1:
            raise ValueError("Insufficient data length for warmup + delay")

        random.seed(random_seed)
        run_id = run_id or f"bt-{uuid.uuid4().hex[:12]}"
        run_cfg = {
            "symbol": symbol,
            "timeframe": timeframe,
            "start_time": pd.to_datetime(df["timestamp"].iloc[0]).isoformat(),
            "end_time": pd.to_datetime(df["timestamp"].iloc[-1]).isoformat(),
            "random_seed": random_seed,
            "warmup_bars": self.warmup_bars,
            "initial_cash": self.broker.initial_cash,
            "allow_negative_cash": self.broker.allow_negative_cash,
            "config_hash": "",
        }
        run_cfg["config_hash"] = self._config_hash(run_cfg)
        self.ledger.log_run_start(run_id, run_cfg)

        self._emit(
            run_id=run_id,
            event_type="run_started",
            bar_time=datetime.now(tz=UTC),
            symbol=symbol,
            strategy="system",
            strategy_version="v1",
            payload=run_cfg,
        )

        pending: list[PendingOrder] = []
        errors = 0
        total_trades = 0
        equity_curve: list[float] = []

        for idx in range(self.warmup_bars, len(df) - 1):
            current_bar = df.iloc[idx].to_dict()
            current_ts = pd.to_datetime(current_bar["timestamp"]).to_pydatetime()
            next_bar = df.iloc[idx + 1].to_dict()

            due_orders = [o for o in pending if o.execute_at_idx == idx]
            pending = [o for o in pending if o.execute_at_idx != idx]
            for order in due_orders:
                start = perf_counter()
                result = self.broker.execute_signal(order.signal, current_bar=current_bar, next_bar=next_bar)
                latency_ms = int((perf_counter() - start) * 1000)
                self._emit(
                    run_id=run_id,
                    event_type="fill_applied",
                    bar_time=current_ts,
                    symbol=symbol,
                    strategy=order.signal["strategy_name"],
                    strategy_version=order.signal["strategy_version"],
                    payload={"signal": order.signal_payload, "execution": result},
                    latency_ms=latency_ms,
                )
                self._emit(
                    run_id=run_id,
                    event_type="position_updated",
                    bar_time=current_ts,
                    symbol=symbol,
                    strategy=order.signal["strategy_name"],
                    strategy_version=order.signal["strategy_version"],
                    payload={
                        "cash": result["cash_after"],
                        "position_qty": result["position_after"],
                        "realized_pnl": result["realized_pnl"],
                    },
                )
                self._emit(
                    run_id=run_id,
                    event_type="equity_snapshot",
                    bar_time=current_ts,
                    symbol=symbol,
                    strategy=order.signal["strategy_name"],
                    strategy_version=order.signal["strategy_version"],
                    payload={
                        "equity": result["total_equity"],
                        "unrealized_pnl": result["unrealized_pnl"],
                        "cash": result["cash_after"],
                    },
                )
                equity_curve.append(float(result["total_equity"]))
                if result["status"] == "filled":
                    total_trades += 1
                    self._last_side = result["side"]

            window_df = df.iloc[: idx + 1]
            signals: list[SignalInput] = []
            for strategy in strategies:
                strategy_name = strategy.__class__.__name__
                strategy_version = self._strategy_version(strategy)
                try:
                    start = perf_counter()
                    out = strategy.safe_generate(window_df)
                    latency_ms = int((perf_counter() - start) * 1000)
                    signal: SignalInput = {
                        "symbol": symbol,
                        "side": out["signal"],
                        "confidence": float(out["confidence"]),
                        "reason": out["reason"],
                        "timestamp": current_ts,
                        "strategy_name": strategy_name,
                        "strategy_version": strategy_version,
                        "strategy_meta": {
                            "kronos_forecast": getattr(strategy, "last_forecast_summary", None),
                        },
                    }
                    signals.append(signal)
                    self._emit(
                        run_id=run_id,
                        event_type="signal_received",
                        bar_time=current_ts,
                        symbol=symbol,
                        strategy=strategy_name,
                        strategy_version=strategy_version,
                        payload={"signal": signal},
                        latency_ms=latency_ms,
                    )
                except Exception as exc:
                    errors += 1
                    self._emit(
                        run_id=run_id,
                        event_type="error",
                        bar_time=current_ts,
                        symbol=symbol,
                        strategy=strategy_name,
                        strategy_version=strategy_version,
                        payload={"traceback": traceback.format_exc()},
                        error=str(exc),
                    )

            chosen = self._choose_signal(signals)
            if chosen is None:
                continue
            if self._last_side == chosen["side"]:
                continue

            pending_order = PendingOrder(
                execute_at_idx=min(idx + self.delay_bars, len(df) - 2),
                signal=chosen,
                signal_payload=dict(chosen),
            )
            pending.append(pending_order)
            self._emit(
                run_id=run_id,
                event_type="order_simulated",
                bar_time=current_ts,
                symbol=symbol,
                strategy=chosen["strategy_name"],
                strategy_version=chosen["strategy_version"],
                payload={"queued_for_idx": pending_order.execute_at_idx, "signal": chosen},
            )

        if not equity_curve:
            equity_curve = [self.broker.initial_cash]
        peak = equity_curve[0]
        max_dd = 0.0
        for eq in equity_curve:
            peak = max(peak, eq)
            dd = (peak - eq) / peak if peak else 0.0
            max_dd = max(max_dd, dd)

        summary = {
            "total_trades": total_trades,
            "net_pnl": float(self.broker.cash - self.broker.initial_cash),
            "max_drawdown": max_dd,
            "errors": errors,
            "equity_final": equity_curve[-1],
            "run_id": run_id,
        }
        self.ledger.log_run_end(
            run_id,
            total_trades=summary["total_trades"],
            net_pnl=summary["net_pnl"],
            max_drawdown=summary["max_drawdown"],
            errors=summary["errors"],
        )
        self._emit(
            run_id=run_id,
            event_type="run_finished",
            bar_time=datetime.now(tz=UTC),
            symbol=symbol,
            strategy="system",
            strategy_version="v1",
            payload=summary,
        )
        return summary

    def validate_run_outputs(self, run_id: str, *, allow_negative_cash: bool = False) -> dict[str, Any]:
        """Run acceptance checks for event coverage and portfolio invariants.

        Args:
            run_id (str): Run identifier to validate.
            allow_negative_cash (bool): Whether negative cash is acceptable.

        Returns:
            dict[str, Any]: Validation summary with counts, violations, and status.
        """
        counts = self.ledger.count_events(run_id)
        required = ("signal_received", "equity_snapshot")
        missing = [event_type for event_type in required if counts.get(event_type, 0) == 0]

        equity_events = self.ledger.fetch_events(run_id, event_type="equity_snapshot")
        equities: list[float] = []
        cash_values: list[float] = []
        for event in equity_events:
            payload = json.loads(event["payload_json"])
            equities.append(float(payload.get("equity", 0.0)))
            cash_values.append(float(payload.get("cash", 0.0)))

        equity_non_positive = any(eq <= 0 for eq in equities)
        negative_cash = any(c < 0 for c in cash_values)
        violations: list[str] = []
        if missing:
            violations.append(f"missing event types: {', '.join(missing)}")
        if equity_non_positive:
            violations.append("equity contains non-positive values")
        if negative_cash and not allow_negative_cash:
            violations.append("cash dropped below zero")

        return {
            "run_id": run_id,
            "event_counts": counts,
            "equity_points": len(equities),
            "violations": violations,
            "ok": len(violations) == 0,
        }
