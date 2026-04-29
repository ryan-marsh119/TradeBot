"""Paper broker that mirrors live execution interface."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from core.backtest_schema import ExecutionResult, SignalInput


@dataclass
class PositionState:
    qty: float = 0.0
    avg_entry: float = 0.0


class PaperBroker:
    """Simulate spot fills with fees, delay, and slippage."""

    def __init__(
        self,
        *,
        initial_cash: float = 10_000.0,
        fee_bps_taker: float = 10.0,
        fee_bps_maker: float = 5.0,
        slippage_bps_fixed: float = 3.0,
        slippage_vol_multiplier: float = 0.75,
        min_qty: float = 1e-8,
        allow_negative_cash: bool = False,
        supported_symbols: set[str] | None = None,
    ) -> None:
        self.initial_cash = float(initial_cash)
        self.cash = float(initial_cash)
        self.fee_bps_taker = float(fee_bps_taker)
        self.fee_bps_maker = float(fee_bps_maker)
        self.slippage_bps_fixed = float(slippage_bps_fixed)
        self.slippage_vol_multiplier = float(slippage_vol_multiplier)
        self.min_qty = float(min_qty)
        self.allow_negative_cash = allow_negative_cash
        self.supported_symbols = supported_symbols or set()
        self.positions: dict[str, PositionState] = {}
        self.realized_pnl = 0.0
        self.total_fees = 0.0

    def _position(self, symbol: str) -> PositionState:
        if symbol not in self.positions:
            self.positions[symbol] = PositionState()
        return self.positions[symbol]

    @staticmethod
    def _volatility_bps(bar: dict[str, Any]) -> float:
        high = float(bar.get("high", 0.0))
        low = float(bar.get("low", 0.0))
        close = max(float(bar.get("close", 0.0)), 1e-12)
        return max((high - low) / close * 10_000.0, 0.0)

    def _slippage_bps(self, bar: dict[str, Any], confidence: float) -> float:
        vol_bps = self._volatility_bps(bar)
        conf_scale = 1.0 + max(min(float(confidence), 1.0), 0.0) * 0.5
        return self.slippage_bps_fixed + (vol_bps * self.slippage_vol_multiplier / 100.0) * conf_scale

    def _mark_to_market(self, symbol: str, mark_price: float) -> tuple[float, float]:
        pos = self._position(symbol)
        unrealized = (mark_price - pos.avg_entry) * pos.qty if pos.qty > 0 else 0.0
        equity = self.cash + (pos.qty * mark_price)
        return unrealized, equity

    def execute_signal(
        self,
        signal: SignalInput,
        *,
        current_bar: dict[str, Any],
        next_bar: dict[str, Any],
    ) -> ExecutionResult:
        """Execute a single signal using next-bar open fill model."""
        symbol = signal["symbol"]
        side = signal["side"]
        conf = float(signal["confidence"])
        pos = self._position(symbol)
        if self.supported_symbols and symbol not in self.supported_symbols:
            unrealized, equity = self._mark_to_market(symbol, float(current_bar["close"]))
            return {
                "status": "rejected",
                "reason": "unsupported_symbol",
                "symbol": symbol,
                "side": side,
                "requested_qty": 0.0,
                "filled_qty": 0.0,
                "fill_price": 0.0,
                "benchmark_price": float(current_bar["close"]),
                "fee": 0.0,
                "slippage_bps": 0.0,
                "cash_before": self.cash,
                "cash_after": self.cash,
                "position_before": pos.qty,
                "position_after": pos.qty,
                "realized_pnl": self.realized_pnl,
                "unrealized_pnl": unrealized,
                "total_equity": equity,
            }

        cash_before = self.cash
        position_before = pos.qty

        if side == "hold":
            unrealized, equity = self._mark_to_market(symbol, float(current_bar["close"]))
            return {
                "status": "ignored",
                "reason": "hold_signal",
                "symbol": symbol,
                "side": side,
                "requested_qty": 0.0,
                "filled_qty": 0.0,
                "fill_price": 0.0,
                "benchmark_price": float(current_bar["close"]),
                "fee": 0.0,
                "slippage_bps": 0.0,
                "cash_before": cash_before,
                "cash_after": self.cash,
                "position_before": position_before,
                "position_after": pos.qty,
                "realized_pnl": self.realized_pnl,
                "unrealized_pnl": unrealized,
                "total_equity": equity,
            }

        benchmark_price = float(next_bar["open"])
        slippage_bps = self._slippage_bps(current_bar, conf)
        slip_mult = 1.0 + (slippage_bps / 10_000.0 if side == "buy" else -slippage_bps / 10_000.0)
        fill_price = benchmark_price * slip_mult

        # Confidence scales sizing but keeps max notional bounded.
        deploy_ratio = min(max(conf, 0.0), 1.0) * 0.95
        requested_qty = (self.cash * deploy_ratio / fill_price) if side == "buy" else pos.qty * deploy_ratio
        requested_qty = float(np.round(requested_qty, 8))
        if requested_qty < self.min_qty:
            unrealized, equity = self._mark_to_market(symbol, float(current_bar["close"]))
            return {
                "status": "rejected",
                "reason": "qty_below_min",
                "symbol": symbol,
                "side": side,
                "requested_qty": requested_qty,
                "filled_qty": 0.0,
                "fill_price": fill_price,
                "benchmark_price": benchmark_price,
                "fee": 0.0,
                "slippage_bps": slippage_bps,
                "cash_before": cash_before,
                "cash_after": self.cash,
                "position_before": position_before,
                "position_after": pos.qty,
                "realized_pnl": self.realized_pnl,
                "unrealized_pnl": unrealized,
                "total_equity": equity,
            }

        meta = signal.get("strategy_meta", {})
        fee_bps = self.fee_bps_maker if bool(meta.get("maker", False)) else self.fee_bps_taker
        fee = requested_qty * fill_price * (fee_bps / 10_000.0)
        if side == "buy":
            cost = requested_qty * fill_price + fee
            if not self.allow_negative_cash and cost > self.cash:
                unrealized, equity = self._mark_to_market(symbol, float(current_bar["close"]))
                return {
                    "status": "rejected",
                    "reason": "insufficient_cash",
                    "symbol": symbol,
                    "side": side,
                    "requested_qty": requested_qty,
                    "filled_qty": 0.0,
                    "fill_price": fill_price,
                    "benchmark_price": benchmark_price,
                    "fee": fee,
                    "slippage_bps": slippage_bps,
                    "cash_before": cash_before,
                    "cash_after": self.cash,
                    "position_before": position_before,
                    "position_after": pos.qty,
                    "realized_pnl": self.realized_pnl,
                    "unrealized_pnl": unrealized,
                    "total_equity": equity,
                }
            prev_cost = pos.qty * pos.avg_entry
            new_qty = pos.qty + requested_qty
            pos.avg_entry = (prev_cost + requested_qty * fill_price) / max(new_qty, 1e-12)
            pos.qty = new_qty
            self.cash -= cost
        else:
            sell_qty = min(requested_qty, pos.qty)
            if sell_qty < self.min_qty:
                unrealized, equity = self._mark_to_market(symbol, float(current_bar["close"]))
                return {
                    "status": "rejected",
                    "reason": "insufficient_position",
                    "symbol": symbol,
                    "side": side,
                    "requested_qty": requested_qty,
                    "filled_qty": 0.0,
                    "fill_price": fill_price,
                    "benchmark_price": benchmark_price,
                    "fee": 0.0,
                    "slippage_bps": slippage_bps,
                    "cash_before": cash_before,
                    "cash_after": self.cash,
                    "position_before": position_before,
                    "position_after": pos.qty,
                    "realized_pnl": self.realized_pnl,
                    "unrealized_pnl": unrealized,
                    "total_equity": equity,
                }
            proceeds = sell_qty * fill_price - fee
            pnl = (fill_price - pos.avg_entry) * sell_qty
            self.realized_pnl += pnl
            pos.qty -= sell_qty
            if pos.qty <= self.min_qty:
                pos.qty = 0.0
                pos.avg_entry = 0.0
            self.cash += proceeds
            requested_qty = sell_qty

        self.total_fees += fee
        unrealized, equity = self._mark_to_market(symbol, float(next_bar["close"]))
        return {
            "status": "filled",
            "reason": "ok",
            "symbol": symbol,
            "side": side,
            "requested_qty": requested_qty,
            "filled_qty": requested_qty,
            "fill_price": fill_price,
            "benchmark_price": benchmark_price,
            "fee": fee,
            "slippage_bps": slippage_bps,
            "cash_before": cash_before,
            "cash_after": self.cash,
            "position_before": position_before,
            "position_after": pos.qty,
            "realized_pnl": self.realized_pnl,
            "unrealized_pnl": unrealized,
            "total_equity": equity,
        }
