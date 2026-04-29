"""Simple probability / drift edge vs. current price (research-style heuristic)."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from strategies.base import Strategy
from strategies.schema import SignalDict
from strategies.utils import clamp_confidence, format_reason, hold_signal, validate_ohlcv_df


class ValueEdgeStrategy(Strategy):
    """Heuristic fair-value strategy based on drift/volatility edge.

    Responsibility:
        Estimates expected short-horizon move from log-return statistics and
        emits directional signals when estimated edge exceeds neutral bands.

    Key Attributes:
        lookback (int): Window for return statistics.
        horizon_bars (int): Forward horizon for fair-value projection.
        buy_edge_pct (float): Minimum positive edge for buy signal.
        sell_edge_pct (float): Maximum negative edge for sell signal.

    Interactions:
        - Consumed by backtester/ensemble as a generic ``Strategy``.
        - Uses helpers from ``strategies.utils`` for validation and formatting.
    """

    def __init__(
        self,
        lookback: int = 60,
        horizon_bars: int = 5,
        buy_edge_pct: float = 0.004,
        sell_edge_pct: float = -0.004,
    ):
        """Initialize value-edge strategy parameters.

        Args:
            lookback (int): Historical bars used for return estimation.
            horizon_bars (int): Horizon for expected move projection.
            buy_edge_pct (float): Positive edge threshold for buys.
            sell_edge_pct (float): Negative edge threshold for sells.

        Returns:
            None: Stores strategy configuration.
        """
        self.lookback = lookback
        self.horizon_bars = horizon_bars
        self.buy_edge_pct = buy_edge_pct
        self.sell_edge_pct = sell_edge_pct

    def generate_signal(self, df: pd.DataFrame) -> SignalDict:
        """Generate fair-value edge based trading signal.

        Args:
            df (pd.DataFrame): OHLCV frame with sufficient lookback.

        Returns:
            SignalDict: Buy/sell/hold output with confidence and reason.
        """
        err = validate_ohlcv_df(df, min_rows=self.lookback + 5)
        if err:
            return hold_signal(0.0, err)

        close = df["close"].astype(float)
        window = close.iloc[-self.lookback :]
        log_ret = np.log(window / window.shift(1)).dropna()
        if len(log_ret) < 10:
            return hold_signal(0.0, "Insufficient log returns in lookback")

        mu = float(log_ret.mean())
        sigma = float(log_ret.std(ddof=1)) if len(log_ret) > 1 else 0.0
        if sigma <= 0 or math.isnan(sigma):
            return hold_signal(0.0, "Volatility estimate is zero or invalid")

        expected_log_move = mu * float(self.horizon_bars)
        fair_ratio = math.exp(expected_log_move)
        last = float(close.iloc[-1])
        fair = last * fair_ratio
        edge_pct = (fair - last) / last if last else 0.0

        # Map edge vs. sigma to a bounded confidence
        z = edge_pct / max(sigma * math.sqrt(float(self.horizon_bars)), 1e-9)
        conf = clamp_confidence(min(1.0, abs(z) / 2.5))

        if edge_pct >= self.buy_edge_pct:
            reason = format_reason(
                [
                    ("edge_pct", round(edge_pct * 100, 4)),
                    ("mu_log_per_bar", round(mu, 6)),
                    ("sigma_log", round(sigma, 6)),
                    ("horizon_bars", self.horizon_bars),
                    ("fair_vs_last", round(fair_ratio - 1.0, 6)),
                ]
            )
            return {"signal": "buy", "confidence": conf, "reason": reason}

        if edge_pct <= self.sell_edge_pct:
            reason = format_reason(
                [
                    ("edge_pct", round(edge_pct * 100, 4)),
                    ("mu_log_per_bar", round(mu, 6)),
                    ("sigma_log", round(sigma, 6)),
                    ("horizon_bars", self.horizon_bars),
                    ("fair_vs_last", round(fair_ratio - 1.0, 6)),
                ]
            )
            return {"signal": "sell", "confidence": conf, "reason": reason}

        return hold_signal(
            clamp_confidence(abs(z) / 5.0),
            format_reason(
                [
                    ("edge_pct", round(edge_pct * 100, 4)),
                    ("note", "inside neutral band"),
                ]
            ),
        )
