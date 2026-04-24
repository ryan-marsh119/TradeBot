"""Multi-horizon momentum from daily return streaks + RSI/MACD hints."""

from __future__ import annotations

import pandas as pd

from strategies.base import Strategy
from strategies.indicators import macd, rsi
from strategies.schema import SignalDict
from strategies.utils import clamp_confidence, format_reason, hold_signal, validate_ohlcv_df


def _daily_close_series(df: pd.DataFrame) -> pd.Series:
    if "timestamp" in df.columns:
        idx = pd.to_datetime(df["timestamp"])
    else:
        idx = pd.to_datetime(df.index)
    work = df.copy()
    work = work.set_index(idx)
    return work["close"].resample("1D").last().dropna()


def _streak_from_newest(daily_returns: pd.Series, positive: bool) -> int:
    rev = daily_returns.dropna().iloc[::-1]
    streak = 0
    for v in rev:
        if positive and v > 0:
            streak += 1
        elif not positive and v < 0:
            streak += 1
        else:
            break
    return streak


def _weighted_streak_component(streak_len: int) -> float:
    """7/14/30-day style weights on capped streak length."""
    s7 = min(streak_len, 7) / 7.0
    s14 = min(streak_len, 14) / 14.0
    s30 = min(streak_len, 30) / 30.0
    return 0.5 * s7 + 0.3 * s14 + 0.2 * s30


class MomentumStrategy(Strategy):
    """Combines 7/14/30-day streak structure with RSI/MACD confidence tweaks."""

    def __init__(
        self,
        buy_threshold: float = 0.18,
        sell_threshold: float = -0.18,
    ):
        self.buy_threshold = buy_threshold
        self.sell_threshold = sell_threshold

    def generate_signal(self, df: pd.DataFrame) -> SignalDict:
        err = validate_ohlcv_df(df, min_rows=64)
        if err:
            return hold_signal(0.0, err)

        try:
            daily_close = _daily_close_series(df)
        except (TypeError, ValueError) as e:
            return hold_signal(0.0, f"Could not build daily series: {e}")

        if len(daily_close) < 14:
            return hold_signal(0.0, "Insufficient daily history after resampling")

        daily_ret = daily_close.pct_change()
        pu = _streak_from_newest(daily_ret, True)
        nu = _streak_from_newest(daily_ret, False)
        combined = _weighted_streak_component(pu) - _weighted_streak_component(nu)
        combined = max(-1.0, min(1.0, combined))

        close = df["close"]
        r = rsi(close)
        _, _, macd_hist = macd(close)
        last_rsi = float(r.iloc[-1]) if not r.empty and not pd.isna(r.iloc[-1]) else 50.0
        last_hist = float(macd_hist.iloc[-1]) if not macd_hist.empty and not pd.isna(macd_hist.iloc[-1]) else 0.0

        if combined >= self.buy_threshold:
            base_conf = clamp_confidence(abs(combined) * 1.15)
            if last_rsi > 72:
                base_conf *= 0.65
            if last_hist < 0:
                base_conf *= 0.85
            reason = format_reason(
                [
                    ("momentum_score", round(combined, 4)),
                    ("pos_streak", pu),
                    ("neg_streak", nu),
                    ("daily_bars", len(daily_close)),
                    ("rsi14", round(last_rsi, 2)),
                    ("macd_hist", round(last_hist, 6)),
                ]
            )
            return {"signal": "buy", "confidence": clamp_confidence(base_conf), "reason": reason}

        if combined <= self.sell_threshold:
            base_conf = clamp_confidence(abs(combined) * 1.15)
            if last_rsi < 28:
                base_conf *= 0.65
            if last_hist > 0:
                base_conf *= 0.85
            reason = format_reason(
                [
                    ("momentum_score", round(combined, 4)),
                    ("pos_streak", pu),
                    ("neg_streak", nu),
                    ("daily_bars", len(daily_close)),
                    ("rsi14", round(last_rsi, 2)),
                    ("macd_hist", round(last_hist, 6)),
                ]
            )
            return {"signal": "sell", "confidence": clamp_confidence(base_conf), "reason": reason}

        return hold_signal(
            clamp_confidence(abs(combined)),
            format_reason([("momentum_score", round(combined, 4)), ("note", "below trade thresholds")]),
        )
