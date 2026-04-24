"""Forecast-based strategy using optional Kronos model."""

from __future__ import annotations

from typing import Any, Callable, Optional

import pandas as pd

from strategies.base import Strategy
from strategies import kronos_model
from strategies.schema import SignalDict
from strategies.utils import clamp_confidence, format_reason, hold_signal, validate_ohlcv_df


class KronosStrategy(Strategy):
    def __init__(
        self,
        lookback: int = 96,
        pred_len: int = 8,
        sample_count: int = 3,
        move_threshold_pct: float = 0.0025,
        predict_override: Optional[Callable[..., pd.DataFrame]] = None,
    ):
        self.lookback = lookback
        self.pred_len = pred_len
        self.sample_count = sample_count
        self.move_threshold_pct = move_threshold_pct
        self.predict_override = predict_override

    def generate_signal(self, df: pd.DataFrame) -> SignalDict:
        need = self.lookback + self.pred_len + 2
        err = validate_ohlcv_df(df, min_rows=need)
        if err:
            return hold_signal(0.0, err)

        last_close = float(df["close"].iloc[-1])

        def _call_predictor(**kwargs: Any) -> pd.DataFrame:
            if self.predict_override is not None:
                return self.predict_override(**kwargs)
            predictor = kronos_model.get_predictor()
            if predictor is None:
                raise RuntimeError(kronos_model.get_load_error() or "predictor unavailable")
            return predictor.predict(**kwargs)

        try:
            x_df, x_timestamp, y_timestamp = kronos_model.ohlcv_to_kronos_frames(
                df,
                lookback=self.lookback,
                pred_len=self.pred_len,
                use_volume=False,
            )
            pred_df = _call_predictor(
                df=x_df,
                x_timestamp=x_timestamp,
                y_timestamp=y_timestamp,
                pred_len=self.pred_len,
                T=1.0,
                top_p=0.9,
                sample_count=self.sample_count,
                verbose=False,
            )
        except Exception as e:  # pragma: no cover - optional heavy stack
            return hold_signal(0.0, f"Kronos prediction failed: {e!s}")

        if pred_df is None or pred_df.empty or "close" not in pred_df.columns:
            return hold_signal(0.0, "Kronos returned empty forecast")

        fc = pred_df["close"].astype(float)
        pred_last = float(fc.iloc[-1])
        pred_mean = float(fc.mean())
        dispersion = float(fc.std(ddof=1)) if len(fc) > 1 else 0.0

        move_pct = (pred_last - last_close) / last_close if last_close else 0.0
        # Confidence rises with expected move; falls with path dispersion
        move_score = min(1.0, abs(move_pct) / max(self.move_threshold_pct, 1e-9))
        vol_penalty = 1.0 / (1.0 + dispersion / max(last_close * 0.001, 1e-9))
        confidence = clamp_confidence(move_score * vol_penalty)

        if move_pct > self.move_threshold_pct:
            reason = format_reason(
                [
                    ("pred_last", round(pred_last, 6)),
                    ("pred_mean", round(pred_mean, 6)),
                    ("last", round(last_close, 6)),
                    ("move_pct", round(move_pct * 100, 4)),
                    ("dispersion", round(dispersion, 6)),
                ]
            )
            return {"signal": "buy", "confidence": confidence, "reason": reason}

        if move_pct < -self.move_threshold_pct:
            reason = format_reason(
                [
                    ("pred_last", round(pred_last, 6)),
                    ("pred_mean", round(pred_mean, 6)),
                    ("last", round(last_close, 6)),
                    ("move_pct", round(move_pct * 100, 4)),
                    ("dispersion", round(dispersion, 6)),
                ]
            )
            return {"signal": "sell", "confidence": confidence, "reason": reason}

        return hold_signal(
            confidence * 0.5,
            format_reason(
                [
                    ("move_pct", round(move_pct * 100, 4)),
                    ("note", "below Kronos threshold"),
                ]
            ),
        )
