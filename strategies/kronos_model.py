"""
Optional Kronos (shiyu-coder/Kronos) integration.

Set env KRONOS_REPO_PATH to the root of a cloned Kronos repo so `model` is importable,
or install/configure the package in your environment.

If unavailable, `get_predictor()` returns None and strategies should hold.
"""

from __future__ import annotations

import logging
import os
import sys
from dataclasses import dataclass
from typing import Any, Callable, Optional

import pandas as pd

logger = logging.getLogger(__name__)

_predictor_singleton: Any = None
_load_error: str | None = None


def _extend_sys_path() -> None:
    repo = os.getenv("KRONOS_REPO_PATH", "").strip()
    if repo and repo not in sys.path:
        sys.path.insert(0, repo)


def _try_build_predictor() -> tuple[Any | None, str | None]:
    _extend_sys_path()
    try:
        from model import Kronos, KronosPredictor, KronosTokenizer  # type: ignore
    except Exception as e:  # pragma: no cover - import path varies
        return None, f"Kronos import failed: {e!s}"

    tokenizer_id = os.getenv("KRONOS_TOKENIZER_ID", "NeoQuasar/Kronos-Tokenizer-base")
    model_id = os.getenv("KRONOS_MODEL_ID", "NeoQuasar/Kronos-mini")
    device = os.getenv("KRONOS_DEVICE", "cpu")
    max_context = int(os.getenv("KRONOS_MAX_CONTEXT", "512"))

    try:
        tokenizer = KronosTokenizer.from_pretrained(tokenizer_id)
        model = Kronos.from_pretrained(model_id)
        predictor = KronosPredictor(model, tokenizer, device=device, max_context=max_context)
    except Exception as e:  # pragma: no cover - heavy deps / no weights
        return None, f"Kronos load failed: {e!s}"

    return predictor, None


def get_predictor(force_reload: bool = False) -> Any | None:
    """Return a shared KronosPredictor instance, or None if not available."""
    global _predictor_singleton, _load_error
    if force_reload:
        _predictor_singleton = None
        _load_error = None
    if _predictor_singleton is not None:
        return _predictor_singleton
    if _load_error is not None:
        return None

    pred, err = _try_build_predictor()
    if err:
        _load_error = err
        logger.info("Kronos disabled: %s", err)
        return None
    _predictor_singleton = pred
    return _predictor_singleton


def get_load_error() -> str | None:
    return _load_error


def set_predictor_for_tests(predictor: Any | None) -> None:
    """Override singleton (used by unit tests)."""
    global _predictor_singleton, _load_error
    _predictor_singleton = predictor
    _load_error = None


def ohlcv_to_kronos_frames(
    df: pd.DataFrame,
    lookback: int,
    pred_len: int,
    use_volume: bool = False,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Build (x_df, x_timestamp, y_timestamp) for KronosPredictor.predict.

    Expects optional `timestamp` column; otherwise uses a synthetic RangeIndex-based timeline.
    """
    if len(df) < lookback + pred_len:
        raise ValueError(f"Need at least {lookback + pred_len} rows, got {len(df)}")

    tail = df.iloc[-(lookback + pred_len) :].copy()
    if "timestamp" in tail.columns:
        ts = pd.to_datetime(tail["timestamp"])
    else:
        # Synthetic uniform spacing for tests / minimal OHLCV frames
        ts = pd.date_range("2020-01-01", periods=len(tail), freq="h")

    tail["_ts"] = ts
    x_part = tail.iloc[:lookback]
    y_part = tail.iloc[lookback : lookback + pred_len]

    if use_volume and "volume" in tail.columns:
        cols = ["open", "high", "low", "close", "volume"]
    else:
        cols = ["open", "high", "low", "close"]

    x_df = x_part[cols].astype(float)
    x_timestamp = pd.Series(x_part["_ts"].values)
    y_timestamp = pd.Series(y_part["_ts"].values)
    return x_df, x_timestamp, y_timestamp


def predict_future_closes(
    df: pd.DataFrame,
    *,
    lookback: int = 128,
    pred_len: int = 8,
    sample_count: int = 3,
    temperature: float = 1.0,
    top_p: float = 0.9,
    use_volume: bool = False,
    predict_fn: Optional[Callable[..., pd.DataFrame]] = None,
) -> tuple[list[float], str | None]:
    """
    Return list of predicted closes (one per sample path if model aggregates — we take last bar mean).

    Uses `predict_fn` when injected (tests); otherwise uses real predictor if loaded.
    """
    predictor = predict_fn or get_predictor()
    if predictor is None:
        return [], get_load_error() or "Kronos predictor not available"

    x_df, x_timestamp, y_timestamp = ohlcv_to_kronos_frames(
        df, lookback=lookback, pred_len=pred_len, use_volume=use_volume
    )

    pred_df: pd.DataFrame = predictor.predict(
        df=x_df,
        x_timestamp=x_timestamp,
        y_timestamp=y_timestamp,
        pred_len=pred_len,
        T=temperature,
        top_p=top_p,
        sample_count=sample_count,
        verbose=False,
    )

    if pred_df is None or pred_df.empty or "close" not in pred_df.columns:
        return [], "Kronos returned empty prediction"

    # Use final forecasted close per returned row (last row is horizon end)
    last_close = float(pred_df["close"].iloc[-1])
    return [last_close], None


@dataclass
class MockKronosPredictor:
    """Deterministic stand-in for unit tests."""

    bias: float = 0.01

    def predict(self, **kwargs: Any) -> pd.DataFrame:
        df: pd.DataFrame = kwargs["df"]
        pred_len: int = int(kwargs.get("pred_len", 5))
        last = float(df["close"].iloc[-1])
        close = last * (1.0 + self.bias)
        return pd.DataFrame({"close": [close] * pred_len})
