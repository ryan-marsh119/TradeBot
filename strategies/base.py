"""Abstract strategy base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from strategies.schema import SignalDict
from strategies.utils import hold_signal, validate_ohlcv_df


class Strategy(ABC):
    """All strategies consume OHLCV DataFrames and return a standard signal dict."""

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> SignalDict:
        """Return {signal, confidence, reason} for the latest bar context."""
        ...

    def _validate(self, df: pd.DataFrame, min_rows: int = 1) -> str | None:
        """Return error message if invalid, else None."""
        return validate_ohlcv_df(df, min_rows=min_rows)

    def safe_generate(self, df: pd.DataFrame, min_rows: int = 1) -> SignalDict:
        """Like generate_signal but never raises on bad input."""
        err = self._validate(df, min_rows=min_rows)
        if err:
            return hold_signal(0.0, err)
        return self.generate_signal(df)
