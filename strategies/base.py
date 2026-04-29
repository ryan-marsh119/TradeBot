"""Abstract strategy base class."""

from __future__ import annotations

from abc import ABC, abstractmethod

import pandas as pd

from strategies.schema import SignalDict
from strategies.utils import hold_signal, validate_ohlcv_df


class Strategy(ABC):
    """Abstract contract for all signal-generation strategies.

    Responsibility:
        Defines the common API every strategy must implement so the ensemble,
        backtester, and future live execution pipelines can invoke strategies
        interchangeably.

    Key Attributes:
        Subclasses may define tuning attributes (thresholds, lookbacks, model
        references), but must return ``SignalDict`` outputs.

    Interactions:
        - Called by ``Backtester`` and ensemble helpers.
        - Uses shared validation helpers from ``strategies.utils``.
    """

    @abstractmethod
    def generate_signal(self, df: pd.DataFrame) -> SignalDict:
        """Generate a trading signal for the latest bar context.

        Args:
            df (pd.DataFrame): OHLCV context frame ordered by time.

        Returns:
            SignalDict: Canonical strategy output with signal/confidence/reason.
        """
        ...

    def _validate(self, df: pd.DataFrame, min_rows: int = 1) -> str | None:
        """Validate strategy input frame shape and required fields.

        Args:
            df (pd.DataFrame): Candidate OHLCV frame.
            min_rows (int): Minimum row count required for strategy logic.

        Returns:
            str | None: Error string when invalid, else ``None``.
        """
        return validate_ohlcv_df(df, min_rows=min_rows)

    def safe_generate(self, df: pd.DataFrame, min_rows: int = 1) -> SignalDict:
        """Generate a signal with guardrails against invalid inputs.

        Args:
            df (pd.DataFrame): Candidate OHLCV frame.
            min_rows (int): Minimum rows required before generation.

        Returns:
            SignalDict: Hold signal on validation failure, else strategy output.

        Notes:
            This helper intentionally avoids raising for common input failures so
            orchestration layers can continue processing.
        """
        err = self._validate(df, min_rows=min_rows)
        if err:
            return hold_signal(0.0, err)
        return self.generate_signal(df)
