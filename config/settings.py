"""Runtime configuration for trading, data, and execution defaults.

This module centralizes environment-driven settings consumed by the rest of the
bot. It is intentionally importable from any layer (strategies, core services,
API wrappers) so new components can share one source of truth for defaults.

Notes:
    Values are loaded at import time via ``python-dotenv``.
"""

import os
from dotenv import load_dotenv

load_dotenv()

PAPER_MODE = os.getenv("PAPER_MODE", "true").lower() == "true"
EXCHANGE = os.getenv("EXCHANGE", "coinbase")
API_KEY = os.getenv("COINBASE_API_KEY")
API_SECRET = os.getenv("COINBASE_API_SECRET")

TARGET_SYMBOLS = [
    "BTC/USD",
    "ETH/USD",
    "SOL/USD",
]

DEFAULT_TIMEFRAME = "1h"
BACKFILL_DAYS = 365