import os
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from config.settings import TARGET_SYMBOLS, DEFAULT_TIMEFRAME, BACKFILL_DAYS
from core.exchange_client import get_exchange


class DataFetcher:
    """Ingest and persist OHLCV market data for downstream trading components.

    Responsibility:
        Owns historical and incremental market-data synchronization between the
        exchange adapter and local SQLite storage.

    Key Attributes:
        exchange: CCXT exchange client returned by ``get_exchange``.
        db_path: Filesystem path for SQLite persistence.
        engine: SQLAlchemy engine used for reads/writes.

    Interactions:
        - Called by runners or schedulers to keep candles current.
        - Supplies data consumed by strategies and backtesting flows.
        - Shares schema with other modules reading ``market_data``.
    """

    def __init__(self, db_path="data/trades.db"):
        """Initialize exchange client and local market-data storage.

        Args:
            db_path (str): SQLite file path used for candle persistence.

        Returns:
            None: Initializes internal state in-place.
        """
        self.exchange = get_exchange()
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{self.db_path}", future=True)
        self._ensure_db_exists()

    def _ensure_db_exists(self):
        """Create and migrate market data table if needed.

        Returns:
            None: Applies schema and lightweight timestamp repair migration.

        Notes:
            Includes a repair query for legacy rows where timestamps were stored
            with incorrect units.
        """
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        with self.engine.begin() as conn:
            conn.execute(
                text(
                    """
                    CREATE TABLE IF NOT EXISTS market_data (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp INTEGER NOT NULL,
                        symbol TEXT NOT NULL,
                        open REAL NOT NULL,
                        high REAL NOT NULL,
                        low REAL NOT NULL,
                        close REAL NOT NULL,
                        volume REAL NOT NULL,
                        inserted_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        UNIQUE(timestamp, symbol)
                    )
                    """
                )
            )

            # Repair legacy bug where milliseconds were accidentally divided by 1e6.
            conn.execute(
                text(
                    "UPDATE market_data SET timestamp = CAST(timestamp * 1000000 AS INTEGER) "
                    "WHERE timestamp < 10000000000"
                )
            )

    def _get_last_timestamp(self, symbol: str):
        """Return the latest persisted candle timestamp for a symbol.

        Args:
            symbol (str): Exchange symbol, slash or dash separated.

        Returns:
            int | None: Millisecond epoch timestamp, or ``None`` if no rows.
        """
        normalized_symbol = symbol.replace("-", "/")
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT MAX(timestamp) FROM market_data WHERE symbol = :symbol"),
                {"symbol": normalized_symbol},
            ).scalar()
        return int(result) if result is not None else None

    def fetch_ohlcv(self, symbol: str, timeframe: str = None, limit: int = 500):
        """Fetch candles incrementally from exchange.

        Args:
            symbol (str): Symbol to fetch from the exchange.
            timeframe (str | None): Candle timeframe (defaults to project config).
            limit (int): Batch size per exchange request.

        Returns:
            pd.DataFrame: Normalized OHLCV frame with ``timestamp`` as datetime.

        Notes:
            If prior rows exist, only unseen candles are returned.
        """
        if timeframe is None:
            timeframe = DEFAULT_TIMEFRAME

        normalized_symbol = symbol.replace("-", "/")
        last_timestamp = self._get_last_timestamp(normalized_symbol)
        now_ms = int(datetime.now().timestamp() * 1000)

        if last_timestamp is None:
            ohlcv = self.exchange.fetch_ohlcv(symbol, timeframe, limit=limit)
        else:
            ohlcv = []
            since = last_timestamp

            while since < now_ms:
                batch = self.exchange.fetch_ohlcv(symbol, timeframe, since=since, limit=limit)
                if not batch:
                    break

                ohlcv.extend(batch)
                newest = batch[-1][0]

                if newest <= since:
                    break

                since = newest + 1

                if len(batch) < limit:
                    break

            # Keep only unseen candles; skip last known timestamp from DB.
            ohlcv = [row for row in ohlcv if row[0] > last_timestamp]

        df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
        if df.empty:
            return df

        df['symbol'] = normalized_symbol
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True).dt.tz_localize(None)

        return df

    def save_to_db(self, df: pd.DataFrame):
        """Insert OHLCV candles into SQLite with de-duplication.

        Args:
            df (pd.DataFrame): Input candles matching expected OHLCV schema.

        Returns:
            int: Number of newly inserted rows.
        """
        if df.empty:
            return 0

        to_insert = df.copy()
        to_insert['timestamp'] = (
            to_insert['timestamp'].astype('datetime64[ns]').astype('int64') // 10**6
        )

        records = to_insert[['timestamp', 'symbol', 'open', 'high', 'low', 'close', 'volume']].to_dict("records")
        if not records:
            return 0

        with self.engine.begin() as conn:
            result = conn.execute(
                text(
                    """
                    INSERT OR IGNORE INTO market_data (
                        timestamp, symbol, open, high, low, close, volume
                    ) VALUES (
                        :timestamp, :symbol, :open, :high, :low, :close, :volume
                    )
                    """
                ),
                records,
            )

        return result.rowcount or 0

    def backfill_symbol(self, symbol: str, days: int = None):
        """Backfill one symbol for a trailing day window.

        Args:
            symbol (str): Symbol to backfill.
            days (int | None): Number of trailing days; falls back to config.

        Returns:
            None: Prints progress and writes rows when available.
        """
        if days is None:
            days = BACKFILL_DAYS

        print(f"Backfilling {symbol} for last {days} days...")

        since = int((datetime.now() - timedelta(days=days)).timestamp() * 1000)
        df = self.fetch_ohlcv(symbol, limit=1000)
        if not df.empty and since:
            df = df[df["timestamp"] >= pd.to_datetime(since, unit="ms")]

        if not df.empty:
            saved = self.save_to_db(df)
            print(f"Saved {saved} new candles for {symbol}")

        else:
            print(f"No data recieved for {symbol}")

    def backfill_all(self):
        """Run backfill for all configured target symbols.

        Returns:
            None: Iterates configured symbols and delegates to ``backfill_symbol``.
        """
        for symbol in TARGET_SYMBOLS:
            self.backfill_symbol(symbol)

    def get_latest_data(self, symbol: str, limit: int = 500):
        """Read latest candles for a symbol from local storage.

        Args:
            symbol (str): Symbol to query.
            limit (int): Maximum rows to return.

        Returns:
            pd.DataFrame: Descending timestamp-ordered rows from SQLite.
        """
        normalized_symbol = symbol.replace("-", "/")
        query = text(
            """
            SELECT * FROM market_data
            WHERE symbol = :symbol
            ORDER BY timestamp DESC
            LIMIT :limit
            """
        )
        df = pd.read_sql_query(query, self.engine, params={"symbol": normalized_symbol, "limit": limit})

        if not df.empty:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')

        return df