import os
import pandas as pd
from datetime import datetime, timedelta
from sqlalchemy import create_engine, text
from config.settings import TARGET_SYMBOLS, DEFAULT_TIMEFRAME, BACKFILL_DAYS
from core.exchange_client import get_exchange


class DataFetcher:
    def __init__(self, db_path="data/trades.db"):
        self.exchange = get_exchange()
        self.db_path = db_path
        self.engine = create_engine(f"sqlite:///{self.db_path}", future=True)
        self._ensure_db_exists()

    def _ensure_db_exists(self):
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
        normalized_symbol = symbol.replace("-", "/")
        with self.engine.connect() as conn:
            result = conn.execute(
                text("SELECT MAX(timestamp) FROM market_data WHERE symbol = :symbol"),
                {"symbol": normalized_symbol},
            ).scalar()
        return int(result) if result is not None else None

    def fetch_ohlcv(self, symbol: str, timeframe: str = None, limit: int = 500):
        """Fetch OHLCV data from exchange incrementally from last stored timestamp."""
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
        """Save OHLCV data to database"""
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
        """Download historical data for one symbol"""
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
        """Backfill all target symbols"""
        for symbol in TARGET_SYMBOLS:
            self.backfill_symbol(symbol)

    def get_latest_data(self, symbol: str, limit: int = 500):
        """Get most recent data for a symbol"""
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