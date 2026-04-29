"""Persist backtest events to SQLite and CSV."""

from __future__ import annotations

import csv
import json
import os
import sqlite3
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from core.backtest_schema import LedgerEvent


class TradeLedger:
    """Persist simulation events and run summaries for audit and analytics.

    Responsibility:
        Provides append-only event storage in both SQLite and CSV so runs are
        replayable and inspectable with SQL or filesystem tooling.

    Key Attributes:
        db_path (str): SQLite file used as system-of-record.
        csv_dir (Path): Directory where per-run CSV event logs are written.
        _conn (sqlite3.Connection): Shared connection for all DB operations.

    Interactions:
        - Consumed by ``Backtester`` for run lifecycle and per-bar events.
        - Queried by validation/test harnesses for acceptance checks.
        - Produces artifacts suitable for downstream ETL/reporting.
    """

    EVENT_FIELDS = [
        "run_id",
        "event_id",
        "event_type",
        "bar_time",
        "symbol",
        "strategy",
        "strategy_version",
        "latency_ms",
        "payload_json",
        "error",
        "created_at",
    ]

    def __init__(self, db_path: str = "data/trades.db", csv_dir: str = "data/ledger_csv") -> None:
        """Initialize storage backends and ensure required schema exists.

        Args:
            db_path (str): SQLite database file path.
            csv_dir (str): Directory for per-run CSV event files.

        Returns:
            None: Configures filesystem and opens SQLite connection.
        """
        self.db_path = db_path
        self.csv_dir = Path(csv_dir)
        self.csv_dir.mkdir(parents=True, exist_ok=True)

        db_parent = Path(db_path).parent
        db_parent.mkdir(parents=True, exist_ok=True)

        self._conn = sqlite3.connect(self.db_path)
        self._conn.row_factory = sqlite3.Row
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        """Create ledger tables if they do not already exist.

        Returns:
            None: Executes idempotent DDL statements.
        """
        with self._conn:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS backtest_events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_id TEXT NOT NULL,
                    event_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    bar_time TEXT NOT NULL,
                    symbol TEXT,
                    strategy TEXT,
                    strategy_version TEXT,
                    latency_ms INTEGER NOT NULL DEFAULT 0,
                    payload_json TEXT NOT NULL,
                    error TEXT,
                    created_at TEXT NOT NULL DEFAULT (datetime('now')),
                    UNIQUE(run_id, event_id)
                )
                """
            )
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS backtest_runs (
                    run_id TEXT PRIMARY KEY,
                    symbol TEXT NOT NULL,
                    timeframe TEXT NOT NULL,
                    start_time TEXT NOT NULL,
                    end_time TEXT NOT NULL,
                    random_seed INTEGER NOT NULL,
                    warmup_bars INTEGER NOT NULL,
                    initial_cash REAL NOT NULL,
                    config_json TEXT NOT NULL,
                    status TEXT NOT NULL,
                    total_trades INTEGER DEFAULT 0,
                    net_pnl REAL DEFAULT 0,
                    max_drawdown REAL DEFAULT 0,
                    errors INTEGER DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
                """
            )

    @staticmethod
    def _serialize_payload(payload: dict[str, Any]) -> str:
        """Serialize payload dictionaries to compact JSON strings.

        Args:
            payload (dict[str, Any]): Event or run payload dictionary.

        Returns:
            str: JSON string with deterministic compact separators.
        """
        return json.dumps(payload, separators=(",", ":"), default=str)

    @staticmethod
    def _iso(ts: datetime) -> str:
        """Convert datetimes to UTC ISO-8601 strings.

        Args:
            ts (datetime): Input datetime.

        Returns:
            str: UTC-normalized ISO timestamp.
        """
        return ts.astimezone(UTC).replace(tzinfo=UTC).isoformat()

    def log_run_start(self, run_id: str, config: dict[str, Any]) -> None:
        """Insert or update run metadata when execution begins.

        Args:
            run_id (str): Run identifier.
            config (dict[str, Any]): Serialized run configuration payload.

        Returns:
            None: Writes a ``running`` status row in ``backtest_runs``.
        """
        now = datetime.now(tz=UTC).isoformat()
        with self._conn:
            self._conn.execute(
                """
                INSERT INTO backtest_runs (
                    run_id, symbol, timeframe, start_time, end_time, random_seed, warmup_bars,
                    initial_cash, config_json, status, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(run_id) DO UPDATE SET
                    config_json=excluded.config_json,
                    status=excluded.status,
                    updated_at=excluded.updated_at
                """,
                (
                    run_id,
                    str(config["symbol"]),
                    str(config["timeframe"]),
                    str(config["start_time"]),
                    str(config["end_time"]),
                    int(config["random_seed"]),
                    int(config["warmup_bars"]),
                    float(config["initial_cash"]),
                    self._serialize_payload(config),
                    "running",
                    now,
                ),
            )

    def log_run_end(
        self,
        run_id: str,
        *,
        total_trades: int,
        net_pnl: float,
        max_drawdown: float,
        errors: int,
        status: str = "completed",
    ) -> None:
        """Finalize run summary metrics at completion.

        Args:
            run_id (str): Run identifier.
            total_trades (int): Number of filled trade executions.
            net_pnl (float): Net profit/loss for run.
            max_drawdown (float): Max drawdown over equity curve.
            errors (int): Count of captured non-fatal errors.
            status (str): Final run status label.

        Returns:
            None: Updates terminal metrics in ``backtest_runs``.
        """
        with self._conn:
            self._conn.execute(
                """
                UPDATE backtest_runs
                SET total_trades = ?,
                    net_pnl = ?,
                    max_drawdown = ?,
                    errors = ?,
                    status = ?,
                    updated_at = datetime('now')
                WHERE run_id = ?
                """,
                (int(total_trades), float(net_pnl), float(max_drawdown), int(errors), status, run_id),
            )

    def log_event(self, event: LedgerEvent) -> None:
        """Write one normalized event record to SQL and CSV outputs.

        Args:
            event (LedgerEvent): Canonical event payload.

        Returns:
            None: Persists row in both storage sinks.
        """
        payload = {
            "run_id": event["run_id"],
            "event_id": event["event_id"],
            "event_type": event["event_type"],
            "bar_time": self._iso(event["bar_time"]),
            "symbol": event["symbol"],
            "strategy": event["strategy"],
            "strategy_version": event["strategy_version"],
            "latency_ms": int(event["latency_ms"]),
            "payload_json": event["payload_json"],
            "error": event["error"],
            "created_at": datetime.now(tz=UTC).isoformat(),
        }
        self._write_sql(payload)
        self._write_csv(payload)

    def _write_sql(self, row: dict[str, Any]) -> None:
        """Insert one event row into SQLite with idempotent semantics.

        Args:
            row (dict[str, Any]): Event row payload in ``EVENT_FIELDS`` order.

        Returns:
            None: Uses ``INSERT OR IGNORE`` against unique event key.
        """
        with self._conn:
            self._conn.execute(
                """
                INSERT OR IGNORE INTO backtest_events (
                    run_id, event_id, event_type, bar_time, symbol, strategy, strategy_version,
                    latency_ms, payload_json, error, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                tuple(row[k] for k in self.EVENT_FIELDS),
            )

    def _write_csv(self, row: dict[str, Any]) -> None:
        """Append one event row to per-run CSV.

        Args:
            row (dict[str, Any]): Event row payload.

        Returns:
            None: Creates headers on first write for each run file.
        """
        csv_path = self.csv_dir / f"{row['run_id']}.csv"
        is_new = not csv_path.exists() or os.path.getsize(csv_path) == 0
        with csv_path.open("a", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=self.EVENT_FIELDS)
            if is_new:
                writer.writeheader()
            writer.writerow({k: row[k] for k in self.EVENT_FIELDS})

    def count_events(self, run_id: str) -> dict[str, int]:
        """Aggregate event counts by type for a run.

        Args:
            run_id (str): Run identifier.

        Returns:
            dict[str, int]: Mapping of event type to row count.
        """
        rows = self._conn.execute(
            """
            SELECT event_type, COUNT(*) AS c
            FROM backtest_events
            WHERE run_id = ?
            GROUP BY event_type
            """,
            (run_id,),
        ).fetchall()
        return {str(r["event_type"]): int(r["c"]) for r in rows}

    def fetch_events(self, run_id: str, event_type: str | None = None) -> list[dict[str, Any]]:
        """Read ordered events for a run, optionally filtered by type.

        Args:
            run_id (str): Run identifier.
            event_type (str | None): Optional event type filter.

        Returns:
            list[dict[str, Any]]: Ordered event dictionaries from SQLite.
        """
        if event_type:
            rows = self._conn.execute(
                """
                SELECT run_id, event_id, event_type, bar_time, symbol, strategy, strategy_version,
                       latency_ms, payload_json, error, created_at
                FROM backtest_events
                WHERE run_id = ? AND event_type = ?
                ORDER BY id ASC
                """,
                (run_id, event_type),
            ).fetchall()
        else:
            rows = self._conn.execute(
                """
                SELECT run_id, event_id, event_type, bar_time, symbol, strategy, strategy_version,
                       latency_ms, payload_json, error, created_at
                FROM backtest_events
                WHERE run_id = ?
                ORDER BY id ASC
                """,
                (run_id,),
            ).fetchall()
        return [dict(row) for row in rows]

    def close(self) -> None:
        """Close the underlying SQLite connection.

        Returns:
            None: Releases DB resources for this ledger instance.
        """
        self._conn.close()
