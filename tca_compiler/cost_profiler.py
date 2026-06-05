"""
CostProfiler: persistent per-node cost learning via SQLite.

Tracks observed token counts and accuracy per (node_class, tier) pair.
Starts from bundled priors on cold start; improves with every execution.
Used by TierAssigner to estimate injection cost before running a node.

Design:
- SQLite WAL mode for safe concurrent writes
- Running average update (no full history stored — space efficient)
- Bundled priors shipped with TCA-Compiler for cold start
- Thread-safe via connection-per-call pattern
"""
from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .pricing import Tier, TIER_ORDER


# ── Bundled priors (from paper Appendix B) ───────────────────────────────────
# These are the cold-start values used when no observations exist yet.
# Derived from our benchmark experiments; update after running real experiments.
BUNDLED_PRIORS: list[dict] = [
    # node_class       tier       base  output inject  accuracy  n
    {"node_class": "extract",      "tier": "haiku",  "avg_input_base": 100, "avg_output": 150, "avg_inject": 280, "accuracy": 0.85, "n_observations": 0},
    {"node_class": "extract",      "tier": "sonnet", "avg_input_base": 100, "avg_output": 170, "avg_inject": 280, "accuracy": 0.93, "n_observations": 0},
    {"node_class": "extract",      "tier": "opus",   "avg_input_base": 100, "avg_output": 180, "avg_inject": 280, "accuracy": 0.97, "n_observations": 0},
    {"node_class": "sql-gen",      "tier": "haiku",  "avg_input_base": 120, "avg_output": 180, "avg_inject": 350, "accuracy": 0.72, "n_observations": 0},
    {"node_class": "sql-gen",      "tier": "sonnet", "avg_input_base": 120, "avg_output": 200, "avg_inject": 350, "accuracy": 0.88, "n_observations": 0},
    {"node_class": "sql-gen",      "tier": "opus",   "avg_input_base": 120, "avg_output": 220, "avg_inject": 350, "accuracy": 0.95, "n_observations": 0},
    {"node_class": "billing-recon","tier": "haiku",  "avg_input_base": 150, "avg_output": 200, "avg_inject": 500, "accuracy": 0.75, "n_observations": 0},
    {"node_class": "billing-recon","tier": "sonnet", "avg_input_base": 150, "avg_output": 230, "avg_inject": 500, "accuracy": 0.90, "n_observations": 0},
    {"node_class": "billing-recon","tier": "opus",   "avg_input_base": 150, "avg_output": 250, "avg_inject": 500, "accuracy": 0.96, "n_observations": 0},
    {"node_class": "policy-check", "tier": "haiku",  "avg_input_base": 160, "avg_output": 200, "avg_inject": 500, "accuracy": 0.80, "n_observations": 0},
    {"node_class": "policy-check", "tier": "sonnet", "avg_input_base": 160, "avg_output": 230, "avg_inject": 500, "accuracy": 0.92, "n_observations": 0},
    {"node_class": "policy-check", "tier": "opus",   "avg_input_base": 160, "avg_output": 250, "avg_inject": 500, "accuracy": 0.97, "n_observations": 0},
    {"node_class": "cross-recon",  "tier": "haiku",  "avg_input_base": 180, "avg_output": 250, "avg_inject": 600, "accuracy": 0.70, "n_observations": 0},
    {"node_class": "cross-recon",  "tier": "sonnet", "avg_input_base": 180, "avg_output": 280, "avg_inject": 600, "accuracy": 0.88, "n_observations": 0},
    {"node_class": "cross-recon",  "tier": "opus",   "avg_input_base": 180, "avg_output": 300, "avg_inject": 600, "accuracy": 0.95, "n_observations": 0},
    {"node_class": "iam-audit",    "tier": "haiku",  "avg_input_base": 120, "avg_output": 160, "avg_inject": 300, "accuracy": 0.82, "n_observations": 0},
    {"node_class": "iam-audit",    "tier": "sonnet", "avg_input_base": 120, "avg_output": 180, "avg_inject": 300, "accuracy": 0.92, "n_observations": 0},
    {"node_class": "iam-audit",    "tier": "opus",   "avg_input_base": 120, "avg_output": 200, "avg_inject": 300, "accuracy": 0.97, "n_observations": 0},
]


@dataclass(slots=True)
class NodeProfile:
    """Cost and accuracy profile for one (node_class, tier) pair."""
    node_class: str
    tier: Tier
    avg_input_base: float
    avg_output: float
    avg_inject: float
    accuracy: float
    n_observations: int

    def is_cold(self) -> bool:
        """True if this profile has no real observations yet."""
        return self.n_observations == 0


class CostProfiler:
    """
    Persistent cost learning database for TCA-Compiler.

    On cold start: uses bundled priors.
    After each execution: updates running averages.
    Survives across runs via SQLite persistence.
    """

    def __init__(self, db_path: str | Path = "~/.tca/cost_profiles.db") -> None:
        self.db_path = Path(db_path).expanduser()
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.execute("PRAGMA journal_mode=WAL")  # safe concurrent writes
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        """Create schema and seed bundled priors if table is empty."""
        with self._connect() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS node_profiles (
                    node_class      TEXT NOT NULL,
                    tier            TEXT NOT NULL,
                    avg_input_base  REAL DEFAULT 150,
                    avg_output      REAL DEFAULT 200,
                    avg_inject      REAL DEFAULT 400,
                    accuracy        REAL DEFAULT 0.80,
                    n_observations  INTEGER DEFAULT 0,
                    last_updated    TEXT,
                    PRIMARY KEY (node_class, tier)
                )
            """)
            conn.commit()

            # Seed priors only if table is empty
            count = conn.execute(
                "SELECT COUNT(*) FROM node_profiles"
            ).fetchone()[0]

            if count == 0:
                from datetime import datetime
                now = datetime.utcnow().isoformat()
                conn.executemany("""
                    INSERT OR IGNORE INTO node_profiles
                        (node_class, tier, avg_input_base, avg_output,
                         avg_inject, accuracy, n_observations, last_updated)
                    VALUES
                        (:node_class, :tier, :avg_input_base, :avg_output,
                         :avg_inject, :accuracy, :n_observations, :last_updated)
                """, [{**p, "last_updated": now} for p in BUNDLED_PRIORS],
                )
                conn.commit()

    def get(self, node_class: str, tier: Tier) -> NodeProfile:
        """
        Get profile for (node_class, tier).
        Returns a generic prior if not found.
        """
        with self._connect() as conn:
            row = conn.execute("""
                SELECT * FROM node_profiles
                WHERE node_class = ? AND tier = ?
            """, (node_class, tier)).fetchone()

        if row is None:
            # Unknown node class — return conservative prior
            return NodeProfile(
                node_class=node_class,
                tier=tier,
                avg_input_base=150,
                avg_output=200,
                avg_inject=400,
                accuracy=0.80,
                n_observations=0,
            )

        return NodeProfile(
            node_class=row["node_class"],
            tier=row["tier"],
            avg_input_base=row["avg_input_base"],
            avg_output=row["avg_output"],
            avg_inject=row["avg_inject"],
            accuracy=row["accuracy"],
            n_observations=row["n_observations"],
        )

    def update(
        self,
        node_class: str,
        tier: Tier,
        observed_input_base: int,
        observed_output: int,
        observed_inject: int,
        observed_accuracy: float,
    ) -> None:
        """
        Update profile with one new observation using running average.

        Running average formula:
            new_avg = (old_avg * n + new_value) / (n + 1)

        This is called by the execution callback after each node completes.
        """
        from datetime import datetime

        with self._connect() as conn:
            row = conn.execute("""
                SELECT avg_input_base, avg_output, avg_inject,
                       accuracy, n_observations
                FROM node_profiles
                WHERE node_class = ? AND tier = ?
            """, (node_class, tier)).fetchone()

            if row is None:
                # First observation for this node class
                conn.execute("""
                    INSERT INTO node_profiles
                        (node_class, tier, avg_input_base, avg_output,
                         avg_inject, accuracy, n_observations, last_updated)
                    VALUES (?, ?, ?, ?, ?, ?, 1, ?)
                """, (node_class, tier,
                      observed_input_base, observed_output,
                      observed_inject, observed_accuracy,
                      datetime.utcnow().isoformat()))
            else:
                n = row["n_observations"]
                new_base   = (row["avg_input_base"] * n + observed_input_base) / (n + 1)
                new_output = (row["avg_output"] * n + observed_output) / (n + 1)
                new_inject = (row["avg_inject"] * n + observed_inject) / (n + 1)
                new_acc    = (row["accuracy"] * n + observed_accuracy) / (n + 1)

                conn.execute("""
                    UPDATE node_profiles
                    SET avg_input_base = ?,
                        avg_output     = ?,
                        avg_inject     = ?,
                        accuracy       = ?,
                        n_observations = n_observations + 1,
                        last_updated   = ?
                    WHERE node_class = ? AND tier = ?
                """, (new_base, new_output, new_inject, new_acc,
                      datetime.utcnow().isoformat(),
                      node_class, tier))
            conn.commit()

    def all_profiles(self) -> list[NodeProfile]:
        """Return all profiles — used by TierAssigner."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM node_profiles ORDER BY node_class, tier"
            ).fetchall()
        return [
            NodeProfile(
                node_class=r["node_class"],
                tier=r["tier"],
                avg_input_base=r["avg_input_base"],
                avg_output=r["avg_output"],
                avg_inject=r["avg_inject"],
                accuracy=r["accuracy"],
                n_observations=r["n_observations"],
            )
            for r in rows
        ]

    def reset(self) -> None:
        """
        Delete all observations and re-seed bundled priors.
        Use only when starting a fresh experiment series.
        """
        if self.db_path.exists():
            self.db_path.unlink()
        self._init_db()

    def summary(self) -> dict:
        """Summary stats for logging."""
        profiles = self.all_profiles()
        cold = sum(1 for p in profiles if p.is_cold())
        warm = len(profiles) - cold
        return {
            "total_profiles": len(profiles),
            "cold_start": cold,
            "warm": warm,
            "db_path": str(self.db_path),
        }
