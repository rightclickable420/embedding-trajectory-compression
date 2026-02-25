#!/usr/bin/env python3
"""
Field Tracker — Automatic telemetry for holographic reconsolidation.

Every reconsolidation snapshots the full field state:
- Per-memory similarity (standard vs promoted)
- Access energy vector
- Promotion deltas
- Frequency band classification

Query the history to see how any memory drifts over time.
No manual logging required — just data.
"""

import json
import sqlite3
import time
import numpy as np
from scipy.fft import dct, idct


class FieldTracker:
    """Tracks holographic field evolution over time."""

    def __init__(self, db_path: str = "memory.db"):
        self.db_path = db_path
        self._init_db()

    def _init_db(self):
        db = sqlite3.connect(self.db_path)
        db.execute("""
            CREATE TABLE IF NOT EXISTS field_snapshots (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL NOT NULL,
                n_messages INTEGER NOT NULL,
                k_coefficients INTEGER NOT NULL,
                keep_ratio REAL NOT NULL,
                promotion_strength REAL NOT NULL,
                avg_sim_standard REAL NOT NULL,
                avg_sim_promoted REAL NOT NULL,
                avg_delta REAL NOT NULL,
                total_accesses INTEGER NOT NULL,
                memories_with_energy INTEGER NOT NULL,
                sim_standard BLOB,
                sim_promoted BLOB,
                access_energy BLOB,
                deltas BLOB
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS field_trajectories (
                snapshot_id INTEGER NOT NULL,
                msg_id INTEGER NOT NULL,
                sim_standard REAL NOT NULL,
                sim_promoted REAL NOT NULL,
                delta REAL NOT NULL,
                access_energy REAL NOT NULL,
                freq_band TEXT,
                PRIMARY KEY (snapshot_id, msg_id),
                FOREIGN KEY (snapshot_id) REFERENCES field_snapshots(id)
            )
        """)
        db.commit()
        db.close()

    def snapshot(self, recon_result: dict,
                 sim_standard: np.ndarray, sim_promoted: np.ndarray,
                 access_energy: np.ndarray, msg_ids: list[int],
                 promotion_strength: float = 2.0) -> int:
        """Record a full field snapshot after reconsolidation."""
        db = sqlite3.connect(self.db_path)
        now = time.time()
        deltas = sim_promoted - sim_standard

        cursor = db.execute(
            "INSERT INTO field_snapshots "
            "(timestamp, n_messages, k_coefficients, keep_ratio, promotion_strength, "
            "avg_sim_standard, avg_sim_promoted, avg_delta, total_accesses, "
            "memories_with_energy, sim_standard, sim_promoted, access_energy, deltas) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (now, len(msg_ids),
             recon_result.get('k_coefficients', 0),
             recon_result.get('keep_ratio', 0.1),
             promotion_strength,
             float(np.mean(sim_standard)),
             float(np.mean(sim_promoted)),
             float(np.mean(deltas)),
             int(np.sum(access_energy > 0)),
             int(np.sum(access_energy > 0)),
             sim_standard.astype(np.float32).tobytes(),
             sim_promoted.astype(np.float32).tobytes(),
             access_energy.astype(np.float32).tobytes(),
             deltas.astype(np.float32).tobytes())
        )
        snapshot_id = cursor.lastrowid

        for i, msg_id in enumerate(msg_ids):
            db.execute(
                "INSERT OR REPLACE INTO field_trajectories "
                "(snapshot_id, msg_id, sim_standard, sim_promoted, delta, access_energy, freq_band) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (snapshot_id, msg_id,
                 float(sim_standard[i]), float(sim_promoted[i]),
                 float(deltas[i]), float(access_energy[i]), 'unknown')
            )

        db.commit()
        db.close()
        return snapshot_id

    def memory_drift(self, msg_id: int) -> list[dict]:
        """How has a specific memory moved across all snapshots?"""
        db = sqlite3.connect(self.db_path)
        rows = db.execute(
            "SELECT s.timestamp, t.sim_standard, t.sim_promoted, t.delta, "
            "t.access_energy, t.freq_band "
            "FROM field_trajectories t JOIN field_snapshots s ON t.snapshot_id = s.id "
            "WHERE t.msg_id = ? ORDER BY s.timestamp",
            (msg_id,)
        ).fetchall()
        db.close()

        return [
            {"timestamp": r[0], "sim_standard": r[1], "sim_promoted": r[2],
             "delta": r[3], "access_energy": r[4], "freq_band": r[5]}
            for r in rows
        ]

    def field_health(self) -> dict:
        """Overall field status."""
        db = sqlite3.connect(self.db_path)
        n_snapshots = db.execute("SELECT COUNT(*) FROM field_snapshots").fetchone()[0]
        trend = []
        if n_snapshots > 0:
            for r in db.execute(
                "SELECT timestamp, avg_sim_standard, avg_sim_promoted, avg_delta, "
                "memories_with_energy FROM field_snapshots ORDER BY timestamp"
            ).fetchall():
                trend.append({
                    "timestamp": r[0], "standard": round(r[1], 4),
                    "promoted": round(r[2], 4), "delta": round(r[3], 4),
                    "energized": r[4]
                })
        db.close()
        return {"snapshots": n_snapshots, "trend": trend}

    def summary(self) -> str:
        """Human-readable field status."""
        h = self.field_health()
        lines = [f"Field Status: {h['snapshots']} snapshots"]
        if h['trend']:
            latest = h['trend'][-1]
            lines.append(f"  Latest: standard={latest['standard']}, "
                         f"promoted={latest['promoted']}, Δ={latest['delta']:+.4f}")
        return "\n".join(lines)


if __name__ == "__main__":
    tracker = FieldTracker()
    print(tracker.summary())
