#!/usr/bin/env python3
"""
Access-Driven Reconsolidation Engine

Tracks memory access patterns and uses them to weight the DCT,
physically promoting frequently-accessed memories toward low-frequency
bands where they survive compression.

The mechanism:
  1. Track every query against the memory store
  2. Build access energy vector (how often each memory is recalled)
  3. Amplify accessed embeddings BEFORE the DCT
  4. Their energy shifts to low-frequency coefficients → survives truncation
  5. Divide back out after reconstruction → original scale, new frequency profile

This changes the *representation*, not just the ranking — distinct from LRU
caches or recency-boosted retrieval. The compressed memory becomes a
physically different object after reconsolidation.

Structural parallel to biological memory reconsolidation (Nader et al., 2000):
recalled memories become labile and are re-encoded, potentially strengthening.
"""

import json
import sqlite3
import time
import numpy as np
from scipy.fft import dct, idct
from typing import Optional


class ReconsolidationEngine:
    """Tracks access patterns and recomputes holographic fields with promotion."""

    def __init__(self, db_path: str = "memory.db", model_name: str = "BAAI/bge-small-en-v1.5"):
        self.db_path = db_path
        self.model_name = model_name
        self._model = None
        self._init_db()

    def _init_db(self):
        db = sqlite3.connect(self.db_path)
        db.execute("""
            CREATE TABLE IF NOT EXISTS messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                content TEXT NOT NULL,
                created_at REAL DEFAULT (unixepoch())
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                msg_id INTEGER PRIMARY KEY,
                embedding BLOB NOT NULL,
                FOREIGN KEY (msg_id) REFERENCES messages(id)
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS access_log (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_text TEXT NOT NULL,
                query_embedding BLOB,
                timestamp REAL NOT NULL,
                top_k_ids TEXT,
                top_k_sims TEXT
            )
        """)
        db.execute("""
            CREATE TABLE IF NOT EXISTS access_energy (
                msg_id INTEGER PRIMARY KEY,
                total_accesses INTEGER DEFAULT 0,
                total_similarity REAL DEFAULT 0.0,
                last_accessed REAL,
                decayed_energy REAL DEFAULT 0.0
            )
        """)
        db.commit()
        db.close()

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def _embed(self, text: str) -> np.ndarray:
        return self.model.encode([text])[0].astype(np.float32)

    def ingest(self, sections: list[str]):
        """Load text sections into the database with embeddings."""
        db = sqlite3.connect(self.db_path)
        embeddings = self.model.encode(sections, show_progress_bar=True)
        for i, (text, emb) in enumerate(zip(sections, embeddings)):
            cursor = db.execute("INSERT INTO messages (content) VALUES (?)", (text,))
            msg_id = cursor.lastrowid
            db.execute("INSERT INTO embeddings (msg_id, embedding) VALUES (?, ?)",
                       (msg_id, emb.astype(np.float32).tobytes()))
        db.commit()
        db.close()
        print(f"Ingested {len(sections)} sections")

    def access(self, query: str, top_k: int = 20) -> dict:
        """Query memory and record access pattern."""
        query_emb = self._embed(query)
        db = sqlite3.connect(self.db_path)

        rows = db.execute(
            "SELECT e.msg_id, e.embedding, m.content "
            "FROM embeddings e JOIN messages m ON e.msg_id = m.id "
            "ORDER BY e.msg_id"
        ).fetchall()

        if not rows:
            db.close()
            return {"matches": [], "activated": 0}

        results = []
        for msg_id, emb_blob, content in rows:
            emb = np.frombuffer(emb_blob, dtype=np.float32)
            norm_q, norm_e = np.linalg.norm(query_emb), np.linalg.norm(emb)
            sim = float(np.dot(query_emb, emb) / (norm_q * norm_e + 1e-10))
            results.append((msg_id, sim, content))

        results.sort(key=lambda x: x[1], reverse=True)
        top = results[:top_k]

        now = time.time()
        top_ids = json.dumps([r[0] for r in top])
        top_sims = json.dumps([round(r[1], 4) for r in top])

        db.execute(
            "INSERT INTO access_log (query_text, query_embedding, timestamp, top_k_ids, top_k_sims) "
            "VALUES (?, ?, ?, ?, ?)",
            (query, query_emb.tobytes(), now, top_ids, top_sims)
        )

        for msg_id, sim, _ in top:
            if sim < 0.3:
                continue
            existing = db.execute(
                "SELECT total_accesses FROM access_energy WHERE msg_id = ?", (msg_id,)
            ).fetchone()
            if existing:
                db.execute(
                    "UPDATE access_energy SET total_accesses = total_accesses + 1, "
                    "total_similarity = total_similarity + ?, last_accessed = ? WHERE msg_id = ?",
                    (sim, now, msg_id))
            else:
                db.execute(
                    "INSERT INTO access_energy (msg_id, total_accesses, total_similarity, last_accessed) "
                    "VALUES (?, 1, ?, ?)", (msg_id, sim, now))

        db.commit()
        db.close()

        return {
            "matches": [(mid, sim, content[:100]) for mid, sim, content in top[:10]],
            "activated": sum(1 for _, s, _ in top if s >= 0.3),
        }

    def compute_access_energy(self, half_life_hours: float = 168.0) -> np.ndarray:
        """Compute access energy vector with exponential decay."""
        db = sqlite3.connect(self.db_path)
        now = time.time()
        decay_rate = np.log(2) / (half_life_hours * 3600)

        msg_ids = [r[0] for r in db.execute("SELECT id FROM messages ORDER BY id").fetchall()]
        N = len(msg_ids)
        energy = np.zeros(N)
        id_to_idx = {mid: i for i, mid in enumerate(msg_ids)}

        for ids_json, sims_json, ts in db.execute(
            "SELECT top_k_ids, top_k_sims, timestamp FROM access_log"
        ).fetchall():
            ids = json.loads(ids_json)
            sims = json.loads(sims_json)
            decay = np.exp(-decay_rate * (now - ts))
            for mid, sim in zip(ids, sims):
                if sim >= 0.3 and mid in id_to_idx:
                    energy[id_to_idx[mid]] += sim * decay

        db.close()
        if energy.max() > 0:
            energy = energy / energy.max()
        return energy

    def reconsolidate(self, keep_ratio: float = 0.1, promotion_strength: float = 2.0) -> dict:
        """
        Recompute the holographic field with access-driven promotion.

        Standard:  R = IDCT(DCT(E)[:k])
        Promoted:  E_w = E * (1 + γ·α)
                   R = IDCT(DCT(E_w)[:k]) / (1 + γ·α)

        Accessed memories are amplified before DCT → more energy in low freq
        → survives truncation → divide back out → same scale, new freq profile.
        """
        db = sqlite3.connect(self.db_path)
        rows = db.execute(
            "SELECT e.msg_id, e.embedding, m.content "
            "FROM embeddings e JOIN messages m ON e.msg_id = m.id "
            "ORDER BY e.msg_id"
        ).fetchall()
        db.close()

        N = len(rows)
        if N == 0:
            return {"error": "no embeddings"}

        E = np.array([np.frombuffer(r[1], dtype=np.float32) for r in rows])
        contents = [r[2][:100] for r in rows]
        msg_ids = [r[0] for r in rows]
        access_energy = self.compute_access_energy()

        k = max(1, int(N * keep_ratio))

        # Standard DCT
        C = dct(E, axis=0, norm='ortho')
        C_trunc = np.zeros_like(C)
        C_trunc[:k] = C[:k]
        R_standard = idct(C_trunc, axis=0, norm='ortho')

        sims_standard = np.array([
            float(np.dot(E[i], R_standard[i]) / (np.linalg.norm(E[i]) * np.linalg.norm(R_standard[i]) + 1e-10))
            for i in range(N)
        ])

        # Promoted DCT
        w = 1.0 + promotion_strength * access_energy
        E_w = E * w[:, np.newaxis]
        C_w = dct(E_w, axis=0, norm='ortho')
        C_pw = np.zeros_like(C_w)
        C_pw[:k] = C_w[:k]
        R_promoted = idct(C_pw, axis=0, norm='ortho') / w[:, np.newaxis]

        sims_promoted = np.array([
            float(np.dot(E[i], R_promoted[i]) / (np.linalg.norm(E[i]) * np.linalg.norm(R_promoted[i]) + 1e-10))
            for i in range(N)
        ])

        delta = sims_promoted - sims_standard
        promoted_idx = np.argsort(delta)[-10:][::-1]
        demoted_idx = np.argsort(delta)[:5]

        return {
            "n_messages": N,
            "k_coefficients": k,
            "keep_ratio": keep_ratio,
            "avg_sim_standard": float(np.mean(sims_standard)),
            "avg_sim_promoted": float(np.mean(sims_promoted)),
            "avg_delta": float(np.mean(delta)),
            "memories_with_access": int(np.sum(access_energy > 0)),
            "promoted": [
                {"idx": int(i), "delta": float(delta[i]),
                 "access_energy": float(access_energy[i]), "content": contents[i]}
                for i in promoted_idx if delta[i] > 0.001
            ],
            "demoted": [
                {"idx": int(i), "delta": float(delta[i]),
                 "access_energy": float(access_energy[i]), "content": contents[i]}
                for i in demoted_idx if delta[i] < -0.001
            ]
        }

    def simulate_life(self, queries: list[str]) -> dict:
        """Run queries to build access patterns, then reconsolidate."""
        for q in queries:
            self.access(q)
        return self.reconsolidate()


def demo():
    """Demo with sample corpus."""
    import os
    from pathlib import Path

    corpus_path = Path(__file__).parent.parent / "data" / "sample_corpus.json"
    db_path = "/tmp/etc_demo.db"

    # Clean start
    if os.path.exists(db_path):
        os.remove(db_path)

    with open(corpus_path) as f:
        sections = json.load(f)["sections"]

    engine = ReconsolidationEngine(db_path)
    print(f"Ingesting {len(sections)} sections...")
    engine.ingest(sections)

    # Simulate access patterns
    queries = [
        "what is the project architecture",
        "how do we deploy",
        "database design decisions",
        "competitor analysis",
        "what is the entity resolution system",
        "deployment pipeline CI/CD",
        "how does the reconsolidation engine work",
        "compression benchmarks DCT vs SVD",
        "what are the three tiers of memory",
        "project architecture and tech stack",  # repeat
        "deployment and infrastructure",        # repeat
        "compression and frequency analysis",   # repeat
    ]

    print(f"\nSimulating {len(queries)} queries...")
    result = engine.simulate_life(queries)

    print(f"\n{'='*60}")
    print("RECONSOLIDATION RESULTS")
    print(f"{'='*60}")
    print(f"Messages: {result['n_messages']}, Kept: {result['k_coefficients']}")
    print(f"Standard DCT avg sim:  {result['avg_sim_standard']:.4f}")
    print(f"Promoted DCT avg sim:  {result['avg_sim_promoted']:.4f}")
    print(f"Avg delta:             {result['avg_delta']:+.4f}")
    print(f"Memories with access:  {result['memories_with_access']}/{result['n_messages']}")

    if result['promoted']:
        print(f"\nPROMOTED (survived compression better after access):")
        for p in result['promoted'][:5]:
            print(f"  Δ{p['delta']:+.4f} (energy={p['access_energy']:.3f}) | {p['content']}")

    if result['demoted']:
        print(f"\nDEMOTED (gave up energy to promoted memories):")
        for d in result['demoted']:
            print(f"  Δ{d['delta']:+.4f} (energy={d['access_energy']:.3f}) | {d['content']}")

    # Cleanup
    os.remove(db_path)


if __name__ == "__main__":
    demo()
