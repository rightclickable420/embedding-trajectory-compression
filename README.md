# Embedding Trajectory Compression for Persistent Agent Memory

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18778409.svg)](https://doi.org/10.5281/zenodo.18778409)

**SVD, DCT, and Access-Driven Reconsolidation**

A sequence of sentence embeddings is a trajectory through semantic space — a waveform amenable to the same transforms used in JPEG, MP3, and video compression. This repo implements and evaluates two novel applications of classical transforms to AI agent memory:

1. **Truncated SVD** for variance-optimal compression (best point retrieval)
2. **DCT-based frequency decomposition** with access-driven reconsolidation (adaptive resolution shaped by usage)

Neither approach has been previously applied to sequences of sentence embeddings as a memory compression strategy.

## Results

Evaluated on a 195-section agent memory corpus (30 days of operation):

| Method | 50% Compression | 10% Compression | Top-5 Retrieval (10%) |
|--------|----------------|-----------------|----------------------|
| **SVD** | 0.991 cos sim | 0.929 cos sim | 76% |
| **DCT** | 0.933 cos sim | 0.868 cos sim | 34% |
| **DCT + Reconsolidation** | — | 0.867 cos sim | 36% |

SVD wins on raw quality. DCT provides what SVD cannot: interpretable temporal frequency bands that enable **reconsolidation** — where frequently-accessed memories are physically promoted toward low-frequency components, producing adaptive resolution (sharp where you look, blurry where you don't).

Key reconsolidation result: **+0.032 cosine similarity** for high-access memories, **−0.028** for unaccessed ones. The representation itself changes, not just the ranking.

## Architecture

Three-layer hybrid (like video compression):

```
┌─────────────────────┐
│   Keyframe Window    │  Recent memories at full fidelity (I-frames)
├─────────────────────┤
│   Holographic Core   │  DCT-compressed with access promotion
├─────────────────────┤
│     Fact Store       │  Extracted dates, URLs, IPs, decisions
└─────────────────────┘
```

## Quick Start

```bash
pip install -r requirements.txt

# Basic compression demo
python src/hologram.py

# Reconsolidation with simulated access patterns
python src/reconsolidation.py

# Downstream retrieval benchmark
python src/downstream_eval.py
```

## Files

```
src/
├── hologram.py          # Core DCT compression + evaluation
├── reconsolidation.py   # Access-driven frequency promotion engine
├── field_tracker.py     # Longitudinal telemetry for memory drift
└── downstream_eval.py   # Retrieval benchmark (DCT vs SVD vs full)

data/
└── sample_corpus.json   # Synthetic 50-section agent memory corpus

paper/
└── paper.md             # Full paper (markdown)
```

## How It Works

### Compression
1. Embed each memory section → (N, 384) matrix
2. Apply DCT along sequence axis → frequency coefficients
3. Truncate high-frequency components → K coefficients
4. Reconstruct via inverse DCT → approximate embeddings

### Reconsolidation
1. Track which memories are queried (access energy with exponential decay)
2. Amplify accessed embeddings *before* DCT → their energy shifts to low-frequency bands
3. Truncate → accessed memories survive compression they wouldn't otherwise
4. Divide back out → original scale, but the frequency profile has permanently changed

This is distinct from LRU caches or recency boosting: those change *ranking* without changing the memory. Reconsolidation changes the *representation itself*.

## Key Findings

- **~80% of energy** concentrates in the lowest frequency band (conversations are dominated by slowly-varying structure)
- **Scale-invariant**: compression quality stable from N=30 to N=240
- **+4% temporal signal** over shuffled baseline — most energy concentration comes from embedding correlations, not sequence order
- **Three emergent tiers**: consolidated-into-being (low freq), sharp-and-available (promoted specifics), offloaded (legitimately demoted)

## Paper

See `paper/paper.md` for the full writeup including methodology, baselines, ablations, and a controlled metacognition experiment.

**Citation:**
```
Gill, E. & Ash, K. (2026). Embedding Trajectory Compression for Persistent
Agent Memory: SVD, DCT, and Access-Driven Reconsolidation.
DOI: 10.5281/zenodo.18778409
```

## License

MIT
