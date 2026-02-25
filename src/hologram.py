#!/usr/bin/env python3
"""
Embedding Trajectory Compression — Core Module

Encodes sequences of sentence embeddings into frequency domain via DCT,
truncates high-frequency components, reconstructs with controlled quality loss.

The key insight: a sequence of sentence embeddings is a trajectory through
semantic space — a waveform amenable to the same transforms used in signal
compression (JPEG, MP3, video codecs). Truncating high-frequency DCT
coefficients produces uniform degradation (resolution loss, not information loss).
"""

import json
import sys
import numpy as np
from scipy.fft import dct, idct
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("pip install sentence-transformers")
    sys.exit(1)


def embed_messages(messages: list[str], model_name: str = "BAAI/bge-small-en-v1.5") -> np.ndarray:
    """Embed each message → (N, D) matrix representing the semantic trajectory."""
    model = SentenceTransformer(model_name)
    return np.array(model.encode(messages, show_progress_bar=False))


def dct_compress(embeddings: np.ndarray, keep_ratio: float = 0.1) -> np.ndarray:
    """
    DCT along the sequence axis, truncate high frequencies.
    keep_ratio: fraction of coefficients to keep (0.1 = keep 10%)
    Returns: truncated coefficient matrix (K × D)
    """
    N = embeddings.shape[0]
    K = max(1, int(N * keep_ratio))
    coefficients = dct(embeddings, axis=0, norm='ortho')
    compressed = np.zeros_like(coefficients)
    compressed[:K, :] = coefficients[:K, :]
    return compressed


def dct_reconstruct(compressed: np.ndarray) -> np.ndarray:
    """Inverse DCT to reconstruct embedding trajectory."""
    return idct(compressed, axis=0, norm='ortho')


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def evaluate_compression(embeddings: np.ndarray, messages: list[str],
                         keep_ratios: list[float] = None) -> list[dict]:
    """
    Full pipeline: embed → compress → reconstruct → evaluate.
    Returns list of result dicts per compression ratio.
    """
    if keep_ratios is None:
        keep_ratios = [1.0, 0.5, 0.25, 0.1, 0.05]

    results = []
    for ratio in keep_ratios:
        K = max(1, int(len(messages) * ratio))
        compressed = dct_compress(embeddings, keep_ratio=ratio)
        reconstructed = dct_reconstruct(compressed)

        sims = []
        correct = 0
        for i, recon in enumerate(reconstructed):
            sim = cosine_similarity(recon, embeddings[i])
            sims.append(sim)
            # Position accuracy: does nearest neighbor match original position?
            all_sims = [cosine_similarity(recon, embeddings[j]) for j in range(len(messages))]
            if np.argmax(all_sims) == i:
                correct += 1

        result = {
            'ratio': ratio,
            'K': K,
            'N': len(messages),
            'avg_sim': float(np.mean(sims)),
            'min_sim': float(np.min(sims)),
            'std_sim': float(np.std(sims)),
            'position_accuracy': correct / len(messages),
            'storage_kb': K * embeddings.shape[1] * 4 / 1024,
            'original_kb': len(messages) * embeddings.shape[1] * 4 / 1024,
        }
        results.append(result)

    return results


def analyze_frequency_bands(embeddings: np.ndarray, n_bands: int = 5) -> list[dict]:
    """Analyze energy distribution across frequency bands."""
    N = embeddings.shape[0]
    coefficients = dct(embeddings, axis=0, norm='ortho')
    total_energy = np.sum(coefficients ** 2)
    band_size = max(1, N // n_bands)

    bands = []
    for b in range(n_bands):
        start = b * band_size
        end = min(start + band_size, N)
        if start >= N:
            break
        band_energy = np.sum(coefficients[start:end, :] ** 2)
        bands.append({
            'band': b,
            'range': f'{start}-{end}',
            'energy_pct': float(band_energy / total_energy * 100),
            'label': 'DC/theme' if b == 0 else ('low' if b < n_bands // 2 else 'high'),
        })

    return bands


def demo():
    """Run demo on sample corpus."""
    corpus_path = Path(__file__).parent.parent / "data" / "sample_corpus.json"
    if not corpus_path.exists():
        print(f"Sample corpus not found at {corpus_path}")
        return

    with open(corpus_path) as f:
        data = json.load(f)

    messages = data["sections"]
    print(f"Loaded {len(messages)} sections from sample corpus")
    print(f"Embedding...")

    embeddings = embed_messages(messages)
    print(f"Embedding matrix: {embeddings.shape} ({embeddings.nbytes / 1024:.1f} KB)")

    print(f"\n{'='*60}")
    print("COMPRESSION RESULTS")
    print(f"{'='*60}")

    results = evaluate_compression(embeddings, messages)
    for r in results:
        print(f"  {r['ratio']*100:>5.1f}% (K={r['K']:>3}) | sim={r['avg_sim']:.4f} ± {r['std_sim']:.4f} | "
              f"pos_acc={r['position_accuracy']*100:.1f}% | {r['storage_kb']:.1f}KB / {r['original_kb']:.1f}KB")

    print(f"\n{'='*60}")
    print("FREQUENCY BAND ANALYSIS")
    print(f"{'='*60}")

    bands = analyze_frequency_bands(embeddings)
    for b in bands:
        print(f"  Band {b['band']} ({b['range']:>6}) | {b['energy_pct']:>6.1f}% energy | {b['label']}")

    # Shuffled baseline
    print(f"\n{'='*60}")
    print("TEMPORAL STRUCTURE TEST (original vs shuffled)")
    print(f"{'='*60}")

    shuffled = embeddings.copy()
    np.random.seed(42)
    np.random.shuffle(shuffled)

    C_orig = dct(embeddings, axis=0, norm='ortho')
    C_shuf = dct(shuffled, axis=0, norm='ortho')

    total_orig = np.sum(C_orig ** 2)
    total_shuf = np.sum(C_shuf ** 2)

    K10 = max(1, len(messages) // 10)
    orig_pct = np.sum(C_orig[:K10] ** 2) / total_orig * 100
    shuf_pct = np.sum(C_shuf[:K10] ** 2) / total_shuf * 100

    print(f"  Energy in lowest 10% coefficients:")
    print(f"    Original (sequential): {orig_pct:.1f}%")
    print(f"    Shuffled (random):     {shuf_pct:.1f}%")
    print(f"    Temporal signal:       +{orig_pct - shuf_pct:.1f}%")


if __name__ == "__main__":
    demo()
