#!/usr/bin/env python3
"""
Downstream Retrieval Evaluation

Compares retrieval accuracy across compression methods:
1. Full embeddings (oracle)
2. DCT at 10% compression
3. DCT + reconsolidation at 10%
4. SVD at matching rank
5. Random baseline

Uses keyword-matched ground truth for each question.
"""

import json
import numpy as np
from scipy.fft import dct, idct
from scipy.linalg import svd
from pathlib import Path
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-small-en-v1.5"


def build_questions(contents: list[str]) -> list[dict]:
    """Generate evaluation questions with keyword-based ground truth."""
    questions = [
        {"q": "What is the project architecture?", "keywords": ["react", "typescript", "tailwind", "dashboard"]},
        {"q": "How do we deploy?", "keywords": ["pm2", "github actions", "deploy", "staging"]},
        {"q": "What database do we use?", "keywords": ["postgresql", "sqlite", "knex", "migration"]},
        {"q": "What is the API design?", "keywords": ["rest", "websocket", "trpc", "rate limit"]},
        {"q": "Who are our competitors?", "keywords": ["competitor", "series a", "spatial depth"]},
        {"q": "What is the entity system?", "keywords": ["entity resolution", "fuzzy", "levenshtein"]},
        {"q": "What is the memory architecture?", "keywords": ["tiered memory", "hot cache", "warm store"]},
        {"q": "What security measures exist?", "keywords": ["csrf", "csp", "cors", "npm audit"]},
        {"q": "How does monitoring work?", "keywords": ["grafana", "latency", "error rate", "alert"]},
        {"q": "What compression methods were tested?", "keywords": ["dct", "svd", "compression", "reconstruction"]},
        {"q": "What is the reconsolidation engine?", "keywords": ["reconsolidation", "access-driven", "frequency promotion"]},
        {"q": "What is the three-tier memory model?", "keywords": ["consolidated", "sharp", "offloaded", "three-tier"]},
        {"q": "What embedding models were tested?", "keywords": ["bge-small", "gte-large", "e5-mistral"]},
        {"q": "What was the user feedback?", "keywords": ["beta tester", "collaborative", "pdf", "dark mode"]},
        {"q": "What is the cost structure?", "keywords": ["vps", "sendgrid", "monthly", "break even"]},
    ]

    for q in questions:
        relevant = []
        for i, content in enumerate(contents):
            cl = content.lower()
            if sum(1 for kw in q["keywords"] if kw.lower() in cl) >= 2:
                relevant.append(i)
        if not relevant:
            for i, content in enumerate(contents):
                if any(kw.lower() in content.lower() for kw in q["keywords"]):
                    relevant.append(i)
        q["relevant_indices"] = relevant

    return [q for q in questions if q["relevant_indices"]]


def evaluate(query_emb, memory_matrix, relevant_indices, top_k=5):
    """Compute retrieval metrics for a single query."""
    sims = np.array([
        np.dot(query_emb, memory_matrix[i]) /
        (np.linalg.norm(query_emb) * np.linalg.norm(memory_matrix[i]) + 1e-10)
        for i in range(memory_matrix.shape[0])
    ])
    ranked = np.argsort(sims)[::-1]

    top1 = 1 if ranked[0] in relevant_indices else 0
    top5 = 1 if any(r in relevant_indices for r in ranked[:top_k]) else 0
    mrr = 0.0
    for rank, idx in enumerate(ranked):
        if idx in relevant_indices:
            mrr = 1.0 / (rank + 1)
            break

    return top1, top5, mrr


def demo():
    corpus_path = Path(__file__).parent.parent / "data" / "sample_corpus.json"
    with open(corpus_path) as f:
        contents = json.load(f)["sections"]

    print(f"Loading model and embedding {len(contents)} sections...")
    model = SentenceTransformer(MODEL_NAME)
    E = np.array(model.encode(contents, show_progress_bar=False))
    N, D = E.shape

    questions = build_questions(contents)
    print(f"Questions with ground truth: {len(questions)}")

    query_embeddings = model.encode([q["q"] for q in questions])

    k = max(1, int(N * 0.1))

    # Build reconstructions
    C = dct(E, axis=0, norm='ortho')
    C_trunc = np.zeros_like(C)
    C_trunc[:k] = C[:k]
    R_dct = idct(C_trunc, axis=0, norm='ortho')

    U, s, Vt = svd(E, full_matrices=False)
    R_svd = U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

    systems = {
        "Full (oracle)": E,
        "SVD rank-K": R_svd,
        "DCT 10%": R_dct,
    }

    print(f"\n{'System':<20} | {'Top-1':<8} | {'Top-5':<8} | {'MRR':<8}")
    print("-" * 52)

    for name, R in systems.items():
        t1s, t5s, mrrs = [], [], []
        for i, q in enumerate(questions):
            t1, t5, mrr = evaluate(query_embeddings[i], R, q["relevant_indices"])
            t1s.append(t1)
            t5s.append(t5)
            mrrs.append(mrr)
        print(f"{name:<20} | {np.mean(t1s):<8.1%} | {np.mean(t5s):<8.1%} | {np.mean(mrrs):<8.4f}")


if __name__ == "__main__":
    demo()
