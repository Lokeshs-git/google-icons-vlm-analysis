"""Pairwise distance utilities. Pure numpy, no torch."""

from __future__ import annotations

import numpy as np


def pairwise_cosine_distance(embeddings: np.ndarray) -> np.ndarray:
    """Compute (n, n) cosine distance matrix from L2-normalized embeddings.

    Cosine distance = 1 - cosine similarity. Range [0, 2], typical icon range [0, 1].
    Embeddings are assumed already L2-normalized (the encoders in embed.py do this).
    """
    sim = embeddings @ embeddings.T
    # Clamp for numerical safety; floating-point can push to 1.0000001
    sim = np.clip(sim, -1.0, 1.0)
    return 1.0 - sim


def upper_triangle(matrix: np.ndarray) -> np.ndarray:
    """Return the strict upper triangle (i < j) as a 1D array.

    For a 13x13 matrix this gives 78 pairwise distances.
    """
    n = matrix.shape[0]
    i, j = np.triu_indices(n, k=1)
    return matrix[i, j]


def named_pairs(
    matrix: np.ndarray,
    slugs: list[str],
) -> list[tuple[str, str, float]]:
    """Return [(slug_a, slug_b, distance), ...] for all i < j pairs."""
    n = matrix.shape[0]
    out = []
    for i in range(n):
        for j in range(i + 1, n):
            out.append((slugs[i], slugs[j], float(matrix[i, j])))
    return out

