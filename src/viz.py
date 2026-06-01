"""Plotting utilities. Pure numpy/matplotlib, no torch."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def distance_heatmap(
    matrix: np.ndarray,
    slugs: list[str],
    title: str,
    out_path: Path,
    vmax: float | None = None,
) -> None:
    """Triangular heatmap of pairwise distances."""
    fig, ax = plt.subplots(figsize=(8, 7))
    mask = np.triu(np.ones_like(matrix, dtype=bool), k=1)  # show lower triangle
    sns.heatmap(
        matrix,
        mask=mask,
        xticklabels=slugs,
        yticklabels=slugs,
        cmap="viridis",
        vmin=0,
        vmax=vmax,
        cbar_kws={"label": "cosine distance"},
        square=True,
        ax=ax,
    )
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def distribution_overlay(
    old: np.ndarray,
    new: np.ndarray,
    title: str,
    out_path: Path,
) -> None:
    """KDE overlay of pairwise distance distributions, old vs new."""
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.kdeplot(old, ax=ax, label=f"old (mean={old.mean():.3f})", fill=True, alpha=0.4)
    sns.kdeplot(new, ax=ax, label=f"new (mean={new.mean():.3f})", fill=True, alpha=0.4)
    ax.axvline(old.mean(), linestyle="--", alpha=0.6)
    ax.axvline(new.mean(), linestyle="--", alpha=0.6)
    ax.set_xlabel("pairwise cosine distance")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)


def umap_projection(
    old_embeddings: np.ndarray,
    new_embeddings: np.ndarray,
    slugs: list[str],
    title: str,
    out_path: Path,
    seed: int = 42,
) -> None:
    """2D UMAP of old + new embeddings together, colored by set, labeled by slug."""
    import umap

    combined = np.vstack([old_embeddings, new_embeddings])
    reducer = umap.UMAP(n_components=2, random_state=seed, metric="cosine")
    coords = reducer.fit_transform(combined)

    n = len(slugs)
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.scatter(coords[:n, 0], coords[:n, 1], label="old", marker="o", s=80, alpha=0.7)
    ax.scatter(coords[n:, 0], coords[n:, 1], label="new", marker="^", s=80, alpha=0.7)

    for i, slug in enumerate(slugs):
        ax.annotate(slug, coords[i], fontsize=7, alpha=0.7)
        ax.annotate(slug, coords[i + n], fontsize=7, alpha=0.7)

    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
