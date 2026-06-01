"""End-to-end study driver.

Usage:
    python scripts/run_study.py --output results/run_20260527

Reads configs/encoders.yaml and data/icons_manifest.csv. For each encoder, embeds
both icon sets, computes distance matrices and statistics, and writes plots + a
markdown summary.
"""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import click
import numpy as np
import pandas as pd
import yaml

# Load .env file at repo root if present
def load_env() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    env_path = repo_root / ".env"
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if "=" in line:
                key, val = line.split("=", 1)
                key = key.strip()
                val = val.strip().strip("'\"")
                os.environ[key] = val

load_env()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# pyrefly: ignore [missing-import]
from src.distances import named_pairs, pairwise_cosine_distance, upper_triangle
from src.embed import build_encoder, embed_icon_set
from src.stats import bootstrap_mean_ci, compare_distributions
from src.viz import distance_heatmap, distribution_overlay, umap_projection


REPO = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO / "configs" / "encoders.yaml"
MANIFEST_PATH = REPO / "data" / "icons_manifest.csv"
OLD_DIR = REPO / "data" / "icons_old"
NEW_DIR = REPO / "data" / "icons_new"


@click.command()
@click.option("--output", "output_dir", required=True, type=click.Path(path_type=Path))
def main(output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    config = yaml.safe_load(CONFIG_PATH.read_text())
    manifest = pd.read_csv(MANIFEST_PATH)
    slugs = manifest["slug"].tolist()

    canvas_size = config["preprocessing"]["canvas_size"]
    bg = tuple(config["preprocessing"]["background_rgb"])
    n_boot = config["stats"]["bootstrap_samples"]
    seed = config["stats"]["seed"]

    _write_manifest(output_dir, config, slugs)

    summary_rows: list[dict] = []

    for enc_cfg in config["encoders"]:
        click.echo(f"\n=== Encoder: {enc_cfg['name']} ===")
        encoder = build_encoder(enc_cfg)

        old_emb, _ = embed_icon_set(encoder, OLD_DIR, slugs, canvas_size, bg)
        new_emb, _ = embed_icon_set(encoder, NEW_DIR, slugs, canvas_size, bg)

        old_dist = pairwise_cosine_distance(old_emb)
        new_dist = pairwise_cosine_distance(new_emb)

        old_flat = upper_triangle(old_dist)
        new_flat = upper_triangle(new_dist)

        old_boot = bootstrap_mean_ci(old_flat, n_boot=n_boot, seed=seed)
        new_boot = bootstrap_mean_ci(new_flat, n_boot=n_boot, seed=seed)
        comp = compare_distributions(old=old_flat, new=new_flat)

        # Plots
        enc_dir = output_dir / enc_cfg["name"]
        enc_dir.mkdir(exist_ok=True)
        shared_vmax = float(max(old_flat.max(), new_flat.max()))
        distance_heatmap(old_dist, slugs, f"{enc_cfg['name']} — old icons",
                         enc_dir / "heatmap_old.png", vmax=shared_vmax)
        distance_heatmap(new_dist, slugs, f"{enc_cfg['name']} — new icons",
                         enc_dir / "heatmap_new.png", vmax=shared_vmax)
        distribution_overlay(old_flat, new_flat,
                             f"{enc_cfg['name']} — pairwise distance distributions",
                             enc_dir / "distribution.png")
        umap_projection(old_emb, new_emb, slugs,
                        f"{enc_cfg['name']} — UMAP of old + new",
                        enc_dir / "umap.png", seed=seed)

        # Raw pair dump for further analysis
        pd.DataFrame(named_pairs(old_dist, slugs),
                     columns=["a", "b", "distance"]).to_csv(enc_dir / "pairs_old.csv", index=False)
        pd.DataFrame(named_pairs(new_dist, slugs),
                     columns=["a", "b", "distance"]).to_csv(enc_dir / "pairs_new.csv", index=False)

        summary_rows.append({
            "encoder": enc_cfg["name"],
            "old_mean": old_boot.mean,
            "old_ci_low": old_boot.ci_low,
            "old_ci_high": old_boot.ci_high,
            "new_mean": new_boot.mean,
            "new_ci_low": new_boot.ci_low,
            "new_ci_high": new_boot.ci_high,
            "u_statistic": comp.u_statistic,
            "p_value": comp.p_value,
            "cliffs_delta": comp.cliffs_delta,
            "direction": comp.direction,
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_dir / "summary.csv", index=False)
    _write_summary_md(output_dir / "summary.md", summary_df, slugs)
    click.echo(f"\nDone. Results in {output_dir}")


def _write_summary_md(path: Path, df: pd.DataFrame, slugs: list[str]) -> None:
    lines = [
        "# Run summary",
        "",
        f"Icons (n={len(slugs)}): {', '.join(slugs)}",
        "",
        "## Mean pairwise cosine distance",
        "",
        "| encoder | old mean [95% CI] | new mean [95% CI] | Δ (new − old) | p (MWU) | Cliff's δ | direction |",
        "|---|---|---|---|---|---|---|",
    ]
    for _, r in df.iterrows():
        delta = r["new_mean"] - r["old_mean"]
        lines.append(
            f"| {r['encoder']} "
            f"| {r['old_mean']:.4f} [{r['old_ci_low']:.4f}, {r['old_ci_high']:.4f}] "
            f"| {r['new_mean']:.4f} [{r['new_ci_low']:.4f}, {r['new_ci_high']:.4f}] "
            f"| {delta:+.4f} | {r['p_value']:.4g} | {r['cliffs_delta']:+.3f} | {r['direction']} |"
        )
    lines += [
        "",
        "See each encoder subdirectory for heatmaps, distributions, and UMAP.",
        "",
    ]
    path.write_text("\n".join(lines))


def _write_manifest(output_dir: Path, config: dict, slugs: list[str]) -> None:
    """Record git SHA, config, and icon file hashes for reproducibility."""
    try:
        sha = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        sha = "unknown"

    def hash_dir(d: Path) -> dict[str, str]:
        out = {}
        for s in slugs:
            p = d / f"{s}.png"
            if p.exists():
                out[s] = hashlib.sha256(p.read_bytes()).hexdigest()[:16]
        return out

    manifest = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "git_sha": sha,
        "config": config,
        "icon_hashes": {
            "old": hash_dir(OLD_DIR),
            "new": hash_dir(NEW_DIR),
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
