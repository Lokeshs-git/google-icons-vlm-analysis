# workspace-icons-vlm-study

> Does Google's May 2026 Workspace icon redesign produce more separable representations in vision-language model embedding space?

This repository runs a small, reproducible study comparing the **old** flat four-color Google Workspace icons against the **new** gradient redesign (rolled out May 19, 2026) using three families of vision encoders: OpenCLIP, SigLIP, and DINOv2.

The quantitative claim under test:

> **H1**: The mean pairwise cosine distance between Workspace icons in embedding space is greater for the new icon set than the old icon set, across all three encoder families.

If this holds, this is consistent with Google's stated design intent of making apps "more distinct" — and quantifies it on machine perception rather than human perception.

**Result:** Yes. Across all three tested vision encoders (OpenCLIP, SigLIP, and DINOv2), the new icons showed a statistically significant increase in mean pairwise distance (e.g., OpenCLIP separation increased by ~33%), making them more distinguishable to AI vision models.
## What this is not

- Not a claim about computer-use agents.
- Not a claim about frontier multimodal LLMs (Claude, GPT, Gemini). Those use proprietary visual encoders, so embedding-distance results don't transfer directly. Open encoders are used here because the distance metric requires raw embeddings.
- Not a claim about human perception.

## Repo layout

```
.
├── README.md
├── pyproject.toml
├── configs/
│   └── encoders.yaml         # which encoders to run, weights, image size
├── data/
│   ├── icons_manifest.csv    # canonical app list + expected file paths
│   ├── icons_old/            # YOU provide — see "Sourcing icons" below
│   └── icons_new/            # YOU provide
├── src/
│   ├── embed.py              # multi-encoder embedding pipeline
│   ├── distances.py          # pairwise cosine, summary stats
│   ├── stats.py              # bootstrap CIs, Mann-Whitney U
│   └── viz.py                # heatmaps, KDE overlays, UMAP
├── scripts/
│   └── run_study.py          # end-to-end CLI driver
└── results/                  # populated by run_study.py
```

## Quick start

```bash
# 1. Install (Python 3.11+, CUDA optional)
pip install -e .

# 2. Place icon PNGs at the paths in data/icons_manifest.csv
#    (see "Sourcing icons" below)

# 3. Run the full study
python scripts/run_study.py --output results/run_$(date +%Y%m%d)

# 4. Inspect results/run_<date>/summary.md and the PNG plots
```

## Sourcing icons

Icons are not committed to this repo. Google Workspace icons are trademarked; this study uses them under nominative fair use for non-commercial research commentary, and you should source them yourself rather than redistributing through this repo.

For each app in `data/icons_manifest.csv`, save:

- The **old** (pre-May-2026, flat four-color container) PNG at `data/icons_old/<slug>.png`
- The **new** (post-May-2026, gradient) PNG at `data/icons_new/<slug>.png`

Both at the same resolution (recommend 256×256, transparent background). Sources: Wikimedia Commons for the old set, Google's official Workspace press kit for the new set.

## Statistical design

For each encoder *e* ∈ {OpenCLIP ViT-L/14, SigLIP-SO400M, DINOv2 ViT-L/14}:

1. Embed all old icons → distance matrix *D_old(e)*; take the upper triangle as a sample of pairwise distances.
2. Repeat for new icons → *D_new(e)*.
3. Report mean and 95% bootstrap CI of each.
4. Compare distributions with one-sided Mann-Whitney U (H1: new > old).

n = 8 apps ⇒ 28 pairwise distances per condition per encoder. Small but adequate for the headline test; sensitivity is the limiting factor, not type-I error.

## Reproducibility

All encoder weights are pinned in `configs/encoders.yaml`. Random seeds are set in `scripts/run_study.py`. A `results/<run>/manifest.json` records git SHA, encoder versions, and icon file hashes.

If you encounter rate limits or access issues downloading models from Hugging Face, create a `.env` file in the root directory with `HF_TOKEN="your_hugging_face_token"`.

## Citation

If this becomes an article, cite as: Subramanian, L. (2026). *Embedding-space separation of Google Workspace icons before and after the May 2026 redesign.* [in prep].
