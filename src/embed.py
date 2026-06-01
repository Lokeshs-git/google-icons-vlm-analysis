"""Embedding pipeline supporting OpenCLIP, SigLIP, and DINOv2."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import os
import numpy as np
import torch
from PIL import Image


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



# ---------- Image loading & preprocessing ----------

def load_and_canonicalize(
    path: Path,
    canvas_size: int,
    background_rgb: tuple[int, int, int],
) -> Image.Image:
    """Load PNG (possibly transparent), composite on neutral background, square-pad.

    All encoders see identically-sized RGB images on the same background so
    differences come from icon content, not from background or aspect ratio.
    """
    img = Image.open(path).convert("RGBA")

    # Square pad while preserving aspect ratio
    w, h = img.size
    side = max(w, h)
    padded = Image.new("RGBA", (side, side), (0, 0, 0, 0))
    padded.paste(img, ((side - w) // 2, (side - h) // 2), img)

    # Resize and composite on neutral background
    padded = padded.resize((canvas_size, canvas_size), Image.LANCZOS)
    bg = Image.new("RGB", (canvas_size, canvas_size), background_rgb)
    bg.paste(padded, mask=padded.split()[3])
    return bg


# ---------- Encoder protocol ----------

class IconEmbedder(Protocol):
    """Returns L2-normalized embeddings of shape (n, d)."""
    name: str

    def embed(self, images: list[Image.Image]) -> np.ndarray: ...


# ---------- Concrete encoders ----------

@dataclass
class OpenClipEmbedder:
    name: str
    model_name: str
    pretrained: str
    image_size: int
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        import open_clip
        self._model, _, self._preprocess = open_clip.create_model_and_transforms(
            self.model_name, pretrained=self.pretrained
        )
        self._model.eval().to(self.device)

    @torch.no_grad()
    def embed(self, images: list[Image.Image]) -> np.ndarray:
        batch = torch.stack([self._preprocess(img) for img in images]).to(self.device)
        feats = self._model.encode_image(batch)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()


@dataclass
class HFVisionEmbedder:
    """Shared implementation for SigLIP and DINOv2 via HuggingFace transformers."""
    name: str
    model_id: str
    image_size: int
    device: str = "cuda" if torch.cuda.is_available() else "cpu"

    def __post_init__(self) -> None:
        from transformers import AutoModel, AutoProcessor
        token = os.environ.get("HF_TOKEN")
        self._processor = AutoProcessor.from_pretrained(self.model_id, token=token)
        self._model = AutoModel.from_pretrained(self.model_id, token=token).eval().to(self.device)

    @torch.no_grad()
    def embed(self, images: list[Image.Image]) -> np.ndarray:
        inputs = self._processor(images=images, return_tensors="pt").to(self.device)
        # Both SigLIP and DINOv2 expose `get_image_features` or pooler_output.
        if hasattr(self._model, "get_image_features"):
            out = self._model.get_image_features(**inputs)
        else:
            out = self._model(**inputs)

        if isinstance(out, torch.Tensor):
            feats = out
        elif hasattr(out, "pooler_output") and out.pooler_output is not None:
            feats = out.pooler_output
        elif hasattr(out, "last_hidden_state") and out.last_hidden_state is not None:
            feats = out.last_hidden_state[:, 0]
        elif isinstance(out, dict):
            if "pooler_output" in out and out["pooler_output"] is not None:
                feats = out["pooler_output"]
            else:
                feats = out["last_hidden_state"][:, 0]
        else:
            feats = out[0]

        feats = feats / feats.norm(dim=-1, keepdim=True)
        return feats.cpu().numpy()


# ---------- Factory ----------

def build_encoder(cfg: dict) -> IconEmbedder:
    """Instantiate an encoder from a config dict (one entry of encoders.yaml)."""
    family = cfg["family"]
    if family == "openclip":
        return OpenClipEmbedder(
            name=cfg["name"],
            model_name=cfg["model"],
            pretrained=cfg["pretrained"],
            image_size=cfg["image_size"],
        )
    if family in {"siglip", "dinov2"}:
        return HFVisionEmbedder(
            name=cfg["name"],
            model_id=cfg["model"],
            image_size=cfg["image_size"],
        )
    raise ValueError(f"Unknown encoder family: {family}")


# ---------- Driver ----------

def embed_icon_set(
    encoder: IconEmbedder,
    icon_dir: Path,
    slugs: list[str],
    canvas_size: int,
    background_rgb: tuple[int, int, int],
) -> tuple[np.ndarray, list[str]]:
    """Embed every <slug>.png in `icon_dir`. Returns (embeddings, ordered_slugs).

    Raises FileNotFoundError listing all missing files at once (not just the first).
    """
    paths = [icon_dir / f"{s}.png" for s in slugs]
    missing = [p for p in paths if not p.exists()]
    if missing:
        raise FileNotFoundError(
            f"Missing icons in {icon_dir}:\n  " + "\n  ".join(str(p) for p in missing)
        )
    images = [load_and_canonicalize(p, canvas_size, background_rgb) for p in paths]
    return encoder.embed(images), slugs
