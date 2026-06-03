# ─── Utilities ────────────────────────────────────────────────────────────────
"""
Shared utilities: reproducibility, logging, and augmentation pipelines.
"""
from __future__ import annotations

import logging
import random

import albumentations as A
import numpy as np
import torch
from albumentations.pytorch import ToTensorV2

# ─── ImageNet Normalization Constants ─────────────────────────────────────────
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


# ─── Reproducibility ─────────────────────────────────────────────────────────

def seed_everything(seed: int = 42) -> None:
    """Pin all RNG sources for reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ─── Logging ──────────────────────────────────────────────────────────────────

def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure and return the project-wide logger."""
    logging.basicConfig(
        level=level,
        format='%(asctime)s | %(levelname)-5s | %(message)s',
        datefmt='%H:%M:%S',
    )
    return logging.getLogger('msk')


# ─── Device ───────────────────────────────────────────────────────────────────

def get_device() -> torch.device:
    """Return the best available device."""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


# ─── Augmentation Pipelines ──────────────────────────────────────────────────

def build_train_transforms(image_size: int, is_doppler: bool = False) -> A.Compose:
    """
    Training augmentations for MSK ultrasound images.

    B-Mode gets brightness/contrast + CLAHE (gain variation, tissue enhancement).
    Doppler skips colour transforms to preserve flow-encoding semantics.
    Images are already 256×256 — no Resize needed, but we include it defensively
    in case image_size ever changes.
    """
    colour_transforms = [] if is_doppler else [
        A.RandomBrightnessContrast(brightness_limit=0.20, contrast_limit=0.20, p=0.5),
        A.CLAHE(clip_limit=(1.0, 4.0), tile_grid_size=(8, 8), p=0.4),
    ]

    return A.Compose([
        A.Rotate(limit=12, border_mode=0, p=0.5),
        A.ElasticTransform(alpha=30, sigma=5, p=0.3),
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
            p=0.4,
        ),
        *colour_transforms,
        A.GaussNoise(var_limit=(5.0, 20.0), p=0.4),
        A.CoarseDropout(
            max_holes=4,
            max_height=image_size // 10,
            max_width=image_size // 10,
            fill_value=0,
            p=0.3,
        ),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_val_transforms(image_size: int) -> A.Compose:
    """Validation / inference transforms — normalisation only."""
    return A.Compose([
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
