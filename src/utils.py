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

def build_train_transforms(
    image_size: int,
    is_doppler: bool = False,
    color_augmentation: bool = False,
) -> A.Compose:
    """
    Training augmentations for MSK ultrasound images.

    Colour transforms are governed by color_augmentation:

    color_augmentation=False (default)
        No colour transforms for either modality. Clean geometric-only baseline.

    color_augmentation=True
        Modality-specific domain-robustness block:
        - Doppler: HueSaturationValue + CLAHE + ToGray
          Doppler images are colour-coded by flow velocity; the palette varies
          per machine, so hue/saturation randomisation is the primary fix.
        - Bmode: CLAHE only
          Bmode is essentially greyscale — hue/saturation shifts are meaningless.
          The only meaningful inter-scanner variation is local contrast (gain,
          dynamic range), which CLAHE normalises.
    """
    if color_augmentation:
        if is_doppler:
            colour_transforms = [
                A.HueSaturationValue(
                    hue_shift_limit=20, sat_shift_limit=30,
                    val_shift_limit=20, p=0.7,
                ),
                A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=0.4),
                A.ToGray(p=0.15),
            ]
        else:  # bmode — greyscale, only contrast normalisation makes sense
            colour_transforms = [
                A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=0.5),
            ]
    else:
        colour_transforms = []  # no colour transforms (default)

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
        A.GaussNoise(std_range=(0.009, 0.018), p=0.4),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(image_size // 10, image_size // 10),
            hole_width_range=(image_size // 10, image_size // 10),
            fill=0,
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


# ─── QA Gatekeeper Transforms ─────────────────────────────────────────────────

def build_qa_train_transforms(image_size: int, use_clahe: bool = True) -> A.Compose:
    """
    Training augmentations for the QA Gatekeeper model.

    All images (B-Mode and Doppler) are converted to grayscale first via the
    luminance-weighted average (0.299R + 0.587G + 0.114B), stripping the
    Doppler colour overlay so the backbone sees only structural geometry.
    The output retains 3 identical channels so the pretrained backbone can
    ingest it unchanged.

    After grayscale conversion, CLAHE is optionally applied to normalise
    contrast differences caused by differing scanner gain settings across
    hospitals — the same protocol used for the B-Mode scoring model.

    No HueSaturationValue or random-grayscale transforms are included
    because the images are already achromatic at this point.
    """
    clahe_block = (
        [A.CLAHE(clip_limit=(1.0, 3.0), tile_grid_size=(8, 8), p=0.5)]
        if use_clahe else []
    )

    return A.Compose([
        # ── Grayscale normalisation (strips Doppler colour cues) ──────────────
        A.ToGray(num_output_channels=3, method='weighted_average', p=1.0),
        # ── Geometry ─────────────────────────────────────────────────────────
        A.Rotate(limit=12, border_mode=0, p=0.5),
        A.ElasticTransform(alpha=30, sigma=5, p=0.3),
        A.RandomResizedCrop(
            size=(image_size, image_size),
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
            p=0.4,
        ),
        # ── Contrast ─────────────────────────────────────────────────────────
        *clahe_block,
        # ── Noise / occlusion ────────────────────────────────────────────────
        A.GaussNoise(std_range=(0.009, 0.018), p=0.4),
        A.CoarseDropout(
            num_holes_range=(1, 4),
            hole_height_range=(image_size // 10, image_size // 10),
            hole_width_range=(image_size // 10, image_size // 10),
            fill=0,
            p=0.3,
        ),
        # ── Normalisation ────────────────────────────────────────────────────
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])


def build_qa_val_transforms(image_size: int) -> A.Compose:
    """
    Validation / inference transforms for the QA model.

    Grayscale conversion + ImageNet normalisation only — no augmentation.
    Mirrors build_val_transforms but prepends the grayscale step so that
    validation images are pre-processed identically to training images.
    """
    return A.Compose([
        A.ToGray(num_output_channels=3, method='weighted_average', p=1.0),
        A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ToTensorV2(),
    ])
