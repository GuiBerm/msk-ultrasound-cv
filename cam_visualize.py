#!/usr/bin/env python3
"""
cam_visualize.py — EigenCAM visualisation for the MSK ultrasound models.
=========================================================================
Saves a figure with [original | EigenCAM overlay] pairs, one row per
sampled image, for the specified checkpoint(s).

EigenCAM is used instead of Grad-CAM because:
  • No backprop needed  → faster and works regardless of CORN head complexity
  • No need to pick a specific task / rank to differentiate
  • Produces clean, smooth saliency maps for CNN feature extractors

Target layer: model.backbone.stages[-1]
  For ConvNeXt-Small this is the last spatial stage (768 channels,
  8×8 feature map for 256 px input; 12×12 for 384 px).

Usage
-----
  # B-Mode winner (fold 0 checkpoint)
  venv/bin/python cam_visualize.py \\
      --ckpt artifacts/models/bmode/ConvNext_noBE_noDA_100e_3e-4lr/fold0_best.pth \\
      --modality bmode \\
      --n-samples 4 \\
      --out-dir results/cam/bmode

  # Doppler winner
  venv/bin/python cam_visualize.py \\
      --ckpt artifacts/models/doppler/ConvNext_DA_60e_3e-4lr_none/fold0_best.pth \\
      --modality doppler \\
      --n-samples 4 \\
      --out-dir results/cam/doppler

  # Run both in one call
  venv/bin/python cam_visualize.py \\
      --ckpt artifacts/models/bmode/ConvNext_noBE_noDA_100e_3e-4lr/fold0_best.pth \\
               artifacts/models/doppler/ConvNext_DA_60e_3e-4lr_none/fold0_best.pth \\
      --modality bmode doppler \\
      --n-samples 4 \\
      --out-dir results/cam
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List, Optional

import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from pytorch_grad_cam import EigenCAM
from pytorch_grad_cam.utils.image import show_cam_on_image

from src.config import BmodeConfig, DopplerConfig, JOINT_TYPE_MAP
from src.model import build_model
from src.trainer import Trainer
from src.utils import build_val_transforms, get_device, setup_logging

log = logging.getLogger('msk')

IMAGENET_MEAN = np.array([0.485, 0.456, 0.406])
IMAGENET_STD  = np.array([0.229, 0.224, 0.225])


# ─── Model wrapper ────────────────────────────────────────────────────────────

class _ImageOnlyWrapper(nn.Module):
    """
    Thin wrapper that fixes joint_id so the model signature becomes
    forward(image) → tensor, as expected by pytorch-grad-cam.

    For EigenCAM the *output* is never used (only the hooked activations),
    so we can safely return any task's first-rank logit without affecting
    the saliency map.
    """
    def __init__(self, model: nn.Module, joint_id: torch.Tensor, task_name: str):
        super().__init__()
        self.model     = model
        self.joint_id  = joint_id
        self.task_name = task_name

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        joint_id = self.joint_id.expand(image.size(0)).to(image.device)
        out = self.model(image, joint_id)
        # Return (B, 1) — shape required by pytorch-grad-cam
        return out[self.task_name][:, :1]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def _find_target_layer(backbone: nn.Module) -> nn.Module:
    """
    Return the last spatial stage of the backbone.

    For timm ConvNeXt: backbone.stages[-1]
    Falls back to the last child module that is not a pooling/norm layer
    if the expected attribute is missing.
    """
    if hasattr(backbone, 'stages'):
        return backbone.stages[-1]
    # Fallback: walk children in reverse and pick the first conv-containing block
    for child in reversed(list(backbone.children())):
        if any(isinstance(m, nn.Conv2d) for m in child.modules()):
            return child
    raise RuntimeError(
        "Could not automatically detect the target layer. "
        "Pass --target-layer explicitly (e.g. backbone.stages.3)."
    )


def _load_rgb(image_path: str, image_size: int) -> np.ndarray:
    """Load image as uint8 RGB numpy array, resized to image_size × image_size."""
    img = Image.open(image_path).convert('RGB')
    img = img.resize((image_size, image_size), Image.BILINEAR)
    return np.array(img, dtype=np.uint8)


def _to_tensor(rgb: np.ndarray, transforms) -> torch.Tensor:
    """Apply val transforms and return a (1, 3, H, W) float tensor."""
    augmented = transforms(image=rgb)
    return augmented['image'].unsqueeze(0).float()


def _denormalize(tensor: torch.Tensor) -> np.ndarray:
    """Convert a (3, H, W) normalised tensor back to uint8 RGB for display."""
    arr = tensor.permute(1, 2, 0).cpu().numpy()
    arr = arr * IMAGENET_STD + IMAGENET_MEAN
    arr = np.clip(arr, 0, 1)
    return (arr * 255).astype(np.uint8)


def _get_target_layer_by_path(model: nn.Module, path: str) -> nn.Module:
    """Resolve a dot-separated module path, e.g. 'backbone.stages.3'."""
    parts = path.split('.')
    m = model
    for p in parts:
        m = getattr(m, p)
    return m


# ─── Main extraction ──────────────────────────────────────────────────────────

def run_cam(
    ckpt_path: str,
    modality: str,
    image_dir: str,
    labels_csv: str,
    n_samples: int,
    out_dir: Path,
    device: torch.device,
    target_layer_path: Optional[str],
    joint_name: str,
    seed: int,
) -> None:
    ckpt_path = Path(ckpt_path)
    log.info(f"Loading checkpoint: {ckpt_path}")

    # ── Config ────────────────────────────────────────────────────────────────
    ckpt      = torch.load(str(ckpt_path), map_location='cpu', weights_only=False)
    backbone_name = ckpt.get('backbone_name', 'convnext_small')

    if modality == 'bmode':
        config = BmodeConfig(
            include_bone_erosion=False,
            backbone_name=backbone_name,
            image_dir=image_dir,
        )
    else:
        config = DopplerConfig(
            backbone_name=backbone_name,
            image_dir=image_dir,
        )

    config.image_size = ckpt.get('image_size', 256)   # safe fallback

    # ── Model ─────────────────────────────────────────────────────────────────
    model = build_model(config, pretrained=False).to(device)
    Trainer.load_checkpoint(str(ckpt_path), model, device)
    model.eval()

    # ── Target layer ──────────────────────────────────────────────────────────
    if target_layer_path:
        target_layer = _get_target_layer_by_path(model, target_layer_path)
        log.info(f"Target layer (manual): {target_layer_path}")
    else:
        target_layer = _find_target_layer(model.backbone)
        log.info(f"Target layer (auto): {type(target_layer).__name__}")

    # ── Sample images ─────────────────────────────────────────────────────────
    df = pd.read_csv(labels_csv)
    modality_filter = 'Modo B' if modality == 'bmode' else 'Power Doppler'
    df = df[df['tipo_imagen'] == modality_filter].copy()

    # Remove Hospital B holdout (split < 0)
    if 'split' in df.columns:
        df = df[df['split'] >= 0]

    df = df.drop_duplicates(subset='eco_id').reset_index(drop=True)

    # ── Stratified sampling: one image per grade (0, 1, 2, 3) ────────────────
    grade_col = config.tasks[0].csv_column   # e.g. 'eg_sinovial' or 'pd_sinovial'
    rng = np.random.default_rng(seed)
    sampled_rows = []
    for grade in sorted(df[grade_col].dropna().unique()):
        pool = df[df[grade_col] == grade]
        if pool.empty:
            log.warning(f"No images found for grade {int(grade)} — skipping.")
            continue
        sampled_rows.append(pool.sample(n=1, random_state=int(rng.integers(1e6))))
        log.info(f"  Grade {int(grade)}: sampled 1 of {len(pool)} images.")
    df_sample = pd.concat(sampled_rows).reset_index(drop=True)

    log.info(f"Sampled {len(df_sample)} images for {modality} CAM.")

    # ── Fixed joint_id for wrapper (MCF=0 — doesn't affect EigenCAM output) ──
    joint_id_val = JOINT_TYPE_MAP.get(joint_name, 0)
    joint_id     = torch.tensor([joint_id_val], dtype=torch.long, device=device)
    task_name    = config.tasks[0].name

    wrapped = _ImageOnlyWrapper(model, joint_id, task_name)
    transforms = build_val_transforms(config.image_size)

    cam = EigenCAM(model=wrapped, target_layers=[target_layer])

    # ── Generate and save ─────────────────────────────────────────────────────
    out_dir.mkdir(parents=True, exist_ok=True)
    n_cols = 2   # original | overlay
    n_rows = len(df_sample)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 4))
    if n_rows == 1:
        axes = axes[np.newaxis, :]

    for row_idx, (_, record) in enumerate(df_sample.iterrows()):
        img_path = Path(image_dir) / record['eco_id']
        if not img_path.exists():
            # Try without subdir
            candidates = list(Path(image_dir).rglob(record['eco_id']))
            if not candidates:
                log.warning(f"Image not found: {record['eco_id']} — skipping.")
                continue
            img_path = candidates[0]

        rgb   = _load_rgb(str(img_path), config.image_size)
        inp   = _to_tensor(rgb, transforms).to(device)

        grayscale_cam = cam(input_tensor=inp)[0]   # (H, W)

        rgb_float = rgb.astype(np.float32) / 255.0
        overlay   = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)

        axes[row_idx, 0].imshow(rgb)
        axes[row_idx, 0].axis('off')
        axes[row_idx, 0].set_title(
            f"{record.get('joint_type', '')} | {record.get('eco_id', '')}",
            fontsize=7, pad=2
        )

        axes[row_idx, 1].imshow(overlay)
        axes[row_idx, 1].axis('off')
        axes[row_idx, 1].set_title('EigenCAM', fontsize=7, pad=2)

        # Also save individual pair
        pair = np.concatenate([rgb, overlay], axis=1)
        cv2.imwrite(
            str(out_dir / f"{Path(record['eco_id']).stem}_cam.png"),
            cv2.cvtColor(pair, cv2.COLOR_RGB2BGR)
        )

    fig.suptitle(
        f"EigenCAM — {modality.upper()} | {ckpt_path.parent.name}",
        fontsize=11, fontweight='bold'
    )
    plt.tight_layout()
    grid_path = out_dir / 'cam_grid.png'
    fig.savefig(str(grid_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    log.info(f"Saved grid → {grid_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description='EigenCAM saliency visualisation for MSK ultrasound models.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--ckpt', nargs='+', required=True,
        help='Path(s) to .pth checkpoint file(s).',
    )
    parser.add_argument(
        '--modality', nargs='+', required=True,
        choices=['bmode', 'doppler'],
        help='Modality for each checkpoint (same order as --ckpt).',
    )
    parser.add_argument(
        '--image-dir', type=str, default='pos1',
        help='Root directory containing the images. Default: pos1',
    )
    parser.add_argument(
        '--labels-csv', type=str, default='artifacts/labels_with_splits.csv',
    )
    parser.add_argument(
        '--n-samples', type=int, default=6,
        help='Number of images to sample per model. Default: 6',
    )
    parser.add_argument(
        '--out-dir', type=str, default='results/cam',
        help='Output directory for CAM images. Default: results/cam',
    )
    parser.add_argument(
        '--joint', type=str, default='MCF',
        choices=list(JOINT_TYPE_MAP.keys()),
        help='Joint type ID passed to model wrapper (EigenCAM ignores output so this barely matters).',
    )
    parser.add_argument(
        '--target-layer', type=str, default=None,
        help='Dot-separated path to target layer, e.g. backbone.stages.3. '
             'Auto-detected if omitted.',
    )
    parser.add_argument(
        '--no-cuda', action='store_true',
    )
    parser.add_argument('--seed', type=int, default=42)
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    setup_logging()

    if len(args.ckpt) != len(args.modality):
        raise SystemExit('ERROR: --ckpt and --modality must have the same number of entries.')

    device = torch.device('cpu') if args.no_cuda else get_device()
    log.info(f'Device: {device}')

    for ckpt_path, modality in zip(args.ckpt, args.modality):
        run_cam(
            ckpt_path=ckpt_path,
            modality=modality,
            image_dir=args.image_dir,
            labels_csv=args.labels_csv,
            n_samples=args.n_samples,
            out_dir=Path(args.out_dir) / modality,
            device=device,
            target_layer_path=args.target_layer,
            joint_name=args.joint,
            seed=args.seed,
        )

    log.info('EigenCAM extraction complete.')
