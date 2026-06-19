#!/usr/bin/env python3
"""
extract_oof.py — Out-of-Fold (Hospital A) Prediction Extractor
================================================================
Phase 2: Clinical Validation helper.

For each of the 7 locked model configurations, loops through folds 0–4,
loads the best checkpoint for that fold, runs deterministic inference on
the held-out validation split, and concatenates all 5 folds into a single
CSV ready for confusion-matrix and reliability-diagram analysis.

Output schema (one row per *sample × task*):
    image_id, fold_id, joint_type, true_grade, pred_grade,
    prob_0, prob_1, prob_2, prob_3

CORN → probability conversion
------------------------------
  P(Y ≥ k) = ∏_{j=1}^{k}  sigmoid(logit_j)   (cumulative product)
  P(Y = k) = P(Y ≥ k) − P(Y ≥ k+1)           (discrete probability)

  • prob_0 = 1 − P(Y ≥ 1)
  • prob_1 = P(Y ≥ 1) − P(Y ≥ 2)
  • prob_2 = P(Y ≥ 2) − P(Y ≥ 3)    [if K=3 ranks; else 0]
  • prob_3 = P(Y ≥ 3)                 [if K=3 ranks; else 0]

For models with K=1 rank (bone_erosion): prob_0 = 1−sigmoid, prob_1 = sigmoid,
prob_2 = prob_3 = 0 (schema columns still emitted for consistency).

Usage
-----
  # Run all 7 configurations
  python extract_oof.py

  # Run a specific subset (0-indexed)
  python extract_oof.py --runs 0 2 4

  # Disable CUDA (force CPU)
  python extract_oof.py --no-cuda

  # Override batch size
  python extract_oof.py --batch-size 64
"""
from __future__ import annotations

import argparse
import gc
import logging
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import (
    BmodeConfig, DopplerConfig, QAConfig,
    JOINT_TYPE_MAP, QA_JOINT_TYPE_MAP, QA_MODALITY_MAP,
)
from src.dataset import (
    MSKUltrasoundDataset, msk_collate_fn,
    QADataset, qa_collate_fn,
)
from src.model import build_model, build_qa_model
from src.trainer import Trainer
from src.qa_trainer import QATrainer
from src.utils import get_device, setup_logging

log = logging.getLogger('msk')


# ─── Run Registry ─────────────────────────────────────────────────────────────

@dataclass
class RunSpec:
    """Fully describes a single model configuration to extract."""
    run_id:      int
    label:       str             # human-readable name
    modality:    str             # 'bmode' | 'doppler' | 'qa'
    model_name:  str             # subfolder under artifacts/models/{modality}/
    # Config overrides forwarded to BmodeConfig / DopplerConfig / QAConfig
    config_kwargs: dict = field(default_factory=dict)


# NOTE: Checkpoint paths are derived as:
#   artifacts/models/{modality}/{model_name}/fold{k}_best.pth
# Adjust model_name to match your server's directory names.
RUNS: List[RunSpec] = [
    # ── 0: QA ConvNeXt Gatekeeper (5 joint classes) ───────────────────────────
    RunSpec(
        run_id=0,
        label="QA_ConvNeXt_Gatekeeper",
        modality="qa",
        model_name="qa_convnext",
    ),
    # ── 1: ConvNeXt B-Mode 100e (No Bone Erosion, No DA) ─────────────────────
    RunSpec(
        run_id=1,
        label="ConvNeXt_BMode_100e_noBE_noDA",
        modality="bmode",
        model_name="ConvNext_noBE_noDA_100e_3e-4lr",
        config_kwargs=dict(
            include_bone_erosion=False,
            num_epochs=100,
            learning_rate=3e-4,
            color_augmentation=False,
        ),
    ),
    # ── 2: EfficientNet B-Mode 100e (Architecture Baseline) ──────────────────
    RunSpec(
        run_id=2,
        label="EfficientNet_BMode_100e_noBE_noDA",
        modality="bmode",
        model_name="EfficientNet_noBE_noDA_100e_3e-4lr",
        config_kwargs=dict(
            include_bone_erosion=False,
            backbone_name="efficientnet_b2",
            num_epochs=100,
            learning_rate=3e-4,
            color_augmentation=False,
        ),
    ),
    # ── 3: RadImageNet Dim-256 (Domain Gap Baseline) ──────────────────────────
    RunSpec(
        run_id=3,
        label="RadImageNet_Dim256",
        modality="bmode",
        model_name="RadImageNet_feature-dim256",
        config_kwargs=dict(
            include_bone_erosion=False,
            feature_dim=256,
            color_augmentation=False,
        ),
    ),
    # ── 4: ConvNeXt Doppler DA (Domain Robust Winner) ─────────────────────────
    RunSpec(
        run_id=4,
        label="ConvNeXt_Doppler_DA",
        modality="doppler",
        model_name="ConvNext_DA_60e_3e-4lr",
        config_kwargs=dict(
            color_augmentation=True,
            num_epochs=60,
            learning_rate=3e-4,
        ),
    ),
    # ── 5: ConvNeXt Doppler No-DA (Color Shortcut Baseline) ───────────────────
    RunSpec(
        run_id=5,
        label="ConvNeXt_Doppler_noDA",
        modality="doppler",
        model_name="ConvNext_noDA_60e_3e-4lr_none",
        config_kwargs=dict(
            color_augmentation=False,
            num_epochs=60,
            learning_rate=3e-4,
            backbone_lr_mult=0.05,
        ),
    ),
    # ── 6: ConvNeXt Doppler Fully-Frozen (Capacity Baseline) ─────────────────
    RunSpec(
        run_id=6,
        label="ConvNeXt_Doppler_FullyFrozen",
        modality="doppler",
        model_name="ConvNext_noDA_60e_3e-4lr_fully-frozen",
        config_kwargs=dict(
            color_augmentation=False,
            num_epochs=60,
            learning_rate=3e-4,
            backbone_lr_mult=0.0,   # fully frozen backbone
        ),
    ),
    # ── 7: Bone Erosion Expert (Limitation / Imbalance Baseline) ──────────────
    RunSpec(
        run_id=7,
        label="EfficientNet_BoneErosion_DA",
        modality="bmode",
        model_name="EfficientNet_BE_DA_60e_3e-4lr",  # Ensure this matches your server folder!
        config_kwargs=dict(
            include_bone_erosion=True,
            backbone_name="efficientnet_b2",
            num_epochs=60,
            learning_rate=3e-4,
            color_augmentation=True,
        ),
    ),
]


# ─── CORN → Probabilities ─────────────────────────────────────────────────────

def corn_logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert CORN logits → class probabilities for a 4-column schema.

    The model produces K logits (K = n_ranks).  This function always returns
    a 4-element probability vector [p0, p1, p2, p3] regardless of K, padding
    with zeros when K < 3.

    Conversion (exact CORN decoding):
        cum_prob[k] = P(Y ≥ k+1) = ∏_{j=0}^{k} sigmoid(logits[j])
        p[0]        = 1 − cum_prob[0]
        p[k]        = cum_prob[k-1] − cum_prob[k]  (for k = 1 … K−1)
        p[K]        = cum_prob[K−1]

    Args:
        logits : (B, K)  raw CORN logits

    Returns:
        probs  : (B, 4)  class probabilities, guaranteed to sum to 1
    """
    B, K = logits.shape
    sig  = torch.sigmoid(logits)                        # (B, K)
    cum  = torch.cumprod(sig, dim=-1)                   # (B, K)  P(Y ≥ k+1)

    # Build [p0, p1, ..., pK] — length K+1
    p0   = 1.0 - cum[:, :1]                            # (B, 1)
    pmid = cum[:, :-1] - cum[:, 1:]                    # (B, K-1) if K>1 else empty
    plast = cum[:, -1:]                                 # (B, 1)

    if K == 1:
        # Only two valid classes: 0 and 1
        class_probs = torch.cat([p0, plast], dim=-1)   # (B, 2)
    else:
        class_probs = torch.cat([p0, pmid, plast], dim=-1)  # (B, K+1)

    # Pad to 4 columns (prob_2 and prob_3 stay 0 if K < 3)
    n_classes = class_probs.shape[-1]
    if n_classes < 4:
        pad = torch.zeros(B, 4 - n_classes, dtype=class_probs.dtype, device=class_probs.device)
        class_probs = torch.cat([class_probs, pad], dim=-1)

    return class_probs  # (B, 4)


def corn_probs_to_grade(probs: torch.Tensor) -> torch.Tensor:
    """Argmax of the 4-column probability vector → predicted grade."""
    return probs.argmax(dim=-1)


# ─── Config builders ──────────────────────────────────────────────────────────

def build_config(spec: RunSpec, batch_size: int):
    """Instantiate the correct ModelConfig / QAConfig for a RunSpec."""
    ckpt_dir     = f"artifacts/models/{spec.modality}/{spec.model_name}"
    results_dir  = f"results/{spec.modality}/{spec.model_name}"
    overrides    = dict(
        checkpoint_dir=ckpt_dir,
        results_dir=results_dir,
        batch_size=batch_size,
        **{k: v for k, v in spec.config_kwargs.items()
           if k not in ('include_bone_erosion',)},
    )

    if spec.modality == "bmode":
        include_be = spec.config_kwargs.get("include_bone_erosion", True)
        return BmodeConfig(include_bone_erosion=include_be, **overrides)
    elif spec.modality == "doppler":
        return DopplerConfig(**overrides)
    elif spec.modality == "qa":
        return QAConfig(checkpoint_dir=ckpt_dir, results_dir=results_dir, batch_size=batch_size)
    else:
        raise ValueError(f"Unknown modality: '{spec.modality}'")


# ─── Checkpoint loading ───────────────────────────────────────────────────────

def _load_backbone_name_from_ckpt(ckpt_path: Path, fallback: str) -> str:
    """Read backbone_name from checkpoint metadata, fallback to config default."""
    meta = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    name = meta.get("backbone_name")
    if name is None:
        log.warning(
            f"Checkpoint '{ckpt_path.name}' has no 'backbone_name'. "
            f"Falling back to '{fallback}'."
        )
        return fallback
    return name


def load_scoring_model(
    ckpt_path: Path,
    config,
    device: torch.device,
) -> nn.Module:
    """Load a ModalityModel (bmode / doppler) from checkpoint."""
    config.backbone_name = _load_backbone_name_from_ckpt(
        ckpt_path, config.backbone_name
    )
    model = build_model(config, pretrained=False).to(device)
    Trainer.load_checkpoint(str(ckpt_path), model, device)
    model.eval()
    return model


def load_qa_model_from_ckpt(
    ckpt_path: Path,
    config: QAConfig,
    device: torch.device,
) -> nn.Module:
    """Load a QAModel from checkpoint."""
    config.backbone_name = _load_backbone_name_from_ckpt(
        ckpt_path, config.backbone_name
    )
    model = build_qa_model(config, pretrained=False).to(device)
    QATrainer.load_checkpoint(str(ckpt_path), model, device)
    model.eval()
    return model


# ─── Inference helpers ────────────────────────────────────────────────────────

def _resolve_joint_type_str(joint_id_tensor: torch.Tensor, is_qa: bool) -> List[str]:
    """Convert integer joint_id back to the canonical string label."""
    inv_map = {v: k for k, v in (QA_JOINT_TYPE_MAP if is_qa else JOINT_TYPE_MAP).items()}
    # QA merges Radio+Inter → 4; reverse maps to 'Radiocarpiana' for display
    return [inv_map.get(int(jid), "unknown") for jid in joint_id_tensor.cpu()]


@torch.no_grad()
def run_scoring_inference(
    model: nn.Module,
    loader: DataLoader,
    config,
    device: torch.device,
    fold_idx: int,
) -> List[dict]:
    """
    Run inference for one fold of a scoring model (bmode / doppler).

    Returns a flat list of row dicts, one per (sample, task) pair.
    """
    use_amp = config.use_amp and device.type == "cuda"
    rows: List[dict] = []

    for batch in loader:
        images       = batch["image"].to(device, non_blocking=True)
        joint_id     = batch["joint_id"].to(device, non_blocking=True)
        corn_targets = batch["corn_targets"]
        clin_masks   = batch["clinical_masks"]
        eco_ids      = batch["eco_ids"]

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            preds: Dict[str, torch.Tensor] = model(images, joint_id)

        joint_strs = _resolve_joint_type_str(joint_id, is_qa=False)

        for task in config.tasks:
            task_name    = task.name
            logits       = preds[task_name].float().cpu()   # (B, K)
            probs4       = corn_logits_to_probs(logits)     # (B, 4)
            pred_grades  = corn_probs_to_grade(probs4).numpy()

            # Reconstruct true grade from CORN targets (sum of binary targets)
            corn_t       = corn_targets[task_name]          # (B, K)
            true_grades  = corn_t.sum(dim=-1).long().numpy()
            valid_mask   = clin_masks[task_name].bool().numpy()

            probs_np = probs4.numpy()

            for i, eco_id in enumerate(eco_ids):
                rows.append({
                    "image_id":   eco_id,
                    "fold_id":    fold_idx,
                    "joint_type": joint_strs[i],
                    "task":       task_name,
                    "has_label":  bool(valid_mask[i]),
                    "true_grade": int(true_grades[i]) if valid_mask[i] else None,
                    "pred_grade": int(pred_grades[i]),
                    "prob_0":     float(probs_np[i, 0]),
                    "prob_1":     float(probs_np[i, 1]),
                    "prob_2":     float(probs_np[i, 2]),
                    "prob_3":     float(probs_np[i, 3]),
                })

    return rows


@torch.no_grad()
def run_qa_inference(
    model: nn.Module,
    loader: DataLoader,
    config: QAConfig,
    device: torch.device,
    fold_idx: int,
) -> List[dict]:
    """
    Run inference for one fold of the QA gatekeeper model.

    The QA model is a 5-class classifier, not CORN.  We apply softmax to get
    class probabilities.  The schema columns map to:
        prob_0 = P(MCF)
        prob_1 = P(IFP)
        prob_2 = P(MTF)
        prob_3 = P(Radiocubital distal)
        [prob_4 = P(Wrist Radio+Inter) — NOT in the 4-column schema but logged
         so the sum-to-1 check is easy; it is NOT written to the CSV]

    We write the predicted class (0–4) to pred_grade and the true joint class
    (0–4) to true_grade for downstream confusion-matrix computation.
    Note: only prob_0 … prob_3 are stored; prob_4 is the residual.
    """
    use_amp = config.use_amp and device.type == "cuda"
    rows: List[dict] = []

    for batch in loader:
        images       = batch["image"].to(device, non_blocking=True)
        joint_id     = batch["joint_id"].to(device, non_blocking=True)
        modality_id  = batch["modality_id"].to(device, non_blocking=True)
        eco_ids      = batch["eco_ids"]

        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits: torch.Tensor = model(images, modality_id)  # (B, 5)

        probs_5     = torch.softmax(logits.float(), dim=-1).cpu().numpy()  # (B, 5)
        pred_grades = probs_5.argmax(axis=-1)
        true_grades = joint_id.cpu().numpy()
        joint_strs  = _resolve_joint_type_str(joint_id, is_qa=True)

        for i, eco_id in enumerate(eco_ids):
            rows.append({
                "image_id":   eco_id,
                "fold_id":    fold_idx,
                "joint_type": joint_strs[i],
                "task":       "joint_type_qa",
                "has_label":  True,
                "true_grade": int(true_grades[i]),
                "pred_grade": int(pred_grades[i]),
                "prob_0":     float(probs_5[i, 0]),
                "prob_1":     float(probs_5[i, 1]),
                "prob_2":     float(probs_5[i, 2]),
                "prob_3":     float(probs_5[i, 3]),
            })

    return rows


# ─── Val-loader builders ──────────────────────────────────────────────────────

def build_scoring_val_loader(config, fold_idx: int) -> DataLoader:
    """Build the validation DataLoader for fold *fold_idx* (scoring models)."""
    df_full  = pd.read_csv(config.labels_csv)
    df_pool  = df_full[df_full[config.split_col] >= 0].copy()
    df_val   = df_pool[df_pool[config.split_col] == fold_idx].reset_index(drop=True)

    val_ds = MSKUltrasoundDataset(df_val, config.image_dir, config, is_train=False)
    log.info(f"  Fold {fold_idx} val: {len(val_ds)} samples (modality={config.modality_filter})")

    return DataLoader(
        val_ds,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=msk_collate_fn,
    )


def build_qa_val_loader(config: QAConfig, fold_idx: int) -> DataLoader:
    """Build the validation DataLoader for fold *fold_idx* (QA model)."""
    df_full = pd.read_csv(config.labels_csv)
    df_pool = df_full[df_full[config.split_col] >= 0].copy()
    df_val  = df_pool[df_pool[config.split_col] == fold_idx].copy()

    # Deduplicate (Radio+Inter share same eco_id → same QA label)
    df_val  = df_val.drop_duplicates(subset=config.eco_id_col, keep="first").reset_index(drop=True)
    log.info(f"  Fold {fold_idx} QA val: {len(df_val)} samples (all modalities, deduped)")

    val_ds = QADataset(df_val, config.image_dir, config, is_train=False)

    return DataLoader(
        val_ds,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=qa_collate_fn,
    )


# ─── Per-run extraction ───────────────────────────────────────────────────────

def extract_run(spec: RunSpec, batch_size: int, device: torch.device) -> None:
    """Run full OOF extraction for a single RunSpec and save the CSV."""
    log.info("=" * 70)
    log.info(f"RUN {spec.run_id}: {spec.label}  [{spec.modality}]")
    log.info("=" * 70)

    config = build_config(spec, batch_size)
    ckpt_base = Path(config.checkpoint_dir)

    all_rows: List[dict] = []

    for fold_idx in range(config.n_folds):
        # ── Locate checkpoint ──────────────────────────────────────────────────
        # Support both naming conventions seen in the codebase:
        #   fold{k}_best.pth  (Trainer / QATrainer)
        #   best_model_fold_{k}.pth  (alternative naming requested by user)
        ckpt_candidates = [
            ckpt_base / f"fold{fold_idx}_best.pth",
            ckpt_base / f"best_model_fold_{fold_idx}.pth",
        ]
        ckpt_path: Optional[Path] = next(
            (p for p in ckpt_candidates if p.exists()), None
        )
        if ckpt_path is None:
            log.error(
                f"  Fold {fold_idx}: no checkpoint found. Tried:\n"
                + "\n".join(f"    • {p}" for p in ckpt_candidates)
            )
            log.error("  Skipping fold — output CSV will be incomplete.")
            continue

        log.info(f"  Fold {fold_idx}: loading checkpoint → {ckpt_path.name}")

        # ── Load model ─────────────────────────────────────────────────────────
        if spec.modality == "qa":
            val_loader = build_qa_val_loader(config, fold_idx)
            model = load_qa_model_from_ckpt(ckpt_path, config, device)
            fold_rows = run_qa_inference(model, val_loader, config, device, fold_idx)
        else:
            val_loader = build_scoring_val_loader(config, fold_idx)
            model = load_scoring_model(ckpt_path, config, device)
            fold_rows = run_scoring_inference(model, val_loader, config, device, fold_idx)

        log.info(f"  Fold {fold_idx}: {len(fold_rows)} prediction rows collected.")
        all_rows.extend(fold_rows)

        # ── Explicit CUDA memory cleanup between folds ─────────────────────────
        del model
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

    if not all_rows:
        log.error(f"  No predictions collected for run '{spec.label}'. Skipping save.")
        return

    # ── Assemble DataFrame ─────────────────────────────────────────────────────
    df_out = pd.DataFrame(all_rows)

    # Keep only labelled rows for the "clean" output; NaN true_grade rows are
    # included but flagged so the analyst can filter them.
    output_cols = [
        "image_id", "fold_id", "joint_type", "task",
        "true_grade", "pred_grade",
        "prob_0", "prob_1", "prob_2", "prob_3",
    ]
    df_out = df_out[output_cols]

    # ── Save ───────────────────────────────────────────────────────────────────
    out_dir = Path(config.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hospital_a_oof_predictions.csv"
    df_out.to_csv(out_path, index=False)

    n_labelled = df_out["true_grade"].notna().sum()
    log.info(f"  Saved {len(df_out)} rows ({n_labelled} labelled) → {out_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract OOF predictions from all locked checkpoints (Hospital A).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--runs", type=int, nargs="+",
        metavar="RUN_ID",
        help=(
            "Subset of run IDs (0–6) to extract. "
            "If omitted, all 7 runs are processed."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=64,
        help="Inference batch size (default: 64; doubled internally for val loaders).",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False,
        help="Force CPU inference (useful for debugging).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging()

    device = torch.device("cpu") if args.no_cuda else get_device()
    log.info(f"Device: {device}")

    # Select which runs to process
    run_ids_to_process = set(args.runs) if args.runs else set(r.run_id for r in RUNS)
    selected_runs = [r for r in RUNS if r.run_id in run_ids_to_process]

    if not selected_runs:
        log.error("No runs matched the requested IDs. Exiting.")
        sys.exit(1)

    log.info(f"Processing {len(selected_runs)} run(s): "
             f"{[r.label for r in selected_runs]}")

    for spec in selected_runs:
        try:
            extract_run(spec, batch_size=args.batch_size, device=device)
        except Exception as exc:
            log.exception(f"Run '{spec.label}' failed with error: {exc}")
            # Continue with remaining runs
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    log.info("=" * 70)
    log.info("OOF extraction complete.")
    log.info("=" * 70)
