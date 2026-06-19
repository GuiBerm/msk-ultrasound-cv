#!/usr/bin/env python3
"""
extract_test.py — Hospital B Blind Test Prediction Extractor
==============================================================
Phase 2: Clinical Validation helper.

For each of the 7 locked model configurations, loads the Fold 0 checkpoint
(or an optional ensemble of all 5 folds), runs deterministic inference on
the 50 held-out Hospital B (Coruña) images, and saves the predictions to a
CSV ready for confusion-matrix and reliability-diagram analysis.

Output schema (one row per *sample × task*):
    image_id, fold_id, joint_type, true_grade, pred_grade,
    prob_0, prob_1, prob_2, prob_3

IMPORTANT — blind-test etiquette
---------------------------------
Hospital B images must NEVER be used during training, validation or
hyperparameter tuning.  This script is Phase 2 final evaluation only.
Run it once per model configuration; do not iterate on it.

Ensemble mode
-------------
By default, only the Fold 0 checkpoint is used.  Pass --ensemble to average
logits across all 5 folds (soft ensemble).

Usage
-----
  # Single-fold (fold 0) for all 7 configurations
  python extract_test.py

  # Full 5-fold ensemble for all 7 configurations
  python extract_test.py --ensemble

  # Subset of runs with ensemble
  python extract_test.py --runs 0 4 --ensemble

  # Force CPU
  python extract_test.py --no-cuda

  # Override batch size
  python extract_test.py --batch-size 32
"""
from __future__ import annotations

import argparse
import gc
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

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

log = logging.getLogger("msk")


# ─── Run Registry (identical to extract_oof.py) ───────────────────────────────

@dataclass
class RunSpec:
    """Fully describes a single model configuration to extract."""
    run_id:      int
    label:       str
    modality:    str             # 'bmode' | 'doppler' | 'qa'
    model_name:  str             # subfolder under artifacts/models/{modality}/
    config_kwargs: dict = field(default_factory=dict)


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
            backbone_lr_mult=0.0,
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


# ─── CORN → Probabilities (same logic as extract_oof.py) ─────────────────────

def corn_logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """
    Convert CORN logits → 4-column class probabilities.

    Always returns shape (B, 4); columns beyond the model's actual rank count
    are zero-padded for schema consistency.
    """
    B, K = logits.shape
    sig  = torch.sigmoid(logits)
    cum  = torch.cumprod(sig, dim=-1)

    p0   = 1.0 - cum[:, :1]
    pmid = cum[:, :-1] - cum[:, 1:]
    plast = cum[:, -1:]

    if K == 1:
        class_probs = torch.cat([p0, plast], dim=-1)
    else:
        class_probs = torch.cat([p0, pmid, plast], dim=-1)

    n_classes = class_probs.shape[-1]
    if n_classes < 4:
        pad = torch.zeros(B, 4 - n_classes, dtype=class_probs.dtype, device=class_probs.device)
        class_probs = torch.cat([class_probs, pad], dim=-1)

    return class_probs


def corn_probs_to_grade(probs: torch.Tensor) -> torch.Tensor:
    """Argmax of the 4-column probability vector → predicted grade."""
    return probs.argmax(dim=-1)


# ─── Config / checkpoint helpers ──────────────────────────────────────────────

def build_config(spec: RunSpec, batch_size: int):
    """Instantiate the correct ModelConfig / QAConfig for a RunSpec."""
    ckpt_dir    = f"artifacts/models/{spec.modality}/{spec.model_name}"
    results_dir = f"results/{spec.modality}/{spec.model_name}"
    overrides   = dict(
        checkpoint_dir=ckpt_dir,
        results_dir=results_dir,
        batch_size=batch_size,
        **{k: v for k, v in spec.config_kwargs.items()
           if k not in ("include_bone_erosion",)},
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


def _load_backbone_name(ckpt_path: Path, fallback: str) -> str:
    meta = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    name = meta.get("backbone_name")
    if name is None:
        log.warning(f"No 'backbone_name' in '{ckpt_path.name}'. Using '{fallback}'.")
        return fallback
    return name


def _find_checkpoint(ckpt_base: Path, fold_idx: int) -> Optional[Path]:
    """Locate a fold checkpoint, supporting both naming conventions."""
    candidates = [
        ckpt_base / f"fold{fold_idx}_best.pth",
        ckpt_base / f"best_model_fold_{fold_idx}.pth",
    ]
    return next((p for p in candidates if p.exists()), None)


def load_scoring_models(
    ckpt_paths: List[Path],
    config,
    device: torch.device,
) -> List[nn.Module]:
    """Load one ModalityModel per checkpoint path (all in eval mode)."""
    # Infer backbone from the first checkpoint so architecture matches
    config.backbone_name = _load_backbone_name(ckpt_paths[0], config.backbone_name)
    models = []
    for ckpt in ckpt_paths:
        m = build_model(config, pretrained=False).to(device)
        Trainer.load_checkpoint(str(ckpt), m, device)
        m.eval()
        models.append(m)
    return models


def load_qa_models(
    ckpt_paths: List[Path],
    config: QAConfig,
    device: torch.device,
) -> List[nn.Module]:
    """Load one QAModel per checkpoint path (all in eval mode)."""
    config.backbone_name = _load_backbone_name(ckpt_paths[0], config.backbone_name)
    models = []
    for ckpt in ckpt_paths:
        m = build_qa_model(config, pretrained=False).to(device)
        QATrainer.load_checkpoint(str(ckpt), m, device)
        m.eval()
        models.append(m)
    return models


# ─── Holdout (Hospital B) dataset builder ────────────────────────────────────

def build_test_loader_scoring(config, df_test: pd.DataFrame) -> DataLoader:
    """Scoring model test loader (applies modality filter)."""
    test_ds = MSKUltrasoundDataset(df_test, config.image_dir, config, is_train=False)
    if len(test_ds) == 0:
        return None
    log.info(f"  Hospital B scoring set ({config.modality_filter}): {len(test_ds)} samples.")
    return DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=msk_collate_fn,
    )


def build_test_loader_qa(config: QAConfig, df_test: pd.DataFrame) -> DataLoader:
    """QA model test loader (all modalities; deduplicated by eco_id)."""
    df_dedup = df_test.drop_duplicates(subset=config.eco_id_col, keep="first").reset_index(drop=True)
    log.info(f"  Hospital B QA set: {len(df_dedup)} samples (deduped from {len(df_test)}).")
    test_ds = QADataset(df_dedup, config.image_dir, config, is_train=False)
    return DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=qa_collate_fn,
    )


def identify_hospital_b(config, n_folds: int = 5) -> pd.DataFrame:
    """
    Return the Hospital B holdout rows from the master CSV.

    Hospital B rows have split < 0 (i.e., NOT in {0, 1, 2, 3, 4}).
    """
    df = pd.read_csv(config.labels_csv)
    df_test = df[~df[config.split_col].isin(range(n_folds))].copy()
    if len(df_test) == 0:
        log.error(
            "No Hospital B rows found (expected split < 0). "
            "Check artifacts/labels_with_splits.csv."
        )
    else:
        log.info(f"Hospital B holdout: {len(df_test)} total rows "
                 f"({df_test[config.eco_id_col].nunique()} unique images).")
    return df_test


# ─── Ensemble inference ───────────────────────────────────────────────────────

@torch.no_grad()
def ensemble_predict_scoring(
    models: List[nn.Module],
    images: torch.Tensor,
    joint_id: torch.Tensor,
    config,
    device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Average logits across all fold models (soft ensemble) for scoring models."""
    use_amp = config.use_amp and device.type == "cuda"
    all_logits: Dict[str, List[torch.Tensor]] = {t.name: [] for t in config.tasks}

    for model in models:
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            preds = model(images, joint_id)
        for task_name, logits in preds.items():
            all_logits[task_name].append(logits.float())

    return {
        task_name: torch.stack(llist).mean(dim=0)
        for task_name, llist in all_logits.items()
    }


@torch.no_grad()
def ensemble_predict_qa(
    models: List[nn.Module],
    images: torch.Tensor,
    modality_ids: torch.Tensor,
    config: QAConfig,
    device: torch.device,
) -> torch.Tensor:
    """Average logits across all fold models (soft ensemble) for the QA model."""
    use_amp = config.use_amp and device.type == "cuda"
    all_logits: List[torch.Tensor] = []

    for model in models:
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images, modality_ids)
        all_logits.append(logits.float())

    return torch.stack(all_logits).mean(dim=0)


# ─── Full-run inference ───────────────────────────────────────────────────────

def _resolve_joint_type_str(joint_id_tensor: torch.Tensor, is_qa: bool) -> List[str]:
    inv_map = {v: k for k, v in (QA_JOINT_TYPE_MAP if is_qa else JOINT_TYPE_MAP).items()}
    return [inv_map.get(int(jid), "unknown") for jid in joint_id_tensor.cpu()]


@torch.no_grad()
def infer_scoring(
    models: List[nn.Module],
    loader: DataLoader,
    config,
    device: torch.device,
    fold_label: str,
) -> List[dict]:
    """
    Run ensemble inference over the test loader for scoring models.

    fold_label is stored in the 'fold_id' column to indicate how many
    checkpoints contributed (e.g. 'fold0' or 'ensemble_5').
    """
    rows: List[dict] = []

    for batch in loader:
        images       = batch["image"].to(device, non_blocking=True)
        joint_id     = batch["joint_id"].to(device, non_blocking=True)
        corn_targets = batch["corn_targets"]
        clin_masks   = batch["clinical_masks"]
        eco_ids      = batch["eco_ids"]

        mean_preds = ensemble_predict_scoring(models, images, joint_id, config, device)
        joint_strs = _resolve_joint_type_str(joint_id, is_qa=False)

        for task in config.tasks:
            task_name   = task.name
            logits      = mean_preds[task_name].cpu()
            probs4      = corn_logits_to_probs(logits)
            pred_grades = corn_probs_to_grade(probs4).numpy()

            corn_t      = corn_targets[task_name]
            true_grades = corn_t.sum(dim=-1).long().numpy()
            valid_mask  = clin_masks[task_name].bool().numpy()
            probs_np    = probs4.numpy()

            for i, eco_id in enumerate(eco_ids):
                rows.append({
                    "image_id":   eco_id,
                    "fold_id":    fold_label,
                    "joint_type": joint_strs[i],
                    "task":       task_name,
                    "true_grade": int(true_grades[i]) if valid_mask[i] else None,
                    "pred_grade": int(pred_grades[i]),
                    "prob_0":     float(probs_np[i, 0]),
                    "prob_1":     float(probs_np[i, 1]),
                    "prob_2":     float(probs_np[i, 2]),
                    "prob_3":     float(probs_np[i, 3]),
                })

    return rows


@torch.no_grad()
def infer_qa(
    models: List[nn.Module],
    loader: DataLoader,
    config: QAConfig,
    device: torch.device,
    fold_label: str,
) -> List[dict]:
    """
    Run ensemble inference over the test loader for the QA model.

    The QA model outputs 5-class softmax probabilities.  Only prob_0…prob_3
    are stored in the fixed-width schema (prob_4 is the residual Wrist class).
    """
    rows: List[dict] = []

    for batch in loader:
        images       = batch["image"].to(device, non_blocking=True)
        joint_id     = batch["joint_id"].to(device, non_blocking=True)
        modality_id  = batch["modality_id"].to(device, non_blocking=True)
        eco_ids      = batch["eco_ids"]

        mean_logits  = ensemble_predict_qa(models, images, modality_id, config, device)
        probs_5      = torch.softmax(mean_logits, dim=-1).cpu().numpy()
        pred_grades  = probs_5.argmax(axis=-1)
        true_grades  = joint_id.cpu().numpy()
        joint_strs   = _resolve_joint_type_str(joint_id, is_qa=True)

        for i, eco_id in enumerate(eco_ids):
            rows.append({
                "image_id":   eco_id,
                "fold_id":    fold_label,
                "joint_type": joint_strs[i],
                "task":       "joint_type_qa",
                "true_grade": int(true_grades[i]),
                "pred_grade": int(pred_grades[i]),
                "prob_0":     float(probs_5[i, 0]),
                "prob_1":     float(probs_5[i, 1]),
                "prob_2":     float(probs_5[i, 2]),
                "prob_3":     float(probs_5[i, 3]),
            })

    return rows


# ─── Per-run extraction ───────────────────────────────────────────────────────

def extract_run(
    spec: RunSpec,
    batch_size: int,
    device: torch.device,
    use_ensemble: bool,
) -> None:
    """Run full Hospital B extraction for a single RunSpec and save the CSV."""
    log.info("=" * 70)
    log.info(f"RUN {spec.run_id}: {spec.label}  [{spec.modality}]")
    log.info(f"  Mode: {'5-fold ensemble' if use_ensemble else 'fold-0 single checkpoint'}")
    log.info("=" * 70)

    config   = build_config(spec, batch_size)
    ckpt_base = Path(config.checkpoint_dir)

    # ── Resolve which checkpoints to load ─────────────────────────────────────
    if use_ensemble:
        fold_indices = list(range(config.n_folds))
    else:
        fold_indices = [0]

    ckpt_paths: List[Path] = []
    for fi in fold_indices:
        p = _find_checkpoint(ckpt_base, fi)
        if p is None:
            log.warning(
                f"  Fold {fi}: checkpoint not found under '{ckpt_base}'. "
                "Skipping this fold from ensemble."
            )
        else:
            ckpt_paths.append(p)
            log.info(f"  Fold {fi}: {p.name}")

    if not ckpt_paths:
        log.error(f"  No checkpoints found for run '{spec.label}'. Aborting.")
        return

    fold_label = f"ensemble_{len(ckpt_paths)}" if use_ensemble else "fold0"

    # ── Identify Hospital B rows ───────────────────────────────────────────────
    df_test = identify_hospital_b(config, n_folds=config.n_folds)
    if len(df_test) == 0:
        log.error("  Hospital B set is empty. Aborting.")
        return

    # ── Build DataLoader ───────────────────────────────────────────────────────
    if spec.modality == "qa":
        loader = build_test_loader_qa(config, df_test)
    else:
        loader = build_test_loader_scoring(config, df_test)
        if loader is None:
            log.info(f"  No Hospital B samples for modality '{config.modality_filter}'. Done.")
            return

    # ── Load models ────────────────────────────────────────────────────────────
    if spec.modality == "qa":
        models = load_qa_models(ckpt_paths, config, device)
        rows = infer_qa(models, loader, config, device, fold_label)
    else:
        models = load_scoring_models(ckpt_paths, config, device)
        rows = infer_scoring(models, loader, config, device, fold_label)

    log.info(f"  {len(rows)} prediction rows collected.")

    # ── Cleanup CUDA memory ────────────────────────────────────────────────────
    del models
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    if not rows:
        log.error("  No prediction rows generated. Skipping save.")
        return

    # ── Assemble & save DataFrame ──────────────────────────────────────────────
    df_out = pd.DataFrame(rows)
    output_cols = [
        "image_id", "fold_id", "joint_type", "task",
        "true_grade", "pred_grade",
        "prob_0", "prob_1", "prob_2", "prob_3",
    ]
    df_out = df_out[output_cols]

    out_dir = Path(config.results_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "hospital_b_test_predictions.csv"
    df_out.to_csv(out_path, index=False)

    n_labelled = df_out["true_grade"].notna().sum()
    log.info(f"  Saved {len(df_out)} rows ({n_labelled} labelled) → {out_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Hospital B predictions from all locked checkpoints.",
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
        "--ensemble", action="store_true", default=False,
        help=(
            "Use all 5 fold checkpoints (soft ensemble). "
            "Default: fold 0 only."
        ),
    )
    parser.add_argument(
        "--batch-size", type=int, default=32,
        help="Inference batch size (default: 32).",
    )
    parser.add_argument(
        "--no-cuda", action="store_true", default=False,
        help="Force CPU inference.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    setup_logging()

    device = torch.device("cpu") if args.no_cuda else get_device()
    log.info(f"Device: {device}")
    log.info(
        "⚠️  HOSPITAL B BLIND TEST — ensure modeling decisions are finalised "
        "before running this script."
    )

    run_ids_to_process = set(args.runs) if args.runs else set(r.run_id for r in RUNS)
    selected_runs = [r for r in RUNS if r.run_id in run_ids_to_process]

    if not selected_runs:
        log.error("No runs matched the requested IDs. Exiting.")
        sys.exit(1)

    log.info(
        f"Processing {len(selected_runs)} run(s): "
        f"{[r.label for r in selected_runs]}"
    )

    for spec in selected_runs:
        try:
            extract_run(
                spec,
                batch_size=args.batch_size,
                device=device,
                use_ensemble=args.ensemble,
            )
        except Exception as exc:
            log.exception(f"Run '{spec.label}' failed: {exc}")
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

    log.info("=" * 70)
    log.info("Hospital B extraction complete.")
    log.info("=" * 70)
