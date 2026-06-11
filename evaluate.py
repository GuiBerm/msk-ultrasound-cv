#!/usr/bin/env python3
"""Evaluate the trained model on the Hospital B blind test set.

Checkpoints can be specified in three ways (all equivalent):

  # 1. Explicit list — pass --checkpoints multiple times
  python evaluate.py --model bmode --name baseline_v1 \\
      --checkpoints artifacts/models/bmode/baseline_v1/fold0_best.pth \\
      --checkpoints artifacts/models/bmode/baseline_v1/fold1_best.pth

  # 2. Shell glob — let the shell expand it
  python evaluate.py --model bmode --name baseline_v1 \\
      --checkpoints 'artifacts/models/bmode/baseline_v1/fold*_best.pth'

  # 3. Auto-discover — omit --checkpoints entirely and let the script
  #    find every fold*_best.pth inside artifacts/models/{model}/{name}
  python evaluate.py --model bmode --name baseline_v1

Predictions from all checkpoints are **averaged in logit space** before
decoding to ordinal scores (soft ensemble).
"""
from __future__ import annotations

import argparse
import logging
import numpy as np
from pathlib import Path

import pandas as pd
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.data import DataLoader

from src.config import BmodeConfig, DopplerConfig
from src.dataset import MSKUltrasoundDataset, msk_collate_fn
from src.metrics import MetricAccumulator
from src.model import build_model
from src.trainer import Trainer
from src.utils import get_device, setup_logging


def resolve_checkpoints(checkpoints_arg: list[str] | None, default_dir: str) -> list[Path]:
    """Return a sorted list of checkpoint paths.

    Priority:
      1. Explicit paths / shell-expanded globs passed via --checkpoints.
      2. Auto-discover: every fold*_best.pth inside *default_dir*.
    """
    if checkpoints_arg:
        # argparse already received the (possibly shell-expanded) list;
        # but the user may also have passed a single glob string on some
        # shells — expand each entry just in case.
        paths: list[Path] = []
        for entry in checkpoints_arg:
            expanded = sorted(Path('.').glob(entry)) if '*' in entry or '?' in entry else [Path(entry)]
            paths.extend(expanded)
        if not paths:
            raise FileNotFoundError(
                f"No checkpoint files matched: {checkpoints_arg}"
            )
        return sorted(set(paths))

    # Auto-discover
    discovered = sorted(Path(default_dir).glob('fold*_best.pth'))
    if not discovered:
        raise FileNotFoundError(
            f"No fold*_best.pth checkpoints found in '{default_dir}'. "
            "Pass --checkpoints explicitly or run train.py first."
        )
    return discovered


def load_models(checkpoints: list[Path], config, device: torch.device) -> list[torch.nn.Module]:
    """Build and load one model per checkpoint. All returned in eval mode.

    The backbone name is read directly from the first checkpoint so that the
    correct architecture is reconstructed without any extra CLI flags.
    pretrained=False skips the HuggingFace download — the checkpoint weights
    overwrite everything immediately anyway.
    """
    # ── Infer backbone from checkpoint metadata ───────────────────────────────
    _log = logging.getLogger('msk')
    first_meta = torch.load(str(checkpoints[0]), map_location='cpu', weights_only=False)
    backbone_name = first_meta.get('backbone_name')
    if backbone_name is None:
        _log.warning(
            "Checkpoint has no 'backbone_name' key (old format). "
            f"Falling back to config default: '{config.backbone_name}'. "
            "Re-train to make checkpoints self-describing."
        )
        backbone_name = config.backbone_name
    else:
        _log.info(f"Backbone inferred from checkpoint: '{backbone_name}'")
    config.backbone_name = backbone_name

    models = []
    for ckpt in checkpoints:
        m = build_model(config, pretrained=False).to(device)
        Trainer.load_checkpoint(str(ckpt), m, device)
        m.eval()
        models.append(m)
    return models


def ensemble_predict(models: list[torch.nn.Module],
                     images: torch.Tensor,
                     joint_id: torch.Tensor,
                     config,
                     device: torch.device) -> dict[str, torch.Tensor]:
    """Run all models and return mean logits (soft ensemble)."""
    use_amp = config.use_amp and device.type == 'cuda'
    all_logits: dict[str, list[torch.Tensor]] = {t: [] for t in config.task_names}

    for model in models:
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            preds = model(images, joint_id)
        for task_name, logits in preds.items():
            all_logits[task_name].append(logits)

    return {
        task_name: torch.stack(logit_list).mean(dim=0)
        for task_name, logit_list in all_logits.items()
    }


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate on Hospital B blind test set (fold ensemble)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument('--model', type=str, choices=['bmode', 'doppler'], required=True,
                        help="Model modality to evaluate.")
    parser.add_argument('--name', type=str, required=True,
                        help=(
                            "Name of the training run to evaluate. "
                            "Auto-discover will look for checkpoints under "
                            "artifacts/models/{model}/{name}/ and results are "
                            "saved to results/{model}/{name}/."
                        ))
    parser.add_argument(
        '--checkpoints', type=str, nargs='+', default=None,
        metavar='GLOB_OR_PATH',
        help=(
            "One or more checkpoint paths or glob patterns "
            "(e.g. 'artifacts/models/bmode/baseline_v1/fold*_best.pth'). "
            "If omitted, all fold*_best.pth files under "
            "artifacts/models/{model}/{name}/ are used automatically."
        ),
    )
    parser.add_argument('--batch-size', type=int, default=32,
                        help="Batch size for inference (default: 32).")
    parser.add_argument(
        '--no-bone-erosion', action='store_true', default=False,
        help=(
            '[bmode only] Drop the bone_erosion task — must match the flag '
            'used during training, otherwise the loaded architecture will '
            'not match the checkpoint. Has no effect for doppler.'
        ),
    )
    args = parser.parse_args()

    log = setup_logging()

    # Build paths from --model and --name, consistent with train.py
    checkpoint_dir = f'artifacts/models/{args.model}/{args.name}'
    results_dir    = f'results/{args.model}/{args.name}'

    overrides = dict(
        batch_size=args.batch_size,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
    )

    if args.model == 'bmode':
        include_bone_erosion = not args.no_bone_erosion
        config = BmodeConfig(include_bone_erosion=include_bone_erosion, **overrides)
    else:
        if args.no_bone_erosion:
            log.warning("--no-bone-erosion has no effect for doppler models (ignored).")
        config = DopplerConfig(**overrides)

    device = get_device()

    log.info(f"Run name        : '{args.name}'")
    log.info(f"Tasks           : {config.task_names}")
    log.info(f"Checkpoints dir : {checkpoint_dir}")
    log.info(f"Results dir     : {results_dir}")

    # ── Resolve checkpoints ───────────────────────────────────────────────────
    try:
        checkpoints = resolve_checkpoints(args.checkpoints, config.checkpoint_dir)
    except FileNotFoundError as exc:
        log.error(str(exc))
        return

    log.warning("=" * 70)
    log.warning(" HOSPITAL B — BLIND TEST EVALUATION")
    log.warning(" This script should ONLY be run ONCE after all modeling decisions are final.")
    log.warning("=" * 70)
    log.info(f"Ensemble size: {len(checkpoints)} checkpoint(s)")
    for ckpt in checkpoints:
        log.info(f"  • {ckpt}")

    # ── Identify Hospital B rows ──────────────────────────────────────────────
    df = pd.read_csv(config.labels_csv)
    df_test = df[~df[config.split_col].isin(range(config.n_folds))]

    if len(df_test) == 0:
        log.warning("No Hospital B holdout rows found (split < 0). Attempting fallback identification...")
        # Fallback: Coruña eco_ids
        # Based on data analysis, usually Coruña hospital IDs start with 'coruna' or similar,
        # but the manifest said "coruna": 50. In labels_with_splits.csv, it's model_group or similar.
        # But if split is fully populated, the above should work.
        pass

    if len(df_test) == 0:
        log.error("Could not identify Hospital B holdout set. Aborting.")
        return

    log.info(f"Found {len(df_test)} Hospital B holdout samples.")

    # ── Dataset / DataLoader ──────────────────────────────────────────────────
    test_ds = MSKUltrasoundDataset(df_test, config.image_dir, config, is_train=False)

    if len(test_ds) == 0:
        log.info(f"No Hospital B samples for modality {config.modality_filter}. Done.")
        return

    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=msk_collate_fn,
    )

    # ── Load all models ───────────────────────────────────────────────────────
    models = load_models(checkpoints, config, device)

    # ── Inference ─────────────────────────────────────────────────────────────
    task_n_ranks = {t.name: t.n_ranks for t in config.tasks}
    accumulator = MetricAccumulator(config.task_names, task_n_ranks)

    with torch.no_grad():
        for batch in test_loader:
            images         = batch['image'].to(device, non_blocking=True)
            joint_id       = batch['joint_id'].to(device, non_blocking=True)
            corn_targets   = {t: v.to(device, non_blocking=True) for t, v in batch['corn_targets'].items()}
            clinical_masks = {t: v.to(device, non_blocking=True) for t, v in batch['clinical_masks'].items()}

            ensemble_preds = ensemble_predict(models, images, joint_id, config, device)
            accumulator.update(ensemble_preds, corn_targets, clinical_masks)

    # ── Report ────────────────────────────────────────────────────────────────
    metrics = accumulator.compute()

    log.info("=" * 70)
    log.info(f"HOSPITAL B RESULTS ({args.model.upper()}) | run: '{args.name}'  —  ensemble of {len(models)} model(s)")
    log.info("=" * 70)

    summary_rows = []
    for task_name, m in metrics.items():
        log.info(f"  {task_name:20s}: QWK={m['qwk']:.4f} | MAE={m['mae']:.4f} | n={m['n']}")

        preds_list = accumulator._preds[task_name]
        trues_list = accumulator._trues[task_name]
        if len(preds_list) > 0:
            p = np.concatenate(preds_list)
            t = np.concatenate(trues_list)
            cm = confusion_matrix(t, p)
            log.info(f"  Confusion Matrix:\n{cm}")

        summary_rows.append({
            'task': task_name,
            'qwk':  m['qwk'],
            'mae':  m['mae'],
            'n':    m['n'],
        })

    # ── Save results CSV ──────────────────────────────────────────────────────
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    results_path = Path(results_dir) / f'blind_test_{args.name}.csv'
    pd.DataFrame(summary_rows).to_csv(results_path, index=False)
    log.info(f"Saved blind-test results to {results_path}")


if __name__ == '__main__':
    main()
