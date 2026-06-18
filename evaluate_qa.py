#!/usr/bin/env python3
"""Evaluate the QA Gatekeeper on the Hospital B blind test set.

Checkpoints can be specified in three ways (all equivalent):

  # 1. Explicit list — pass --checkpoints multiple times
  python evaluate_qa.py --name baseline_v1 \\
      --checkpoints artifacts/models/qa/baseline_v1/fold0_best.pth \\
      --checkpoints artifacts/models/qa/baseline_v1/fold1_best.pth

  # 2. Shell glob — let the shell expand it
  python evaluate_qa.py --name baseline_v1 \\
      --checkpoints 'artifacts/models/qa/baseline_v1/fold*_best.pth'

  # 3. Auto-discover (recommended) — omit --checkpoints entirely
  python evaluate_qa.py --name baseline_v1

Predictions from all checkpoints are **averaged in logit space** before
decoding to the predicted class (soft ensemble over folds).
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import QAConfig, QA_CLASS_NAMES, QA_MODALITY_MAP
from src.dataset import QADataset, qa_collate_fn
from src.metrics import QAMetricAccumulator
from src.model import build_qa_model
from src.qa_trainer import QATrainer
from src.utils import get_device, setup_logging


# ─── Helpers ──────────────────────────────────────────────────────────────────

def resolve_checkpoints(checkpoints_arg: list[str] | None, default_dir: str) -> list[Path]:
    """Return a sorted list of QA checkpoint paths.

    Priority:
      1. Explicit paths / shell-expanded globs passed via --checkpoints.
      2. Auto-discover: every fold*_best.pth inside default_dir.
    """
    if checkpoints_arg:
        paths: list[Path] = []
        for entry in checkpoints_arg:
            expanded = (
                sorted(Path('.').glob(entry)) if ('*' in entry or '?' in entry)
                else [Path(entry)]
            )
            paths.extend(expanded)
        if not paths:
            raise FileNotFoundError(f"No checkpoint files matched: {checkpoints_arg}")
        return sorted(set(paths))

    discovered = sorted(Path(default_dir).glob('fold*_best.pth'))
    if not discovered:
        raise FileNotFoundError(
            f"No fold*_best.pth checkpoints found in '{default_dir}'. "
            "Pass --checkpoints explicitly or run train_qa.py first."
        )
    return discovered


def load_qa_models(
    checkpoints: list[Path],
    config: QAConfig,
    device: torch.device,
) -> list[torch.nn.Module]:
    """Build and load one QAModel per checkpoint. All returned in eval mode.

    The backbone name is read from the first checkpoint metadata so the
    correct architecture is reconstructed without extra CLI flags.
    """
    _log = logging.getLogger('msk')
    first_meta = torch.load(str(checkpoints[0]), map_location='cpu', weights_only=False)
    backbone_name = first_meta.get('backbone_name')
    if backbone_name is None:
        _log.warning(
            "QA checkpoint has no 'backbone_name' key. "
            f"Falling back to config default: '{config.backbone_name}'."
        )
        backbone_name = config.backbone_name
    else:
        _log.info(f"QA backbone inferred from checkpoint: '{backbone_name}'")
    config.backbone_name = backbone_name

    models = []
    for ckpt in checkpoints:
        m = build_qa_model(config, pretrained=False).to(device)
        QATrainer.load_checkpoint(str(ckpt), m, device)
        m.eval()
        models.append(m)
    return models


def ensemble_predict_qa(
    models:      list[torch.nn.Module],
    images:      torch.Tensor,
    modality_ids: torch.Tensor,
    config:      QAConfig,
    device:      torch.device,
) -> torch.Tensor:
    """Average logits across all fold models (soft ensemble).

    Returns:
        mean_logits: (B, num_classes)
    """
    use_amp    = config.use_amp and device.type == 'cuda'
    all_logits = []

    for model in models:
        with torch.amp.autocast(device_type=device.type, enabled=use_amp):
            logits = model(images, modality_ids)
        all_logits.append(logits)

    return torch.stack(all_logits).mean(dim=0)


# ─── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate the QA Gatekeeper on Hospital B blind test set",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--name', type=str, required=True,
        help=(
            'Name of the training run to evaluate. '
            'Auto-discover looks for checkpoints under artifacts/models/qa/{name}/.'
        ),
    )
    parser.add_argument(
        '--checkpoints', type=str, nargs='+', default=None, metavar='GLOB_OR_PATH',
        help=(
            'One or more checkpoint paths or glob patterns. '
            'If omitted, all fold*_best.pth under artifacts/models/qa/{name}/ are used.'
        ),
    )
    parser.add_argument('--batch-size', type=int, default=64,
                        help='Batch size for inference (default: 64).')
    args = parser.parse_args()

    log = setup_logging()

    checkpoint_dir = f'artifacts/models/qa/{args.name}'
    results_dir    = f'results/qa/{args.name}'

    config = QAConfig(
        batch_size=args.batch_size,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
    )

    device = get_device()

    log.info(f"QA evaluation | run: '{args.name}'")
    log.info(f"Checkpoints dir : {checkpoint_dir}")
    log.info(f"Results dir     : {results_dir}")

    # ── Resolve checkpoints ───────────────────────────────────────────────────
    try:
        checkpoints = resolve_checkpoints(args.checkpoints, config.checkpoint_dir)
    except FileNotFoundError as exc:
        log.error(str(exc))
        return

    log.warning("=" * 70)
    log.warning(" HOSPITAL B — QA BLIND TEST EVALUATION")
    log.warning(" This script should ONLY be run ONCE after all modeling decisions are final.")
    log.warning("=" * 70)
    log.info(f"Ensemble size: {len(checkpoints)} checkpoint(s)")
    for ckpt in checkpoints:
        log.info(f"  • {ckpt}")

    # ── Identify Hospital B rows ──────────────────────────────────────────────
    df      = pd.read_csv(config.labels_csv)
    df_test = df[~df[config.split_col].isin(range(config.n_folds))]

    if len(df_test) == 0:
        log.error("Could not identify Hospital B holdout set (no rows with split < 0). Aborting.")
        return

    log.info(f"Found {len(df_test)} Hospital B holdout rows (both modalities).")

    # ── Dataset / DataLoader ──────────────────────────────────────────────────
    # QA evaluates on ALL modalities — no modality filter
    test_ds = QADataset(df_test, config.image_dir, config, is_train=False)

    if len(test_ds) == 0:
        log.info("No Hospital B samples available. Done.")
        return

    test_loader = DataLoader(
        test_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        collate_fn=qa_collate_fn,
    )

    # ── Load all fold models ──────────────────────────────────────────────────
    models = load_qa_models(checkpoints, config, device)

    # ── Inference ─────────────────────────────────────────────────────────────
    accumulator = QAMetricAccumulator(
        num_classes=config.num_classes,
        class_names=QA_CLASS_NAMES,
    )

    # Also track per-modality performance
    modality_preds = {0: [], 1: []}
    modality_trues = {0: [], 1: []}

    with torch.no_grad():
        for batch in test_loader:
            images       = batch['image'].to(device, non_blocking=True)
            joint_ids    = batch['joint_id'].to(device, non_blocking=True)
            modality_ids = batch['modality_id'].to(device, non_blocking=True)

            mean_logits = ensemble_predict_qa(models, images, modality_ids, config, device)
            accumulator.update(mean_logits, joint_ids)

            # Collect per-modality preds for split reporting
            preds_np = mean_logits.argmax(dim=-1).cpu().numpy()
            trues_np = joint_ids.cpu().numpy()
            mods_np  = modality_ids.cpu().numpy()
            for mod_id in (0, 1):
                mask = mods_np == mod_id
                if mask.any():
                    modality_preds[mod_id].append(preds_np[mask])
                    modality_trues[mod_id].append(trues_np[mask])

    # ── Report ────────────────────────────────────────────────────────────────
    metrics = accumulator.compute()

    log.info("=" * 70)
    log.info(f"HOSPITAL B — QA RESULTS | run: '{args.name}' — ensemble of {len(models)} model(s)")
    log.info("=" * 70)
    log.info(f"  Overall Accuracy  : {metrics['accuracy']:.4f}")
    log.info(f"  Macro F1          : {metrics['macro_f1']:.4f}")
    log.info(f"  Cohen's Kappa     : {metrics['kappa']:.4f}")
    log.info(f"  N samples         : {metrics['n']}")

    log.info("\nPer-class accuracy:")
    for name, acc in metrics['per_class_acc'].items():
        acc_str = f"{acc:.2%}" if not np.isnan(acc) else "n/a"
        log.info(f"    {name:30s}: {acc_str}")

    log.info("\nConfusion matrix (rows=true, cols=predicted):")
    log.info(f"  Classes: {QA_CLASS_NAMES}")
    if metrics['confusion_matrix'] is not None:
        log.info(f"\n{metrics['confusion_matrix']}")

    # Per-modality breakdown
    modality_names = {v: k for k, v in QA_MODALITY_MAP.items()}
    for mod_id in (0, 1):
        if modality_preds[mod_id]:
            p = np.concatenate(modality_preds[mod_id])
            t = np.concatenate(modality_trues[mod_id])
            acc = float(np.mean(p == t))
            log.info(f"\n  {modality_names[mod_id]} accuracy: {acc:.2%} (n={len(p)})")

    # ── Save results ──────────────────────────────────────────────────────────
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Summary CSV
    summary_rows = [
        {'metric': 'accuracy',  'value': metrics['accuracy']},
        {'metric': 'macro_f1',  'value': metrics['macro_f1']},
        {'metric': 'kappa',     'value': metrics['kappa']},
        {'metric': 'n',         'value': metrics['n']},
    ]
    for name, acc in metrics['per_class_acc'].items():
        summary_rows.append({'metric': f'acc_{name}', 'value': acc})

    results_path = Path(results_dir) / f'blind_test_{args.name}.csv'
    pd.DataFrame(summary_rows).to_csv(results_path, index=False)
    log.info(f"\nSaved blind-test results → {results_path}")

    # Confusion matrix CSV
    if metrics['confusion_matrix'] is not None:
        cm_df = pd.DataFrame(
            metrics['confusion_matrix'],
            index=QA_CLASS_NAMES,
            columns=QA_CLASS_NAMES,
        )
        cm_path = Path(results_dir) / f'confusion_matrix_{args.name}.csv'
        cm_df.to_csv(cm_path)
        log.info(f"Saved confusion matrix → {cm_path}")


if __name__ == '__main__':
    main()
