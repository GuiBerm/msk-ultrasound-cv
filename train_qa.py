#!/usr/bin/env python3
"""Train the QA Gatekeeper joint-type classifier via 5-fold cross-validation.

The QA model is a 5-class image classifier that predicts joint anatomy
(MCF / IFP / MTF / Radiocubital distal / Wrist) purely from image geometry,
acting as a sanity-check on clinician uploads before they reach the scoring
pipeline.  Radiocarpiana and Intercarpiana are merged into one class because
they share the identical eco_id image and cannot be distinguished visually.

All images (B-Mode and Doppler) are fed to a single model after grayscale
conversion, stripping acquisition-mode colour cues.  A small learned
modality embedding corrects for any residual luminance differences between
the two acquisition modes.

Usage:
    # Online: download pretrained backbone from timm/HuggingFace
    python train_qa.py --name baseline_v1

    # Offline: load backbone weights from a local file (air-gapped environments)
    python train_qa.py --name hospital_run --local backbones/efficientnet_b2.pth

    # Override hyperparameters
    python train_qa.py --name exp_convnext --backbone convnext_small --epochs 80 --lr 1e-4

    # Run a specific subset of folds (e.g. quick smoke-test on fold 0 only)
    python train_qa.py --name smoke --epochs 5 --folds 0
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import QAConfig
from src.dataset import build_qa_fold_loaders
from src.model import build_qa_model
from src.qa_trainer import QATrainer
from src.utils import get_device, seed_everything, setup_logging


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train the QA Gatekeeper joint-type classifier",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--name', type=str, required=True,
        help=(
            'Unique name for this run. Artifacts are stored under '
            'artifacts/models/qa/{name} and results/qa/{name}.'
        ),
    )
    parser.add_argument('--backbone', type=str, default='efficientnet_b2',
                        help='timm backbone name (default: efficientnet_b2)')
    parser.add_argument('--epochs',     type=int,   default=60)
    parser.add_argument('--batch-size', type=int,   default=32)
    parser.add_argument('--lr',         type=float, default=3e-4)
    parser.add_argument(
        '--freeze-epochs', type=int, default=5,
        help=(
            'Epochs to keep the backbone frozen while only the head warms up. '
            '0 = never freeze. Default: 5.'
        ),
    )
    parser.add_argument(
        '--backbone-lr-mult', type=float, default=0.05,
        help='Learning-rate multiplier for the backbone. Default: 0.05.',
    )
    parser.add_argument(
        '--no-clahe', action='store_true', default=False,
        help=(
            'Disable CLAHE contrast normalisation in training augmentation. '
            'By default CLAHE is active to normalise scanner gain differences.'
        ),
    )
    parser.add_argument('--seed',  type=int, default=42)
    parser.add_argument(
        '--folds', type=str, default='0,1,2,3,4',
        help='Comma-separated fold indices to run (default: all 5 folds).',
    )
    parser.add_argument(
        '--local', type=str, default=None, metavar='PATH',
        help=(
            'Path to a local backbone file inside the backbones/ folder. '
            'Supported: .pth, .pt, .safetensors. '
            'Use in air-gapped environments where internet is unavailable.'
        ),
    )
    args = parser.parse_args()

    log = setup_logging()
    seed_everything(args.seed)

    config = QAConfig(
        backbone_name=args.backbone,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        backbone_lr_mult=args.backbone_lr_mult,
        freeze_backbone_epochs=args.freeze_epochs,
        use_clahe=not args.no_clahe,
        seed=args.seed,
        checkpoint_dir=f'artifacts/models/qa/{args.name}',
        results_dir=f'results/qa/{args.name}',
        backbone_local_path=args.local,
    )

    seed_everything(config.seed)

    # ── Directories ───────────────────────────────────────────────────────────
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)

    log.info(f"QA Gatekeeper training | run: '{args.name}'")
    log.info(f"  Backbone:           {config.backbone_name}")
    log.info(f"  Num classes:        {config.num_classes}  (5 merged joint types)")
    log.info(f"  backbone_lr_mult:   {config.backbone_lr_mult}")
    log.info(f"  freeze_epochs:      {config.freeze_backbone_epochs}")
    log.info(f"  use_clahe:          {config.use_clahe}")

    device = get_device()
    log.info(f"Device: {device}")

    folds_to_run = [int(f.strip()) for f in args.folds.split(',')]
    cv_results   = []

    for fold_idx in folds_to_run:
        seed_everything(config.seed + fold_idx)

        train_loader, val_loader = build_qa_fold_loaders(config, fold_idx)

        model   = build_qa_model(config).to(device)
        trainer = QATrainer(model, config, device)

        if config.freeze_backbone_epochs > 0:
            model.freeze_backbone()

        result = trainer.run_fold(train_loader, val_loader, fold_idx)
        cv_results.append(result)

        # ── Save epoch history ─────────────────────────────────────────────
        history_df   = pd.DataFrame(result['history'])
        history_path = Path(config.results_dir) / f'fold{fold_idx}_history.csv'
        history_df.to_csv(history_path, index=False)
        log.info(f"  Saved fold history → {history_path}")

        del model, trainer, train_loader, val_loader
        if device.type == 'cuda':
            torch.cuda.empty_cache()

    # ── CV summary ────────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("QA CV SUMMARY")
    log.info("=" * 60)

    macro_f1s = [r['best_macro_f1'] for r in cv_results]
    mean_f1   = float(np.mean(macro_f1s))
    std_f1    = float(np.std(macro_f1s))
    log.info(f"Mean Macro F1 across {len(cv_results)} fold(s): {mean_f1:.4f} ± {std_f1:.4f}")

    summary_rows = []
    for r in cv_results:
        best_idx  = int(np.argmax(r['history']['macro_f1']))
        row = {
            'fold':          r['fold'],
            'best_macro_f1': r['best_macro_f1'],
            'best_val_loss': r['best_val_loss'],
            'best_accuracy': r['history']['accuracy'][best_idx],
            'best_kappa':    r['history']['kappa'][best_idx],
        }
        summary_rows.append(row)

    summary_df   = pd.DataFrame(summary_rows)
    summary_path = Path(config.results_dir) / 'cv_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    log.info(f"Saved CV summary → {summary_path}")


if __name__ == '__main__':
    main()
