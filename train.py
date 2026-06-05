#!/usr/bin/env python3
"""Train the MSK ultrasound model via 5-fold cross-validation.

Usage:
    python train.py --model bmode --name baseline_v1
    python train.py --model doppler --name exp_lr3e4 --backbone efficientnet_b2 --epochs 40
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.config import BmodeConfig, DopplerConfig
from src.dataset import build_fold_loaders
from src.loss import CORNMaskedLoss
from src.model import build_model
from src.trainer import Trainer
from src.utils import get_device, seed_everything, setup_logging


def main():
    parser = argparse.ArgumentParser(description="Train MSK Ultrasound Model")
    parser.add_argument('--model', type=str, choices=['bmode', 'doppler'], required=True)
    parser.add_argument('--name', type=str, required=True,
                        help='Unique name for this run. Artifacts are stored under '
                             'artifacts/models/{model}/{name} and results/{model}/{name}.')
    parser.add_argument('--backbone', type=str, default='efficientnet_b2')
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--folds', type=str, default='0,1,2,3,4', help='Comma-separated fold indices')
    
    args = parser.parse_args()
    
    # Configuration
    overrides = {
        'backbone_name': args.backbone,
        'num_epochs': args.epochs,
        'batch_size': args.batch_size,
        'learning_rate': args.lr,
        'seed': args.seed,
        'checkpoint_dir': f'artifacts/models/{args.model}/{args.name}',
        'results_dir': f'results/{args.model}/{args.name}',
    }
    
    if args.model == 'bmode':
        config = BmodeConfig(**overrides)
    else:
        config = DopplerConfig(**overrides)
        
    seed_everything(config.seed)
    log = setup_logging()
    
    # Directories
    Path(config.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path(config.results_dir).mkdir(parents=True, exist_ok=True)
    
    log.info(f"Configuration loaded for {args.model.upper()} | run name: '{args.name}'")
    log.info(f"  Backbone: {config.backbone_name}")
    log.info(f"  Tasks:    {config.task_names}")
    
    device = get_device()
    log.info(f"Device: {device}")
    
    folds_to_run = [int(f.strip()) for f in args.folds.split(',')]
    cv_results = []
    
    for fold_idx in folds_to_run:
        seed_everything(config.seed + fold_idx)
        
        train_loader, val_loader = build_fold_loaders(config, fold_idx)
        
        model = build_model(config).to(device)
        loss_fn = CORNMaskedLoss(config.tasks).to(device)
        trainer = Trainer(model, config, loss_fn, device)
        
        if config.freeze_backbone_epochs > 0:
            model.freeze_backbone()
            
        result = trainer.run_fold(train_loader, val_loader, fold_idx)
        cv_results.append(result)
        
        del model, trainer, loss_fn, train_loader, val_loader
        if device.type == 'cuda':
            torch.cuda.empty_cache()
            
    # Summary
    log.info("=" * 60)
    log.info("CV SUMMARY")
    log.info("=" * 60)
    
    qwks = [r['best_qwk'] for r in cv_results]
    mean_qwk = np.mean(qwks)
    std_qwk = np.std(qwks)
    log.info(f"Mean QWK across {len(cv_results)} folds: {mean_qwk:.4f} ± {std_qwk:.4f}")
    
    # Save CSV summary
    summary_rows = []
    for r in cv_results:
        row = {'fold': r['fold'], 'best_mean_qwk': r['best_qwk'], 'best_val_loss': r['best_val_loss']}
        # Add per-task metrics from the best epoch (the last one saved in history before early stopping)
        # We can extract it from the history
        best_idx = np.argmax(r['history']['mean_qwk'])
        best_metrics = r['history']['per_task_metrics'][best_idx]
        for t, m in best_metrics.items():
            row[f'{t}_qwk'] = m['qwk']
            row[f'{t}_mae'] = m['mae']
        summary_rows.append(row)
        
    summary_df = pd.DataFrame(summary_rows)
    summary_path = Path(config.results_dir) / 'cv_summary.csv'
    summary_df.to_csv(summary_path, index=False)
    log.info(f"Saved CV summary to {summary_path}")


if __name__ == '__main__':
    main()
