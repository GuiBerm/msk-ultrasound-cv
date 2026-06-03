#!/usr/bin/env python3
"""Evaluate the trained model on the Hospital B blind test set.

Usage:
    python evaluate.py --model bmode --checkpoint artifacts/models/bmode/fold0_best.pth
"""
from __future__ import annotations

import argparse
import logging

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


def main():
    parser = argparse.ArgumentParser(description="Evaluate on Hospital B blind test set")
    parser.add_argument('--model', type=str, choices=['bmode', 'doppler'], required=True)
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--batch-size', type=int, default=32)
    args = parser.parse_args()

    if args.model == 'bmode':
        config = BmodeConfig(batch_size=args.batch_size)
    else:
        config = DopplerConfig(batch_size=args.batch_size)

    log = setup_logging()
    device = get_device()

    log.warning("=" * 70)
    log.warning(" HOSPITAL B — BLIND TEST EVALUATION")
    log.warning(" This script should ONLY be run ONCE after all modeling decisions are final.")
    log.warning("=" * 70)

    # Identify Hospital B rows
    df = pd.read_csv(config.labels_csv)
    # Hospital B test set is assigned split < 0 or not in folds 0-4
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

    # Dataset automatically filters to correct modality
    test_ds = MSKUltrasoundDataset(df_test, config.image_dir, config, is_train=False)
    
    if len(test_ds) == 0:
        log.info(f"No Hospital B samples for modality {config.modality_filter}. Done.")
        return
        
    test_loader = DataLoader(
        test_ds, 
        batch_size=config.batch_size, 
        shuffle=False,
        num_workers=config.num_workers, 
        collate_fn=msk_collate_fn
    )

    model = build_model(config).to(device)
    Trainer.load_checkpoint(args.checkpoint, model, device)
    model.eval()

    task_n_ranks = {t.name: t.n_ranks for t in config.tasks}
    accumulator = MetricAccumulator(config.task_names, task_n_ranks)

    with torch.no_grad():
        for batch in test_loader:
            images = batch['image'].to(device, non_blocking=True)
            joint_id = batch['joint_id'].to(device, non_blocking=True)
            corn_targets = {t: v.to(device, non_blocking=True) for t, v in batch['corn_targets'].items()}
            clinical_masks = {t: v.to(device, non_blocking=True) for t, v in batch['clinical_masks'].items()}
            
            with torch.amp.autocast(device_type=device.type, enabled=(config.use_amp and device.type == 'cuda')):
                predictions = model(images, joint_id)
                
            accumulator.update(predictions, corn_targets, clinical_masks)

    metrics = accumulator.compute()
    
    log.info("=" * 70)
    log.info(f"HOSPITAL B RESULTS ({args.model.upper()})")
    log.info("=" * 70)
    
    for task_name, m in metrics.items():
        log.info(f"  {task_name:20s}: QWK={m['qwk']:.4f} | MAE={m['mae']:.4f} | n={m['n']}")
        
        # Optionally print confusion matrix
        preds = accumulator._preds[task_name]
        trues = accumulator._trues[task_name]
        if len(preds) > 0:
            import numpy as np
            p = np.concatenate(preds)
            t = np.concatenate(trues)
            cm = confusion_matrix(t, p)
            log.info(f"  Confusion Matrix:\n{cm}")


if __name__ == '__main__':
    main()
