from __future__ import annotations

import logging
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.config import ModelConfig
from src.metrics import MetricAccumulator

log = logging.getLogger('msk')


class Trainer:
    """Encapsulates the training and validation loop for a single fold."""
    
    def __init__(self, model: nn.Module, config: ModelConfig, loss_fn: nn.Module, device: torch.device):
        self.model = model
        self.config = config
        self.loss_fn = loss_fn
        self.device = device
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.get_param_groups(self.config.learning_rate, self.config.backbone_lr_mult),
            weight_decay=self.config.weight_decay
        )
        
        # Scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, 
            T_0=self.config.lr_t0, 
            T_mult=self.config.lr_t_mult, 
            eta_min=1e-6
        )
        
        # AMP Scaler
        self.scaler = torch.amp.GradScaler(
            'cuda', 
            enabled=(self.config.use_amp and self.device.type == 'cuda')
        )
        
        # Tracking
        self.best_mean_qwk = float('-inf')
        self.best_val_loss = float('inf')
        self.epochs_no_improve = 0
        
        # Metrics
        task_n_ranks = {t.name: t.n_ranks for t in self.config.tasks}
        self.metric_acc = MetricAccumulator(self.config.task_names, task_n_ranks)

    def train_one_epoch(self, loader: torch.utils.data.DataLoader, epoch: int) -> Tuple[float, Dict[str, float]]:
        self.model.train()
        epoch_loss = 0.0
        per_task_losses_avg = {t: 0.0 for t in self.config.task_names}
        n_batches = 0
        
        pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [TRAIN]", leave=False)
        for batch in pbar:
            # Move to device
            images = batch['image'].to(self.device, non_blocking=True)
            joint_id = batch['joint_id'].to(self.device, non_blocking=True)
            corn_targets = {t: v.to(self.device, non_blocking=True) for t, v in batch['corn_targets'].items()}
            corn_masks = {t: v.to(self.device, non_blocking=True) for t, v in batch['corn_masks'].items()}
            clinical_masks = {t: v.to(self.device, non_blocking=True) for t, v in batch['clinical_masks'].items()}
            
            self.optimizer.zero_grad(set_to_none=True)
            
            # Forward + Loss under AMP
            with torch.amp.autocast(device_type=self.device.type, enabled=(self.config.use_amp and self.device.type == 'cuda')):
                predictions = self.model(images, joint_id)
                loss, per_task_losses = self.loss_fn(predictions, corn_targets, corn_masks, clinical_masks)
            
            # Backward
            if self.config.use_amp and self.device.type == 'cuda':
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                loss.backward()
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                self.optimizer.step()
                
            # Accumulate loss
            epoch_loss += loss.item()
            for t in self.config.task_names:
                per_task_losses_avg[t] += per_task_losses.get(t, torch.tensor(0.0)).item()
            n_batches += 1
            
            pbar.set_postfix(loss=f"{loss.item():.4f}")
            
        n_batches = max(1, n_batches)
        return epoch_loss / n_batches, {t: val / n_batches for t, val in per_task_losses_avg.items()}

    def validate_one_epoch(self, loader: torch.utils.data.DataLoader, epoch: int) -> Tuple[float, Dict[str, dict]]:
        self.model.eval()
        epoch_loss = 0.0
        n_batches = 0
        self.metric_acc.reset()
        
        pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [VAL]  ", leave=False)
        with torch.no_grad():
            for batch in pbar:
                images = batch['image'].to(self.device, non_blocking=True)
                joint_id = batch['joint_id'].to(self.device, non_blocking=True)
                corn_targets = {t: v.to(self.device, non_blocking=True) for t, v in batch['corn_targets'].items()}
                corn_masks = {t: v.to(self.device, non_blocking=True) for t, v in batch['corn_masks'].items()}
                clinical_masks = {t: v.to(self.device, non_blocking=True) for t, v in batch['clinical_masks'].items()}
                
                with torch.amp.autocast(device_type=self.device.type, enabled=(self.config.use_amp and self.device.type == 'cuda')):
                    predictions = self.model(images, joint_id)
                    loss, _ = self.loss_fn(predictions, corn_targets, corn_masks, clinical_masks)
                    
                epoch_loss += loss.item()
                n_batches += 1
                
                # Update metrics
                self.metric_acc.update(predictions, corn_targets, clinical_masks)
                
        n_batches = max(1, n_batches)
        return epoch_loss / n_batches, self.metric_acc.compute()

    def save_checkpoint(self, epoch: int, fold_idx: int, val_loss: float, mean_qwk: float):
        filepath = os.path.join(self.config.checkpoint_dir, f"fold{fold_idx}_best.pth")
        torch.save({
            'epoch': epoch,
            'fold': fold_idx,
            'backbone_name': self.config.backbone_name,
            'model_state': self.model.state_dict(),
            'optim_state': self.optimizer.state_dict(),
            'sched_state': self.scheduler.state_dict(),
            'val_loss': val_loss,
            'mean_qwk': mean_qwk,
        }, filepath)

    @staticmethod
    def load_checkpoint(filepath: str, model: nn.Module, device: torch.device) -> dict:
        ckpt = torch.load(filepath, map_location=device)
        model.load_state_dict(ckpt['model_state'])
        log.info(f"Loaded checkpoint epoch {ckpt.get('epoch')}, val_loss={ckpt.get('val_loss'):.4f}, mean_qwk={ckpt.get('mean_qwk'):.4f}")
        return ckpt

    def run_fold(self, train_loader: torch.utils.data.DataLoader, val_loader: torch.utils.data.DataLoader, fold_idx: int) -> dict:
        log.info("=" * 60)
        log.info(f"FOLD {fold_idx}")
        log.info("=" * 60)
        
        fold_history = {'train_loss': [], 'val_loss': [], 'mean_qwk': [], 'per_task_metrics': []}
        ckpt_path = os.path.join(self.config.checkpoint_dir, f"fold{fold_idx}_best.pth")
        
        for epoch in range(1, self.config.num_epochs + 1):
            # Backbone unfreezing schedule
            if epoch == self.config.freeze_backbone_epochs + 1:
                self.model.unfreeze_backbone()
                
            train_loss, _ = self.train_one_epoch(train_loader, epoch)
            val_loss, val_metrics = self.validate_one_epoch(val_loader, epoch)
            self.scheduler.step()
            
            mean_qwk = self.metric_acc.mean_qwk()
            
            # Logging
            qwk_str = ' | '.join([f"{t}: QWK={m['qwk']:.3f} MAE={m['mae']:.3f}" for t, m in val_metrics.items()])
            log.info(f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Mean QWK: {mean_qwk:.3f}")
            log.info(f"  ↳ {qwk_str}")
            
            fold_history['train_loss'].append(train_loss)
            fold_history['val_loss'].append(val_loss)
            fold_history['mean_qwk'].append(mean_qwk)
            fold_history['per_task_metrics'].append(val_metrics)
            
            if mean_qwk > self.best_mean_qwk:
                self.best_mean_qwk = mean_qwk
                self.best_val_loss = val_loss
                self.epochs_no_improve = 0
                self.save_checkpoint(epoch, fold_idx, val_loss, mean_qwk)
                log.info(f"  New best checkpoint saved (Mean QWK: {mean_qwk:.4f})")
            else:
                self.epochs_no_improve += 1
                
            if self.epochs_no_improve >= self.config.early_stop_patience:
                log.info(f"Early stopping at epoch {epoch}")
                break
                
        return {
            'fold': fold_idx,
            'best_qwk': self.best_mean_qwk,
            'best_val_loss': self.best_val_loss,
            'history': fold_history,
            'ckpt_path': ckpt_path
        }
