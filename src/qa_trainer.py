from __future__ import annotations

import logging
import os
from typing import Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm.auto import tqdm

from src.config import QAConfig, QA_CLASS_NAMES
from src.metrics import QAMetricAccumulator

log = logging.getLogger('msk')


class QATrainer:
    """
    Encapsulates the training and validation loop for a single QA fold.

    Differences from the scoring-model Trainer:
      - Loss: CrossEntropyLoss (flat multi-class, not CORN ordinal)
      - No masked loss or clinical_masks — every sample has a valid joint_type
      - Early stopping / best-checkpoint criterion: macro F1 (primary)
      - batch keys: 'image', 'joint_id', 'modality_id' (no corn_targets)

    The API (train_one_epoch / validate_one_epoch / run_fold / save/load
    checkpoint) mirrors Trainer so train_qa.py looks identical to train.py.
    """

    def __init__(self, model: nn.Module, config: QAConfig, device: torch.device):
        self.model  = model
        self.config = config
        self.device = device

        # ── Optimizer ─────────────────────────────────────────────────────────
        self.optimizer = torch.optim.AdamW(
            self.model.get_param_groups(self.config.learning_rate, self.config.backbone_lr_mult),
            weight_decay=self.config.weight_decay,
        )

        # ── Scheduler ─────────────────────────────────────────────────────────
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer,
            T_0=self.config.lr_t0,
            T_mult=self.config.lr_t_mult,
            eta_min=1e-6,
        )

        # ── AMP Scaler ────────────────────────────────────────────────────────
        self.scaler = torch.amp.GradScaler(
            'cuda',
            enabled=(self.config.use_amp and self.device.type == 'cuda'),
        )

        # ── Loss ──────────────────────────────────────────────────────────────
        # CrossEntropyLoss with no class weights — the WeightedRandomSampler
        # already balances the class distribution at the batch level.
        self.loss_fn = nn.CrossEntropyLoss()

        # ── Tracking ──────────────────────────────────────────────────────────
        self.best_macro_f1     = float('-inf')
        self.best_val_loss     = float('inf')
        self.epochs_no_improve = 0

        # ── Metrics ───────────────────────────────────────────────────────────
        self.metric_acc = QAMetricAccumulator(
            num_classes=self.config.num_classes,
            class_names=QA_CLASS_NAMES,
        )

    # ── Training loop ─────────────────────────────────────────────────────────

    def train_one_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        epoch: int,
    ) -> float:
        """Run one training epoch. Returns mean cross-entropy loss."""
        self.model.train()
        epoch_loss = 0.0
        n_batches  = 0

        accum_steps = max(1, self.config.accum_steps)
        use_amp = self.config.use_amp and self.device.type == 'cuda'

        pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [TRAIN]", leave=False)
        for step_idx, batch in enumerate(pbar):
            # Zero gradients at the start of each accumulation window
            if step_idx % accum_steps == 0:
                self.optimizer.zero_grad(set_to_none=True)

            images       = batch['image'].to(self.device, non_blocking=True)
            joint_ids    = batch['joint_id'].to(self.device, non_blocking=True)
            modality_ids = batch['modality_id'].to(self.device, non_blocking=True)

            with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                logits = self.model(images, modality_ids)
                loss   = self.loss_fn(logits, joint_ids)

            # Scale loss for accumulation so gradients average (not sum) over the window
            accum_loss = loss / accum_steps

            if use_amp:
                self.scaler.scale(accum_loss).backward()
            else:
                accum_loss.backward()

            # Optimizer step only at the end of an accumulation window (or last batch)
            is_last_batch = (step_idx + 1 == len(loader))
            if (step_idx + 1) % accum_steps == 0 or is_last_batch:
                if use_amp:
                    self.scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)
                    self.optimizer.step()

            # Accumulate the unscaled loss for logging (keeps values comparable across runs)
            epoch_loss += loss.item()
            n_batches  += 1
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        return epoch_loss / max(1, n_batches)

    # ── Validation loop ───────────────────────────────────────────────────────

    def validate_one_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        epoch: int,
    ) -> Tuple[float, Dict]:
        """Run one validation epoch. Returns (mean_loss, metrics_dict)."""
        self.model.eval()
        epoch_loss = 0.0
        n_batches  = 0
        self.metric_acc.reset()

        use_amp = self.config.use_amp and self.device.type == 'cuda'
        pbar = tqdm(loader, desc=f"Epoch {epoch:03d} [VAL]  ", leave=False)
        with torch.no_grad():
            for batch in pbar:
                images       = batch['image'].to(self.device, non_blocking=True)
                joint_ids    = batch['joint_id'].to(self.device, non_blocking=True)
                modality_ids = batch['modality_id'].to(self.device, non_blocking=True)

                with torch.amp.autocast(device_type=self.device.type, enabled=use_amp):
                    logits = self.model(images, modality_ids)
                    loss   = self.loss_fn(logits, joint_ids)

                epoch_loss += loss.item()
                n_batches  += 1
                self.metric_acc.update(logits, joint_ids)

        return epoch_loss / max(1, n_batches), self.metric_acc.compute()

    # ── Checkpoint helpers ────────────────────────────────────────────────────

    def save_checkpoint(self, epoch: int, fold_idx: int, val_loss: float, macro_f1: float) -> None:
        filepath = os.path.join(self.config.checkpoint_dir, f"fold{fold_idx}_best.pth")
        torch.save({
            'epoch':         epoch,
            'fold':          fold_idx,
            'backbone_name': self.config.backbone_name,
            'model_state':   self.model.state_dict(),
            'optim_state':   self.optimizer.state_dict(),
            'sched_state':   self.scheduler.state_dict(),
            'val_loss':      val_loss,
            'macro_f1':      macro_f1,
        }, filepath)

    @staticmethod
    def load_checkpoint(filepath: str, model: nn.Module, device: torch.device) -> dict:
        ckpt = torch.load(filepath, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state'])
        log.info(
            f"Loaded QA checkpoint epoch {ckpt.get('epoch')}, "
            f"val_loss={ckpt.get('val_loss', float('nan')):.4f}, "
            f"macro_f1={ckpt.get('macro_f1', float('nan')):.4f}"
        )
        return ckpt

    # ── Full fold run ─────────────────────────────────────────────────────────

    def run_fold(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader:   torch.utils.data.DataLoader,
        fold_idx:     int,
    ) -> dict:
        """
        Run all epochs for a single fold.

        Mirrors Trainer.run_fold: handles backbone freeze/unfreeze schedule,
        logs per-epoch metrics, saves the best checkpoint, and returns a
        history dict for CSV persistence.
        """
        log.info("=" * 60)
        log.info(f"QA FOLD {fold_idx}")
        log.info("=" * 60)

        fold_history = {'train_loss': [], 'val_loss': [], 'macro_f1': [], 'accuracy': [], 'kappa': []}
        ckpt_path    = os.path.join(self.config.checkpoint_dir, f"fold{fold_idx}_best.pth")

        for epoch in range(1, self.config.num_epochs + 1):
            # Backbone unfreeze after warm-up
            if epoch == self.config.freeze_backbone_epochs + 1:
                self.model.unfreeze_backbone()

            train_loss               = self.train_one_epoch(train_loader, epoch)
            val_loss, val_metrics    = self.validate_one_epoch(val_loader, epoch)
            self.scheduler.step()

            macro_f1 = val_metrics['macro_f1']
            accuracy  = val_metrics['accuracy']
            kappa     = val_metrics['kappa']

            # Per-class accuracy string for log
            pca_str = ' | '.join(
                f"{name}: {acc:.2%}" if not np.isnan(acc) else f"{name}: n/a"
                for name, acc in val_metrics['per_class_acc'].items()
            )
            log.info(
                f"Epoch {epoch:03d} | Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_loss:.4f} | Acc: {accuracy:.4f} | "
                f"Macro F1: {macro_f1:.4f} | Kappa: {kappa:.4f}"
            )
            log.info(f"  ↳ Per-class acc: {pca_str}")

            fold_history['train_loss'].append(train_loss)
            fold_history['val_loss'].append(val_loss)
            fold_history['macro_f1'].append(macro_f1)
            fold_history['accuracy'].append(accuracy)
            fold_history['kappa'].append(kappa)

            for name, acc in val_metrics['per_class_acc'].items():
                key = f'acc_{name}'
                if key not in fold_history:
                    fold_history[key] = []
                fold_history[key].append(acc)

            if macro_f1 > self.best_macro_f1:
                self.best_macro_f1     = macro_f1
                self.best_val_loss     = val_loss
                self.epochs_no_improve = 0
                self.save_checkpoint(epoch, fold_idx, val_loss, macro_f1)
                log.info(f"  ✓ New best QA checkpoint saved (Macro F1: {macro_f1:.4f})")
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.config.early_stop_patience:
                log.info(f"Early stopping at epoch {epoch} (no macro F1 improvement for {self.config.early_stop_patience} epochs)")
                break

        return {
            'fold':          fold_idx,
            'best_macro_f1': self.best_macro_f1,
            'best_val_loss': self.best_val_loss,
            'history':       fold_history,
            'ckpt_path':     ckpt_path,
        }
