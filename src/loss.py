from __future__ import annotations

import logging
from typing import List, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.config import TaskConfig

log = logging.getLogger('msk')


class CORNMaskedLoss(nn.Module):
    """
    Multi-task masked ordinal BCE loss using CORN (Conditional Ordinal Regression).
    
    For each task, computes element-wise BCE across the conditional ranks,
    masked by both the CORN conditional logic and the clinical NaN mask.
    """
    def __init__(self, task_configs: List[TaskConfig]):
        super().__init__()
        self.task_weights = {t.name: t.loss_weight for t in task_configs}

    def forward(
        self, 
        predictions: Dict[str, torch.Tensor], 
        corn_targets: Dict[str, torch.Tensor], 
        corn_masks: Dict[str, torch.Tensor], 
        clinical_masks: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Args:
            predictions:    Dict[task_name, Tensor(B, K)]  — raw logits
            corn_targets:   Dict[task_name, Tensor(B, K)]  — binary targets
            corn_masks:     Dict[task_name, Tensor(B, K)]  — per-rank CORN masks
            clinical_masks: Dict[task_name, Tensor(B,)]    — NaN masks
            
        Returns:
            total_loss: Scalar tensor
            per_task_losses: Dict of detached loss per task for logging
        """
        total_loss = torch.tensor(0.0, device=next(iter(predictions.values())).device)
        per_task_losses = {}
        
        for task_name, logits in predictions.items():
            target = corn_targets[task_name]
            # Expand clinical_mask: (B,) -> (B, K)
            c_mask = clinical_masks[task_name].unsqueeze(-1)
            
            # Combine CORN condition mask and clinical NaN mask
            full_mask = corn_masks[task_name] * c_mask
            
            n_valid = full_mask.sum()
            if n_valid == 0:
                per_task_losses[task_name] = torch.tensor(0.0, device=logits.device)
                continue
                
            # Element-wise BCE
            bce = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
            
            # Apply mask
            masked_bce = bce * full_mask
            
            # Normalize by valid count
            task_loss = masked_bce.sum() / n_valid.clamp(min=1)
            
            # Apply task weight
            w = self.task_weights.get(task_name, 1.0)
            total_loss = total_loss + w * task_loss
            
            per_task_losses[task_name] = task_loss.detach()
            
        return total_loss, per_task_losses
