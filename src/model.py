from __future__ import annotations

import logging
from typing import Dict

import torch
import torch.nn as nn

from src.backbone import build_backbone
from src.config import ModelConfig

log = logging.getLogger('msk')


class CORNHead(nn.Module):
    """
    Ordinal task head for Conditional Ordinal Regression (CORN).
    Contains K independent binary classifiers.
    """
    def __init__(self, in_dim: int, n_ranks: int, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.classifiers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(in_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 1),
            )
            for _ in range(n_ranks)
        ])
        
        # Initialize final layer
        for clf in self.classifiers:
            nn.init.normal_(clf[-1].weight, std=0.01)
            nn.init.zeros_(clf[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, in_dim)
        Returns:
            (B, n_ranks) logits
        """
        # Concatenate outputs of all rank classifiers
        return torch.cat([clf(x) for clf in self.classifiers], dim=-1)


class ModalityModel(nn.Module):
    """
    Unified multi-task ordinal regression model for a single modality.
    Uses joint_type embedding to condition the task heads.
    """
    def __init__(self, config: ModelConfig, backbone: nn.Module, backbone_out_dim: int):
        super().__init__()
        self.config = config
        self.backbone = backbone
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(backbone_out_dim, config.feature_dim),
            nn.GroupNorm(32, config.feature_dim),
            nn.GELU(),
            nn.Dropout(config.projection_dropout),
        )
        
        # Joint-type conditioning
        self.joint_embedding = nn.Embedding(config.num_joint_types, config.joint_embed_dim)
        
        # Conditioned dimension: features + joint embedding
        conditioned_dim = config.feature_dim + config.joint_embed_dim
        
        # Task heads
        self.task_heads = nn.ModuleDict({
            task.name: CORNHead(
                in_dim=conditioned_dim,
                n_ranks=task.n_ranks,
                hidden_dim=config.head_hidden_dim,
                dropout=config.head_dropout
            )
            for task in config.tasks
        })
        
        total_params = sum(p.numel() for p in self.parameters())
        backbone_params = sum(p.numel() for p in self.backbone.parameters())
        head_params = total_params - backbone_params
        
        log.info(f"ModalityModel initialized.")
        log.info(f"  Total params:    {total_params / 1e6:.2f}M")
        log.info(f"  Backbone params: {backbone_params / 1e6:.2f}M")
        log.info(f"  Head params:     {head_params / 1e6:.2f}M")

    def forward(self, image: torch.Tensor, joint_id: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            image: (B, 3, H, W)
            joint_id: (B,)
            
        Returns:
            Dict[task_name, logits_tensor(B, K)]
        """
        feats = self.backbone(image)
        feats = self.feature_proj(feats)
        
        joint_emb = self.joint_embedding(joint_id)
        
        conditioned = torch.cat([feats, joint_emb], dim=-1)
        
        return {task_name: head(conditioned) for task_name, head in self.task_heads.items()}

    def freeze_backbone(self) -> None:
        """Freezes backbone parameters."""
        for p in self.backbone.parameters():
            p.requires_grad = False
        log.info("Backbone frozen.")

    def unfreeze_backbone(self) -> None:
        """Unfreezes backbone parameters."""
        for p in self.backbone.parameters():
            p.requires_grad = True
        log.info("Backbone unfrozen.")

    def get_param_groups(self, base_lr: float, backbone_lr_mult: float) -> list:
        """Returns parameter groups with differential learning rates."""
        backbone_ids = set(id(p) for p in self.backbone.parameters())
        backbone_params = [p for p in self.parameters() if id(p) in backbone_ids]
        head_params = [p for p in self.parameters() if id(p) not in backbone_ids]
        
        return [
            {'params': head_params, 'lr': base_lr},
            {'params': backbone_params, 'lr': base_lr * backbone_lr_mult}
        ]


def build_model(config: ModelConfig) -> ModalityModel:
    """Factory function to build the complete model."""
    backbone, out_dim = build_backbone(config.backbone_name, config.backbone_local_path)
    return ModalityModel(config, backbone, out_dim)
