# ─── Backbone Factory ─────────────────────────────────────────────────────────
"""
Centralised backbone loading via timm.

All backbones are returned with their classification head removed
(num_classes=0) and global average pooling applied, producing a flat
feature vector of shape (B, backbone_out_dim).
"""
from __future__ import annotations

import logging
from typing import Tuple

import timm
import torch.nn as nn

log = logging.getLogger('msk')


def build_backbone(name: str = 'efficientnet_b2') -> Tuple[nn.Module, int]:
    """
    Build a pretrained backbone from timm.

    Parameters
    ----------
    name : str
        Any valid timm model name (e.g. 'efficientnet_b2', 'resnet50',
        'densenet121', 'convnext_tiny').

    Returns
    -------
    backbone : nn.Module
        The backbone with classification head removed.
    out_dim : int
        Dimensionality of the output feature vector.
    """
    model = timm.create_model(
        name,
        pretrained=True,
        num_classes=0,       # strip classifier → returns features
        global_pool='avg',   # global average pooling
    )
    out_dim = model.num_features
    log.info(f"Backbone '{name}' loaded (pretrained=True). "
             f"Output dim: {out_dim}")
    return model, out_dim
