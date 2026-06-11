# ─── Backbone Factory ─────────────────────────────────────────────────────────
"""
Centralised backbone loading via timm.

All backbones are returned with their classification head removed
(num_classes=0) and global average pooling applied, producing a flat
feature vector of shape (B, backbone_out_dim).

Two loading modes are supported:
  • Online  (default): downloads pretrained weights from timm/HuggingFace.
  • Offline (--local): builds the architecture with timm (pretrained=False)
    and loads weights from a local file inside the backbones/ folder.
    Supported formats:
      - PyTorch checkpoint : .pth / .pt
      - SafeTensors        : .safetensors
    This is the required mode for air-gapped hospital deployments.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple

import timm
import torch
import torch.nn as nn

log = logging.getLogger('msk')

# Recognised file extensions and the loader to use
_PYTORCH_EXTENSIONS    = {'.pth', '.pt'}
_SAFETENSORS_EXTENSION = '.safetensors'


def _load_state_dict(path: Path) -> dict:
    """Load a state-dict from a .pth/.pt or .safetensors file."""
    suffix = path.suffix.lower()

    if suffix in _PYTORCH_EXTENSIONS:
        state_dict = torch.load(path, map_location='cpu', weights_only=True)
        # Support both raw state-dicts and checkpoint dicts with a nested key
        if isinstance(state_dict, dict) and 'state_dict' in state_dict:
            state_dict = state_dict['state_dict']
        return state_dict

    if suffix == _SAFETENSORS_EXTENSION:
        from safetensors.torch import load_file  # lazy import — optional dependency
        return load_file(str(path), device='cpu')

    raise ValueError(
        f"Unsupported backbone file format: '{suffix}'. "
        f"Accepted extensions: {sorted(_PYTORCH_EXTENSIONS | {_SAFETENSORS_EXTENSION})}"
    )


def build_backbone(
    name: str = 'efficientnet_b2',
    local_path: Optional[str] = None,
    pretrained: bool = True,
) -> Tuple[nn.Module, int]:
    """
    Build a backbone, either from timm (online) or from a local file (offline).

    Parameters
    ----------
    name : str
        Any valid timm model name (e.g. 'efficientnet_b2', 'resnet50',
        'densenet121', 'convnext_tiny').
    local_path : str or None
        Path to a local checkpoint inside the backbones/ folder.
        Supported formats: .pth, .pt, .safetensors.
        When provided, timm builds the architecture without downloading
        weights and the state-dict is loaded from this file instead.
        Use this for air-gapped environments (e.g. hospital computers).
    pretrained : bool
        Only relevant when local_path is None.
        When False, timm builds the architecture without downloading any
        pretrained weights — use this when a full model checkpoint will be
        loaded immediately afterwards (e.g. in evaluate.py), so the
        pretrained download is a complete waste.

    Returns
    -------
    backbone : nn.Module
        The backbone with classification head removed.
    out_dim : int
        Dimensionality of the output feature vector.
    """
    if local_path is not None:
        path = Path(local_path)
        if not path.exists():
            raise FileNotFoundError(
                f"Local backbone file not found: '{path}'. "
                "Make sure the file is placed inside the backbones/ folder "
                "and the --local path is correct. "
                f"Supported formats: .pth, .pt, .safetensors"
            )

        # Build architecture without downloading weights
        model = timm.create_model(
            name,
            pretrained=False,
            num_classes=0,
            global_pool='avg',
        )

        state_dict = _load_state_dict(path)
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        if missing:
            log.warning(f"  Missing keys when loading local backbone: {missing}")
        if unexpected:
            log.warning(f"  Unexpected keys when loading local backbone: {unexpected}")

        out_dim = model.num_features
        log.info(f"Backbone '{name}' loaded from local file '{path}' "
                 f"(format: {path.suffix}). Output dim: {out_dim}")

    else:
        model = timm.create_model(
            name,
            pretrained=pretrained,
            num_classes=0,       # strip classifier → returns features
            global_pool='avg',   # global average pooling
        )
        out_dim = model.num_features
        if pretrained:
            log.info(f"Backbone '{name}' loaded from timm (pretrained=True). "
                     f"Output dim: {out_dim}")
        else:
            log.info(f"Backbone '{name}' architecture built without pretrained weights. "
                     f"Output dim: {out_dim}")

    return model, out_dim
