# ─── Configuration ────────────────────────────────────────────────────────────
"""
Central configuration for the MSK Ultrasound ML pipeline.

Two factory functions produce the model-specific configs:
  - BmodeConfig()  → structural tasks (eg_sinovial, bone_erosion)
  - DopplerConfig() → vascular tasks  (pd_sinovial)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


# ─── Joint-Type Encoding ─────────────────────────────────────────────────────
JOINT_TYPE_MAP = {
    'MCF':                 0,
    'IFP':                 1,
    'MTF':                 2,
    'Radiocubital distal': 3,
    'Radiocarpiana':       4,
    'Intercarpiana':       5,
}

NUM_JOINT_TYPES = len(JOINT_TYPE_MAP)


# ─── Task Definition ─────────────────────────────────────────────────────────
@dataclass
class TaskConfig:
    """Describes a single ordinal regression task."""
    name:        str            # Internal task name (e.g. 'eg_sinovial')
    csv_column:  str            # Column name in labels_with_splits.csv
    n_ranks:     int            # Number of CORN ranks (= max_score for that task)
    loss_weight: float = 1.0    # Relative weight in multi-task loss


# ─── Model Configuration ─────────────────────────────────────────────────────
@dataclass
class ModelConfig:
    """Single source of truth for every hyperparameter."""

    # ── Identity ──────────────────────────────────────────────────────────────
    modality:        str = 'bmode'        # 'bmode' or 'doppler'
    modality_filter: str = 'Modo B'       # Value to filter tipo_imagen column

    # ── Tasks ─────────────────────────────────────────────────────────────────
    tasks: List[TaskConfig] = field(default_factory=list)

    # ── Paths ─────────────────────────────────────────────────────────────────
    labels_csv:     str = 'artifacts/labels_with_splits.csv'
    image_dir:      str = 'data/cropped_images'
    checkpoint_dir: str = 'artifacts/models'
    results_dir:    str = 'results'

    # ── Backbone ──────────────────────────────────────────────────────────────
    backbone_name:         str = 'efficientnet_b2'
    freeze_backbone_epochs: int = 5

    # ── Architecture ──────────────────────────────────────────────────────────
    feature_dim:        int   = 512
    joint_embed_dim:    int   = 32
    num_joint_types:    int   = NUM_JOINT_TYPES
    head_hidden_dim:    int   = 128
    projection_dropout: float = 0.40
    head_dropout:       float = 0.30

    # ── Image ─────────────────────────────────────────────────────────────────
    image_size: int = 256

    # ── Training ──────────────────────────────────────────────────────────────
    batch_size:       int   = 32
    num_epochs:       int   = 60
    learning_rate:    float = 3e-4
    backbone_lr_mult: float = 0.05
    weight_decay:     float = 1e-4
    num_workers:      int   = 4
    pin_memory:       bool  = True
    use_amp:          bool  = True
    grad_clip_norm:   float = 1.0

    # ── Scheduler ─────────────────────────────────────────────────────────────
    lr_t0:     int = 20
    lr_t_mult: int = 2

    # ── Early Stopping ────────────────────────────────────────────────────────
    early_stop_patience: int = 15

    # ── Cross-Validation ──────────────────────────────────────────────────────
    n_folds: int = 5
    seed:    int = 42

    # ── DataFrame Column Names ────────────────────────────────────────────────
    eco_id_col:     str = 'eco_id'
    joint_type_col: str = 'joint_type'
    modality_col:   str = 'tipo_imagen'
    split_col:      str = 'split'

    @property
    def task_names(self) -> List[str]:
        return [t.name for t in self.tasks]

    @property
    def task_weights(self) -> dict:
        return {t.name: t.loss_weight for t in self.tasks}


# ─── Factory Functions ────────────────────────────────────────────────────────

def BmodeConfig(**overrides) -> ModelConfig:
    """B-Mode structural model: eg_sinovial (K=3) + bone_erosion (K=1)."""
    defaults = dict(
        modality='bmode',
        modality_filter='Modo B',
        tasks=[
            TaskConfig(name='eg_sinovial',  csv_column='eg_sinovial',  n_ranks=3, loss_weight=1.0),
            TaskConfig(name='bone_erosion', csv_column='bone_erosion', n_ranks=1, loss_weight=1.5),
        ],
        checkpoint_dir='artifacts/models/bmode',
        results_dir='results/bmode',
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)


def DopplerConfig(**overrides) -> ModelConfig:
    """Doppler vascular model: pd_sinovial (K=3)."""
    defaults = dict(
        modality='doppler',
        modality_filter='Power Doppler',
        tasks=[
            TaskConfig(name='pd_sinovial', csv_column='pd_sinovial', n_ranks=3, loss_weight=1.0),
        ],
        checkpoint_dir='artifacts/models/doppler',
        results_dir='results/doppler',
    )
    defaults.update(overrides)
    return ModelConfig(**defaults)
