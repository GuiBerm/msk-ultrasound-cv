# ─── Configuration ────────────────────────────────────────────────────────────
"""
Central configuration for the MSK Ultrasound ML pipeline.

Two factory functions produce the model-specific configs:
  - BmodeConfig()   → structural tasks (eg_sinovial, bone_erosion)
  - DopplerConfig() → vascular tasks  (pd_sinovial)

A separate QAConfig dataclass drives the QA Gatekeeper classifier:
  - QAConfig        → joint-type classification (5 classes, wrist merged)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional


# ─── Joint-Type Encoding (scoring models) ────────────────────────────────────
JOINT_TYPE_MAP = {
    'MCF':                 0,
    'IFP':                 1,
    'MTF':                 2,
    'Radiocubital distal': 3,
    'Radiocarpiana':       4,
    'Intercarpiana':       5,
}

NUM_JOINT_TYPES = len(JOINT_TYPE_MAP)


# ─── QA Joint-Type Encoding (5 classes — wrist sub-joints merged) ─────────────
# Radiocarpiana and Intercarpiana share the exact same ultrasound image, so
# the QA model cannot distinguish them from image geometry alone.  Both are
# collapsed into a single "Wrist (Radio+Inter)" class (index 4).
QA_JOINT_TYPE_MAP = {
    'MCF':                 0,
    'IFP':                 1,
    'MTF':                 2,
    'Radiocubital distal': 3,
    'Radiocarpiana':       4,   # ─┐ merged
    'Intercarpiana':       4,   # ─┘ merged → "Wrist (Radio+Inter)"
}
QA_CLASS_NAMES: List[str] = [
    'MCF', 'IFP', 'MTF', 'Radiocubital distal', 'Wrist (Radio+Inter)'
]
NUM_QA_CLASSES = len(QA_CLASS_NAMES)  # 5

# ─── QA Modality Encoding ─────────────────────────────────────────────────────
# A small learned embedding corrects for residual luminance differences between
# B-Mode and Doppler acquisitions after grayscale conversion.
QA_MODALITY_MAP = {
    'Modo B':        0,
    'Power Doppler': 1,
}
NUM_QA_MODALITIES = len(QA_MODALITY_MAP)  # 2


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

    # ── Paths ───────────────────────────────────────────────────────────────────────
    labels_csv:          str = 'artifacts/labels_with_splits.csv'
    image_dir:           str = 'data/cropped_images'
    checkpoint_dir:      str = 'artifacts/models'
    results_dir:         str = 'results'
    backbone_local_path: Optional[str] = None   # set via --local for offline use

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

    # ── Augmentation ──────────────────────────────────────────────────────────
    color_augmentation: bool = False   # activate domain-robustness colour block

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

def BmodeConfig(include_bone_erosion: bool = True, **overrides) -> ModelConfig:
    """
    B-Mode structural model.

    Tasks:
      - eg_sinovial  (K=3, always included)
      - bone_erosion (K=1, included by default; set include_bone_erosion=False to drop it)

    Parameters
    ----------
    include_bone_erosion : bool
        When False, the bone_erosion task head is omitted entirely from the
        model, loss, and metrics. Useful when the severe class imbalance
        (≈90 % negatives) is suspected to hurt eg_sinovial optimisation.
    """
    tasks = [
        TaskConfig(name='eg_sinovial', csv_column='eg_sinovial', n_ranks=3, loss_weight=1.0),
    ]
    if include_bone_erosion:
        tasks.append(
            TaskConfig(name='bone_erosion', csv_column='bone_erosion', n_ranks=1, loss_weight=1.5)
        )

    defaults = dict(
        modality='bmode',
        modality_filter='Modo B',
        tasks=tasks,
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


# ─── QA Gatekeeper Configuration ─────────────────────────────────────────────

@dataclass
class QAConfig:
    """
    Configuration for the QA Gatekeeper joint-type classifier.

    Trains a single model on ALL rows (both B-Mode and Doppler), converting
    every image to grayscale to strip modality colour cues and expose pure
    joint geometry.  A small modality embedding lets the network correct for
    any residual luminance differences between the two acquisition modes.

    Wrist note:
      Radiocarpiana and Intercarpiana are merged into class 4 ("Wrist
      Radio+Inter") because they share the identical eco_id image and the
      geometry is indistinguishable.  This gives 5 output classes total.
    """

    # ── Classes ───────────────────────────────────────────────────────────────
    num_classes:    int = NUM_QA_CLASSES      # 5 merged joint classes
    num_modalities: int = NUM_QA_MODALITIES   # 2 (bmode / doppler)

    # ── Paths ─────────────────────────────────────────────────────────────────
    labels_csv:          str = 'artifacts/labels_with_splits.csv'
    image_dir:           str = 'data/cropped_images'
    checkpoint_dir:      str = 'artifacts/models/qa'
    results_dir:         str = 'results/qa'
    backbone_local_path: Optional[str] = None

    # ── Backbone ──────────────────────────────────────────────────────────────
    backbone_name:          str = 'efficientnet_b2'
    freeze_backbone_epochs: int = 5

    # ── Architecture ──────────────────────────────────────────────────────────
    feature_dim:        int   = 512
    modality_embed_dim: int   = 16     # small embedding for bmode/doppler bias
    head_hidden_dim:    int   = 128
    projection_dropout: float = 0.40
    head_dropout:       float = 0.30

    # ── Image ─────────────────────────────────────────────────────────────────
    image_size: int = 256

    # ── Augmentation ──────────────────────────────────────────────────────────
    use_clahe: bool = True   # CLAHE contrast normalisation after grayscale

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
