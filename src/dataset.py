from __future__ import annotations

import logging
import os
from typing import Tuple, Dict, Any

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

from src.config import ModelConfig, QAConfig, JOINT_TYPE_MAP, QA_JOINT_TYPE_MAP, QA_MODALITY_MAP
from src.utils import build_train_transforms, build_val_transforms, build_qa_train_transforms, build_qa_val_transforms

log = logging.getLogger('msk')


def score_to_corn(score: int, n_ranks: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Converts a raw ordinal score to CORN targets and masks.
    
    For rank k (0-indexed):
      target = 1 if score >= k+1, else 0
      mask   = 1 if score >= k,   else 0 (rank 0 always active)
      
    Returns:
      targets: Tensor[n_ranks]
      masks:   Tensor[n_ranks]
    """
    targets = torch.zeros(n_ranks, dtype=torch.float32)
    masks = torch.zeros(n_ranks, dtype=torch.float32)
    
    for k in range(n_ranks):
        targets[k] = 1.0 if score >= k + 1 else 0.0
        masks[k] = 1.0 if score >= k else 0.0
        
    return targets, masks


class MSKUltrasoundDataset(Dataset):
    """Dataset for MSK Ultrasound images with CORN encoding."""
    def __init__(self, df: pd.DataFrame, image_dir: str, config: ModelConfig, is_train: bool = True):
        super().__init__()
        # Filter to correct modality
        self.df = df[df[config.modality_col] == config.modality_filter].copy().reset_index(drop=True)
        self.image_dir = image_dir
        self.config = config
        self.is_train = is_train
        
        # Build transforms
        if is_train:
            is_doppler = (config.modality == 'doppler')
            self.transforms = build_train_transforms(
                config.image_size,
                is_doppler=is_doppler,
                color_augmentation=config.color_augmentation,
            )
        else:
            self.transforms = build_val_transforms(config.image_size)

        log.info(f"Initialized MSKUltrasoundDataset (train={is_train}): {len(self.df)} samples")

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, eco_id: str) -> np.ndarray:
        img_path = os.path.join(self.image_dir, f"{eco_id}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, f"{eco_id}.bmp")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found for eco_id {eco_id} (.png or .bmp)")
        
        img = Image.open(img_path).convert('RGB')
        return np.array(img)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        eco_id = str(row[self.config.eco_id_col])
        joint_type_str = str(row[self.config.joint_type_col])
        joint_id = JOINT_TYPE_MAP.get(joint_type_str, -1)
        if joint_id == -1:
            raise ValueError(f"Unknown joint_type: {joint_type_str}")
            
        # Load and transform image
        img_np = self._load_image(eco_id)
        if self.transforms is not None:
            augmented = self.transforms(image=img_np)
            img_tensor = augmented['image']
        else:
            # Fallback if no transforms (should not happen)
            img_tensor = torch.from_numpy(img_np.transpose(2, 0, 1)).float() / 255.0

        corn_targets = {}
        corn_masks = {}
        clinical_masks = {}

        for task in self.config.tasks:
            raw_score = row[task.csv_column]
            if pd.isna(raw_score):
                clinical_masks[task.name] = torch.tensor(0.0, dtype=torch.float32)
                corn_targets[task.name] = torch.zeros(task.n_ranks, dtype=torch.float32)
                corn_masks[task.name] = torch.zeros(task.n_ranks, dtype=torch.float32)
            else:
                clinical_masks[task.name] = torch.tensor(1.0, dtype=torch.float32)
                t, m = score_to_corn(int(raw_score), task.n_ranks)
                corn_targets[task.name] = t
                corn_masks[task.name] = m

        return {
            'image': img_tensor,
            'joint_id': torch.tensor(joint_id, dtype=torch.long),
            'corn_targets': corn_targets,
            'corn_masks': corn_masks,
            'clinical_masks': clinical_masks,
            'eco_id': eco_id,
        }


def msk_collate_fn(batch: list) -> Dict[str, Any]:
    """Custom collate function for nested dict structures."""
    images = torch.stack([b['image'] for b in batch])
    joint_ids = torch.stack([b['joint_id'] for b in batch])
    eco_ids = [b['eco_id'] for b in batch]
    
    task_names = list(batch[0]['corn_targets'].keys())
    
    corn_targets = {t: torch.stack([b['corn_targets'][t] for b in batch]) for t in task_names}
    corn_masks = {t: torch.stack([b['corn_masks'][t] for b in batch]) for t in task_names}
    clinical_masks = {t: torch.stack([b['clinical_masks'][t] for b in batch]) for t in task_names}
    
    return {
        'image': images, 
        'joint_id': joint_ids, 
        'corn_targets': corn_targets,
        'corn_masks': corn_masks, 
        'clinical_masks': clinical_masks, 
        'eco_ids': eco_ids
    }


def build_fold_loaders(config: ModelConfig, fold_idx: int) -> Tuple[DataLoader, DataLoader]:
    """Builds train and validation DataLoaders for a specific fold."""
    df = pd.read_csv(config.labels_csv)
    
    # Train on all splits >= 0 (ignoring holdout which might be -1 or separate)
    df_train_pool = df[df[config.split_col] >= 0].copy()
    
    df_train = df_train_pool[df_train_pool[config.split_col] != fold_idx]
    df_val = df_train_pool[df_train_pool[config.split_col] == fold_idx]
    
    train_ds = MSKUltrasoundDataset(df_train, config.image_dir, config, is_train=True)
    val_ds = MSKUltrasoundDataset(df_val, config.image_dir, config, is_train=False)
    
    # Build WeightedRandomSampler based on the primary task
    primary_task = config.tasks[0]
    # Re-filter train df to match dataset's modality filter
    df_train_modality = df_train[df_train[config.modality_col] == config.modality_filter]
    
    scores = df_train_modality[primary_task.csv_column].fillna(0)
    score_counts = scores.value_counts()
    # Weights are inversely proportional to class frequencies
    weights = 1.0 / score_counts.clip(lower=1)
    sample_weights = scores.map(weights).fillna(1.0).values
    
    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_ds),
        replacement=True
    )
    
    train_loader = DataLoader(
        train_ds, 
        batch_size=config.batch_size,
        sampler=sampler, 
        num_workers=config.num_workers,
        pin_memory=config.pin_memory, 
        collate_fn=msk_collate_fn, 
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_ds, 
        batch_size=config.batch_size * 2,
        shuffle=False, 
        num_workers=config.num_workers,
        pin_memory=config.pin_memory, 
        collate_fn=msk_collate_fn
    )

    return train_loader, val_loader


# ─── QA Gatekeeper Dataset ────────────────────────────────────────────────────

class QADataset(Dataset):
    """
    Dataset for the QA Gatekeeper joint-type classifier.

    Key differences from MSKUltrasoundDataset:
      - No modality filter: trains on ALL rows (B-Mode + Doppler combined).
      - Target is the merged joint class (5 classes, wrist sub-joints merged).
      - Images are converted to grayscale in the transform pipeline so the
        backbone sees pure anatomy regardless of acquisition mode.
      - Returns modality_id (0=bmode, 1=doppler) for the learned embedding.
    """

    def __init__(self, df: pd.DataFrame, image_dir: str, config: QAConfig, is_train: bool = True):
        super().__init__()
        self.df = df.copy().reset_index(drop=True)
        self.image_dir = image_dir
        self.config = config
        self.is_train = is_train

        if is_train:
            self.transforms = build_qa_train_transforms(config.image_size, use_clahe=config.use_clahe)
        else:
            self.transforms = build_qa_val_transforms(config.image_size)

        log.info(f"Initialized QADataset (train={is_train}): {len(self.df)} samples (all modalities)")

    def __len__(self) -> int:
        return len(self.df)

    def _load_image(self, eco_id: str) -> np.ndarray:
        img_path = os.path.join(self.image_dir, f"{eco_id}.png")
        if not os.path.exists(img_path):
            img_path = os.path.join(self.image_dir, f"{eco_id}.bmp")
            if not os.path.exists(img_path):
                raise FileNotFoundError(f"Image not found for eco_id {eco_id} (.png or .bmp)")
        img = Image.open(img_path).convert('RGB')
        return np.array(img)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx]
        eco_id = str(row[self.config.eco_id_col])

        # ── Joint type label (merged 5-class) ─────────────────────────────────
        joint_type_str = str(row[self.config.joint_type_col])
        joint_id = QA_JOINT_TYPE_MAP.get(joint_type_str, -1)
        if joint_id == -1:
            raise ValueError(f"Unknown joint_type for QA: '{joint_type_str}'")

        # ── Modality id (for bias-correction embedding) ───────────────────────
        modality_str = str(row[self.config.modality_col])
        modality_id  = QA_MODALITY_MAP.get(modality_str, -1)
        if modality_id == -1:
            raise ValueError(f"Unknown modality for QA: '{modality_str}'")

        # ── Image (grayscale transform applied inside pipeline) ───────────────
        img_np   = self._load_image(eco_id)
        augmented = self.transforms(image=img_np)
        img_tensor = augmented['image']

        return {
            'image':       img_tensor,
            'joint_id':    torch.tensor(joint_id,    dtype=torch.long),
            'modality_id': torch.tensor(modality_id, dtype=torch.long),
            'eco_id':      eco_id,
        }


def qa_collate_fn(batch: list) -> Dict[str, Any]:
    """Custom collate function for QADataset batches."""
    return {
        'image':       torch.stack([b['image']       for b in batch]),
        'joint_id':    torch.stack([b['joint_id']    for b in batch]),
        'modality_id': torch.stack([b['modality_id'] for b in batch]),
        'eco_ids':     [b['eco_id'] for b in batch],
    }


def build_qa_fold_loaders(config: QAConfig, fold_idx: int) -> Tuple[DataLoader, DataLoader]:
    """
    Build train and validation DataLoaders for a single QA cross-validation fold.

    Uses the pre-computed `split` column (StratifiedGroupKFold on eco_id) so
    that both Radiocarpiana and Intercarpiana rows for the same image stay in
    the same fold — preventing label leakage for the merged wrist class.

    All rows are used (no modality filter).  Holdout rows (split < 0) are
    excluded from the train/val pool.
    """
    df = pd.read_csv(config.labels_csv)
    df_pool  = df[df[config.split_col] >= 0].copy()

    df_train = df_pool[df_pool[config.split_col] != fold_idx]
    df_val   = df_pool[df_pool[config.split_col] == fold_idx]

    train_ds = QADataset(df_train, config.image_dir, config, is_train=True)
    val_ds   = QADataset(df_val,   config.image_dir, config, is_train=False)

    # ── WeightedRandomSampler on the merged 5-class label ─────────────────────
    train_joint_ids = df_train[config.joint_type_col].map(QA_JOINT_TYPE_MAP)
    class_counts    = train_joint_ids.value_counts()
    inv_freq        = 1.0 / class_counts.clip(lower=1)
    sample_weights  = train_joint_ids.map(inv_freq).fillna(1.0).values

    sampler = WeightedRandomSampler(
        weights=torch.tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_ds),
        replacement=True,
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        sampler=sampler,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=qa_collate_fn,
        drop_last=True,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size * 2,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        collate_fn=qa_collate_fn,
    )

    return train_loader, val_loader
