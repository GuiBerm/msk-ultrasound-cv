from __future__ import annotations

import logging
from typing import Dict, List

import numpy as np
import torch
from sklearn.metrics import cohen_kappa_score, f1_score, confusion_matrix

log = logging.getLogger('msk')


def corn_logits_to_score(logits: torch.Tensor) -> torch.Tensor:
    """
    Converts CORN logits to a predicted ordinal score.
    
    Args:
        logits: (B, K) raw logits for K conditional ranks
        
    Returns:
        (B,) integer tensor with the predicted score (0 to K)
    """
    # P(Y >= k) = prod_{j=1}^k sigmoid(logit_j)
    probs = torch.sigmoid(logits)
    cum_probs = torch.cumprod(probs, dim=-1)
    
    # Predict rank if conditional probability > 0.5
    predicted_score = (cum_probs > 0.5).sum(dim=-1).long()
    return predicted_score


class MetricAccumulator:
    """
    Accumulates predictions and targets over an epoch to compute QWK and MAE.
    """
    def __init__(self, task_names: List[str], task_n_ranks: Dict[str, int]):
        self.task_names = task_names
        self.task_n_ranks = task_n_ranks
        self._preds = {t: [] for t in task_names}
        self._trues = {t: [] for t in task_names}

    def update(
        self, 
        logits_dict: Dict[str, torch.Tensor], 
        corn_targets_dict: Dict[str, torch.Tensor], 
        clinical_masks_dict: Dict[str, torch.Tensor]
    ) -> None:
        """
        Update accumulators with a new batch.
        Only unmasked samples are stored.
        """
        for task_name in self.task_names:
            mask = clinical_masks_dict[task_name].bool().cpu()
            if mask.sum() == 0:
                continue
                
            logits = logits_dict[task_name].detach().cpu()
            corn_targets = corn_targets_dict[task_name].detach().cpu()
            
            # Reconstruct true score: sum the CORN targets
            true_scores = corn_targets.sum(dim=-1).long()
            
            # Predict score
            pred_scores = corn_logits_to_score(logits)
            
            # Apply mask
            self._preds[task_name].append(pred_scores[mask].numpy())
            self._trues[task_name].append(true_scores[mask].numpy())

    def compute(self) -> Dict[str, dict]:
        """
        Compute metrics for all tasks.
        Returns {task_name: {'qwk': float, 'mae': float, 'n': int}}
        """
        results = {}
        for task_name in self.task_names:
            if not self._preds[task_name]:
                results[task_name] = {'qwk': float('nan'), 'mae': float('nan'), 'n': 0}
                continue
                
            preds = np.concatenate(self._preds[task_name])
            trues = np.concatenate(self._trues[task_name])
            n_samples = len(preds)
            
            # Compute MAE
            mae = float(np.mean(np.abs(preds - trues)))
            
            # Compute QWK
            n_ranks = self.task_n_ranks[task_name]
            labels = list(range(n_ranks + 1))
            
            # Need at least 2 unique classes for QWK
            if len(np.unique(trues)) < 2:
                qwk = float('nan')
            else:
                try:
                    qwk = cohen_kappa_score(trues, preds, weights='quadratic', labels=labels)
                except ValueError:
                    qwk = float('nan')
                    
            results[task_name] = {'qwk': qwk, 'mae': mae, 'n': n_samples}
            
        return results

    def reset(self) -> None:
        """Clear all accumulators."""
        self._preds = {t: [] for t in self.task_names}
        self._trues = {t: [] for t in self.task_names}

    def mean_qwk(self) -> float:
        """Compute the average QWK across all tasks."""
        metrics = self.compute()
        qwks = [v['qwk'] for v in metrics.values() if not np.isnan(v['qwk'])]
        return float(np.mean(qwks)) if qwks else float('nan')


# ─── QA Gatekeeper Metrics ────────────────────────────────────────────────────

class QAMetricAccumulator:
    """
    Accumulates predictions and ground-truth labels over an epoch for the
    QA Gatekeeper multi-class classification task.

    Primary metric: macro F1 (robust to the 5-class imbalance).
    Secondary metrics: overall accuracy, per-class accuracy, unweighted Cohen's Kappa.
    """

    def __init__(self, num_classes: int, class_names: List[str]):
        self.num_classes  = num_classes
        self.class_names  = class_names
        self._preds: List[np.ndarray] = []
        self._trues: List[np.ndarray] = []

    def update(self, logits: torch.Tensor, targets: torch.Tensor) -> None:
        """
        Update accumulators with a new batch.

        Args:
            logits:  (B, num_classes) — raw class logits from the model
            targets: (B,)             — ground-truth class indices
        """
        preds = logits.argmax(dim=-1).detach().cpu().numpy()
        trues = targets.detach().cpu().numpy()
        self._preds.append(preds)
        self._trues.append(trues)

    def compute(self) -> Dict[str, object]:
        """
        Compute all QA metrics.

        Returns a dict with:
          - 'accuracy':       float  — overall accuracy
          - 'macro_f1':       float  — macro-averaged F1 (primary metric)
          - 'kappa':          float  — unweighted Cohen's Kappa
          - 'per_class_acc':  dict[class_name, float]
          - 'confusion_matrix': np.ndarray (num_classes x num_classes)
          - 'n':              int    — total samples evaluated
        """
        if not self._preds:
            return {
                'accuracy': float('nan'), 'macro_f1': float('nan'),
                'kappa': float('nan'), 'per_class_acc': {}, 'confusion_matrix': None, 'n': 0,
            }

        preds = np.concatenate(self._preds)
        trues = np.concatenate(self._trues)
        n     = len(preds)

        # Overall accuracy
        accuracy = float(np.mean(preds == trues))

        # Macro F1 (primary — handles class imbalance)
        labels = list(range(self.num_classes))
        macro_f1 = float(f1_score(trues, preds, labels=labels, average='macro', zero_division=0))

        # Unweighted Cohen's Kappa
        try:
            kappa = float(cohen_kappa_score(trues, preds, labels=labels))
        except ValueError:
            kappa = float('nan')

        # Confusion matrix
        cm = confusion_matrix(trues, preds, labels=labels)

        # Per-class accuracy (diagonal of row-normalised confusion matrix)
        per_class_acc = {}
        for i, name in enumerate(self.class_names):
            row_sum = cm[i].sum()
            per_class_acc[name] = float(cm[i, i] / row_sum) if row_sum > 0 else float('nan')

        return {
            'accuracy':        accuracy,
            'macro_f1':        macro_f1,
            'kappa':           kappa,
            'per_class_acc':   per_class_acc,
            'confusion_matrix': cm,
            'n':               n,
        }

    def reset(self) -> None:
        """Clear all accumulators."""
        self._preds = []
        self._trues = []

    def macro_f1(self) -> float:
        """Convenience method: return macro F1 directly (used by QATrainer for early stopping)."""
        return self.compute()['macro_f1']
