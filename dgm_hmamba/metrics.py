"""Classification metrics: OA, AA, Cohen's kappa and per-class accuracy.

All functions work on plain numpy arrays so they can be used without torch.
"""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import numpy as np


def confusion_matrix(y_true: Sequence[int], y_pred: Sequence[int], n_classes: int) -> np.ndarray:
    """Rows = true class, columns = predicted class."""
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (y_true, y_pred), 1)
    return cm


def metrics_from_confusion(cm: np.ndarray) -> Dict[str, object]:
    """OA, AA (mean of per-class recall over classes that occur) and kappa."""
    cm = np.asarray(cm, dtype=np.float64)
    total = cm.sum()
    if total == 0:
        return {"oa": 0.0, "aa": 0.0, "kappa": 0.0, "per_class": []}
    diag = np.diag(cm)
    support = cm.sum(axis=1)
    oa = diag.sum() / total
    present = support > 0
    per_class = np.where(present, diag / np.maximum(support, 1), np.nan)
    aa = float(np.nanmean(per_class)) if present.any() else 0.0
    pe = float((cm.sum(axis=0) * cm.sum(axis=1)).sum()) / (total * total)
    kappa = (oa - pe) / (1 - pe) if pe < 1 else 0.0
    predicted = cm.sum(axis=0)
    precision = np.where(predicted > 0, diag / np.maximum(predicted, 1), np.nan)
    f1 = np.where((precision + per_class) > 0,
                  2 * precision * per_class / np.maximum(precision + per_class, 1e-12), np.nan)
    tolist = lambda a: [None if np.isnan(v) else float(v) for v in a]
    return {
        "oa": float(oa),
        "aa": aa,
        "kappa": float(kappa),
        "per_class": tolist(per_class),          # recall = class accuracy
        "precision": tolist(precision),
        "f1": tolist(f1),
    }


def classification_report(
    y_true: Sequence[int], y_pred: Sequence[int], n_classes: Optional[int] = None
) -> Dict[str, object]:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)
    if n_classes is None:
        n_classes = int(max(y_true.max(initial=-1), y_pred.max(initial=-1)) + 1)
    cm = confusion_matrix(y_true, y_pred, n_classes)
    out = metrics_from_confusion(cm)
    out["confusion_matrix"] = cm.tolist()
    return out
