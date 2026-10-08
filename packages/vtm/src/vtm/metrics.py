"""Segmentation metrics over role maps."""

from __future__ import annotations

import numpy as np

from vtcore.roles import N_ROLES, ROLES


def confusion(preds, golds, n: int = N_ROLES) -> np.ndarray:
    cm = np.zeros((n, n), np.int64)
    for p, g in zip(preds, golds):
        m = g >= 0
        cm += np.bincount(g[m].astype(np.int64) * n + p[m].astype(np.int64), minlength=n * n).reshape(n, n)
    return cm  # rows = gold, cols = pred


def report(cm: np.ndarray) -> dict:
    tp = np.diag(cm).astype(float)
    fp, fn = cm.sum(0) - tp, cm.sum(1) - tp
    iou = tp / np.maximum(tp + fp + fn, 1)
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
    support = cm.sum(1)
    present = support > 0
    nb = slice(1, None)  # exclude blank
    return {
        "per_class": {ROLES[i]: {"iou": round(iou[i], 4), "f1": round(f1[i], 4), "support": int(support[i])} for i in range(len(tp))},
        "macro_iou": float(iou[nb][present[nb]].mean()),
        "macro_f1": float(f1[nb][present[nb]].mean()),
        "acc_nonblank": float(tp[nb].sum() / max(cm[nb].sum(), 1)),
    }
