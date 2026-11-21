"""ROC, precision-recall and their summary statistics.

Two deliberate choices, both because the negative class piles tens of thousands
of sequences at P(S) ~ 0:

* **AUC is the rank statistic** (Mann-Whitney U), not the area under the drawn
  curve. With that many near-ties, trapezoid integration over the plotted points
  is biased; the rank form handles ties exactly.
* **Average precision is the step-wise sum** ``sum (R_n - R_{n-1}) * P_n`` with no
  interpolation, which is the standard estimator and does not flatter the curve.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import rankdata


def _ordered(score: np.ndarray, positive: np.ndarray, negative: np.ndarray):
    labels = np.concatenate([np.ones(int(positive.sum())), np.zeros(int(negative.sum()))])
    values = np.concatenate([score[positive], score[negative]])
    order = np.argsort(-values, kind="mergesort")
    return labels[order], values, int(positive.sum()), int(negative.sum())


def auc(score: np.ndarray, positive: np.ndarray, negative: np.ndarray) -> float:
    """Mann-Whitney U / (n1 n0): the probability a positive outranks a negative."""
    _, values, n1, n0 = _ordered(score, positive, negative)
    ranks = rankdata(values)
    return float((ranks[:n1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def roc(score: np.ndarray, positive: np.ndarray, negative: np.ndarray):
    """``(fpr, tpr, auc)``."""
    labels, _, n1, n0 = _ordered(score, positive, negative)
    tpr = np.concatenate([[0.0], np.cumsum(labels) / n1])
    fpr = np.concatenate([[0.0], np.cumsum(1 - labels) / n0])
    return fpr, tpr, auc(score, positive, negative)


def pr(score: np.ndarray, positive: np.ndarray, negative: np.ndarray):
    """``(recall, precision, average_precision)``."""
    labels, _, n1, _ = _ordered(score, positive, negative)
    tp = np.cumsum(labels)
    fp = np.cumsum(1 - labels)
    precision = tp / (tp + fp)
    recall = tp / n1
    ap = float(np.sum(np.diff(np.concatenate([[0.0], recall])) * precision))
    return recall, precision, ap


def precision_at_recall(recall: np.ndarray, precision: np.ndarray, target: float) -> float:
    """Precision at the first operating point reaching ``target`` recall."""
    index = int(np.searchsorted(recall, target))
    return float(precision[min(index, len(precision) - 1)])
