from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: tuple[str, ...] | tuple[int, ...] | list[str] | list[int],
) -> np.ndarray:
    truth = np.asarray(y_true).reshape(-1)
    prediction = np.asarray(y_pred).reshape(-1)
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have matching shapes")
    index = {label: position for position, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for expected, observed in zip(truth, prediction):
        if expected not in index or observed not in index:
            raise ValueError(f"unknown label pair: {expected!r}, {observed!r}")
        matrix[index[expected], index[observed]] += 1
    return matrix


def accuracy_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    truth = np.asarray(y_true).reshape(-1)
    prediction = np.asarray(y_pred).reshape(-1)
    if truth.size == 0 or truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must be non-empty and aligned")
    return float(np.mean(truth == prediction))


def f1_from_confusion(matrix: np.ndarray) -> tuple[np.ndarray, float]:
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("matrix must be square")
    scores = np.zeros(matrix.shape[0], dtype=np.float64)
    for index in range(matrix.shape[0]):
        true_positive = int(matrix[index, index])
        false_positive = int(matrix[:, index].sum() - true_positive)
        false_negative = int(matrix[index, :].sum() - true_positive)
        denominator = 2 * true_positive + false_positive + false_negative
        scores[index] = 0.0 if denominator == 0 else 2.0 * true_positive / denominator
    return scores, float(scores.mean())


def macro_f1_score(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: tuple[str, ...] | tuple[int, ...] | list[str] | list[int],
) -> float:
    _, score = f1_from_confusion(confusion_matrix(y_true, y_pred, labels))
    return score


def binary_roc_curve(y_true: np.ndarray, scores: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    if truth.size == 0 or truth.shape != scores.shape:
        raise ValueError("labels and scores must be non-empty and aligned")
    positives = int(np.sum(truth == 1))
    negatives = int(np.sum(truth == 0))
    if positives == 0 or negatives == 0:
        return np.array([np.nan]), np.array([np.nan])
    order = np.argsort(-scores, kind="mergesort")
    ordered_truth = truth[order]
    ordered_scores = scores[order]
    distinct = np.r_[np.flatnonzero(np.diff(ordered_scores)), truth.size - 1]
    true_positives = np.cumsum(ordered_truth == 1)[distinct]
    false_positives = (distinct + 1) - true_positives
    tpr = np.r_[0.0, true_positives / positives]
    fpr = np.r_[0.0, false_positives / negatives]
    return fpr.astype(np.float64), tpr.astype(np.float64)


def binary_auroc(y_true: np.ndarray, scores: np.ndarray) -> float:
    fpr, tpr = binary_roc_curve(y_true, scores)
    if np.isnan(fpr).any():
        return float("nan")
    return float(np.trapezoid(tpr, fpr))


def binary_aupr(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Average precision for label 1, using descending score thresholds."""

    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    positives = int(np.sum(truth == 1))
    if positives == 0 or truth.shape != scores.shape:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    ordered = truth[order]
    true_positives = np.cumsum(ordered == 1)
    false_positives = np.cumsum(ordered == 0)
    precision = true_positives / (true_positives + false_positives)
    recall_step = (ordered == 1).astype(np.float64) / positives
    return float(np.sum(precision * recall_step))


def fpr_at_tpr(y_true: np.ndarray, scores: np.ndarray, target_tpr: float = 0.95) -> float:
    """Minimum negative-class FPR among thresholds reaching target positive TPR."""

    fpr, tpr = binary_roc_curve(y_true, scores)
    if np.isnan(fpr).any():
        return float("nan")
    candidates = fpr[tpr >= target_tpr]
    return float(np.min(candidates)) if candidates.size else 1.0


@dataclass(frozen=True)
class ConfidenceInterval:
    estimate: float
    lower: float
    upper: float


def bootstrap_confidence_interval(
    sample_size: int,
    metric: Callable[[np.ndarray], float],
    seed: int,
    resamples: int = 1000,
    confidence: float = 0.95,
) -> ConfidenceInterval:
    if sample_size <= 0 or resamples <= 0:
        raise ValueError("sample_size and resamples must be positive")
    if not 0.0 < confidence < 1.0:
        raise ValueError("confidence must be in (0, 1)")
    full_indices = np.arange(sample_size)
    estimate = float(metric(full_indices))
    rng = np.random.default_rng(seed)
    values = np.empty(resamples, dtype=np.float64)
    for index in range(resamples):
        values[index] = metric(rng.integers(0, sample_size, size=sample_size))
    alpha = (1.0 - confidence) / 2.0
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return ConfidenceInterval(estimate, float("nan"), float("nan"))
    lower, upper = np.quantile(finite, (alpha, 1.0 - alpha))
    return ConfidenceInterval(estimate, float(lower), float(upper))
