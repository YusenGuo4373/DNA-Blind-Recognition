from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any
import json

import numpy as np

from .config import KNOWN_CODE_TYPES


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    positive = values >= 0
    result = np.empty_like(values)
    result[positive] = 1.0 / (1.0 + np.exp(-values[positive]))
    exp_values = np.exp(values[~positive])
    result[~positive] = exp_values / (1.0 + exp_values)
    return result


def softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float64)
    shifted = logits - np.max(logits, axis=axis, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=axis, keepdims=True)


def logsumexp(values: np.ndarray, axis: int = -1) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    maximum = np.max(values, axis=axis, keepdims=True)
    return np.squeeze(maximum, axis=axis) + np.log(
        np.sum(np.exp(values - maximum), axis=axis)
    )


def two_level_soft_vote(read_values: np.ndarray) -> np.ndarray:
    """Average q reads within molecules, then M molecule scores."""

    values = np.asarray(read_values, dtype=np.float64)
    if values.ndim < 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("read_values must begin with non-empty [M, q] dimensions")
    molecule_values = values.mean(axis=1)
    return molecule_values.mean(axis=0)


def energy_score(logits: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    logits = np.asarray(logits, dtype=np.float64)
    return -temperature * logsumexp(logits / temperature, axis=-1)


def binary_confusion(y_true: np.ndarray, y_pred: np.ndarray) -> tuple[int, int, int, int]:
    truth = np.asarray(y_true, dtype=np.int64).reshape(-1)
    prediction = np.asarray(y_pred, dtype=np.int64).reshape(-1)
    if truth.shape != prediction.shape:
        raise ValueError("y_true and y_pred must have matching shapes")
    tn = int(np.sum((truth == 0) & (prediction == 0)))
    fp = int(np.sum((truth == 0) & (prediction == 1)))
    fn = int(np.sum((truth == 1) & (prediction == 0)))
    tp = int(np.sum((truth == 1) & (prediction == 1)))
    return tn, fp, fn, tp


def binary_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, fn, tp = binary_confusion(y_true, y_pred)

    def f1(true_positive: int, false_positive: int, false_negative: int) -> float:
        denominator = 2 * true_positive + false_positive + false_negative
        return 0.0 if denominator == 0 else 2.0 * true_positive / denominator

    return float((f1(tn, fn, fp) + f1(tp, fp, fn)) / 2.0)


def select_presence_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    """Select tau1 exclusively by validation macro-F1."""

    scores = np.asarray(scores, dtype=np.float64).reshape(-1)
    labels = np.asarray(labels, dtype=np.int64).reshape(-1)
    if scores.size != labels.size or scores.size == 0:
        raise ValueError("scores and labels must be non-empty and aligned")
    if np.unique(labels).size != 2:
        raise ValueError("both validation classes are required")
    unique = np.unique(scores)
    if unique.size == 1:
        candidates = np.array([unique[0], np.nextafter(unique[0], np.inf)])
    else:
        midpoints = (unique[:-1] + unique[1:]) / 2.0
        candidates = np.concatenate(
            ([np.nextafter(unique[0], -np.inf)], midpoints, [np.nextafter(unique[-1], np.inf)])
        )
    scored = [binary_macro_f1(labels, scores >= threshold) for threshold in candidates]
    best_score = max(scored)
    tied = [
        float(threshold)
        for threshold, score in zip(candidates, scored)
        if np.isclose(score, best_score, rtol=0.0, atol=1e-12)
    ]
    threshold = min(tied, key=lambda value: (abs(value - 0.5), value))
    return threshold, float(best_score)


def select_unknown_threshold(known_energy_scores: np.ndarray, known_acceptance: float = 0.95) -> float:
    """Set tau2 from known validation energies only.

    Lower energy is accepted as known.  The threshold is the smallest order
    statistic that accepts at least the requested finite-sample fraction.
    """

    scores = np.asarray(known_energy_scores, dtype=np.float64).reshape(-1)
    scores = scores[np.isfinite(scores)]
    if scores.size == 0:
        raise ValueError("at least one finite known energy score is required")
    if not 0.0 < known_acceptance <= 1.0:
        raise ValueError("known_acceptance must be in (0, 1]")
    ordered = np.sort(scores)
    rank = max(0, int(np.ceil(known_acceptance * ordered.size)) - 1)
    return float(ordered[rank])


@dataclass(frozen=True)
class Thresholds:
    presence: float
    unknown_energy: float
    known_acceptance: float = 0.95
    temperature: float = 1.0

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "Thresholds":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class HierarchicalDecision:
    status: str
    code_type: str | None
    code_rate: None
    code_length: None
    ecc_score: float
    unknown_score: float | None
    type_probabilities: tuple[float, ...] | None
    max_type_probability: float | None
    max_logit: float | None
    q: int
    M: int
    thresholds: Thresholds

    def to_dict(self) -> dict[str, Any]:
        result = asdict(self)
        result["thresholds"] = asdict(self.thresholds)
        return result


def hierarchical_decision(
    presence_probabilities: np.ndarray,
    type_logits: np.ndarray | None,
    thresholds: Thresholds,
) -> HierarchicalDecision:
    presence = np.asarray(presence_probabilities, dtype=np.float64)
    if presence.ndim != 2:
        raise ValueError("presence_probabilities must have shape [M, q]")
    molecules, reads = presence.shape
    ecc_score = float(two_level_soft_vote(presence))
    if ecc_score < thresholds.presence:
        return HierarchicalDecision(
            status="no_ecc",
            code_type=None,
            code_rate=None,
            code_length=None,
            ecc_score=ecc_score,
            unknown_score=None,
            type_probabilities=None,
            max_type_probability=None,
            max_logit=None,
            q=reads,
            M=molecules,
            thresholds=thresholds,
        )

    if type_logits is None:
        raise ValueError("type_logits are required after the ECC stage accepts a sample")
    logits = np.asarray(type_logits, dtype=np.float64)
    if logits.shape != (molecules, reads, len(KNOWN_CODE_TYPES)):
        raise ValueError("type_logits must have shape [M, q, 4]")
    archive_energy = float(
        two_level_soft_vote(energy_score(logits, temperature=thresholds.temperature))
    )
    probabilities = np.asarray(two_level_soft_vote(softmax(logits, axis=-1)))
    mean_logits = np.asarray(two_level_soft_vote(logits))
    maximum_probability = float(np.max(probabilities))
    maximum_logit = float(np.max(mean_logits))
    if archive_energy > thresholds.unknown_energy:
        status = "unknown_ecc"
        code_type = None
    else:
        status = "known_ecc"
        code_type = KNOWN_CODE_TYPES[int(np.argmax(probabilities))]
    return HierarchicalDecision(
        status=status,
        code_type=code_type,
        code_rate=None,
        code_length=None,
        ecc_score=ecc_score,
        unknown_score=archive_energy,
        type_probabilities=tuple(float(value) for value in probabilities),
        max_type_probability=maximum_probability,
        max_logit=maximum_logit,
        q=reads,
        M=molecules,
        thresholds=thresholds,
    )
