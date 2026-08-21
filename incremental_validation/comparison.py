from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
import csv
import json

import numpy as np
import torch

from author_baseline.soft_voting import author_soft_vote


KNOWN_TYPES = ("BCH", "Convolutional", "LDPC", "Polar")
NO_ECC_TYPES = ("NoECC-Random", "NoECC-Constrained", "no_ecc")
UNKNOWN_ECC_TYPES = ("HEDGES", "DNA-Aeon", "unknown_ecc")
STAGE1_SUPERVISED_INNER_CODES = ("HEDGES", "DNA-Aeon")
LEGACY_UNKNOWN_TYPES = ("Fountain", "LT")
FORBIDDEN_CALIBRATION_TYPES = ("unknown_ecc",) + LEGACY_UNKNOWN_TYPES
TEST_UNKNOWN_TYPES = UNKNOWN_ECC_TYPES + LEGACY_UNKNOWN_TYPES
END_TO_END_LABELS = ("no_ecc", "unknown_ecc") + KNOWN_TYPES


@dataclass(frozen=True)
class IncrementalThresholds:
    ecc_presence: float
    unknown_energy: float
    known_acceptance_target: float = 0.95
    energy_temperature: float = 1.0

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(asdict(self), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "IncrementalThresholds":
        return cls(**json.loads(Path(path).read_text(encoding="utf-8")))


@dataclass(frozen=True)
class SharedLogitComparison:
    category: str
    closed_set_output: str
    cascade_output: str
    cascade_status: str
    code_type: str | None
    code_rate: None
    code_length: None
    ecc_score: float
    unknown_energy: float
    type_probabilities: tuple[float, ...]
    type_confidence: float
    max_logit: float
    ecc_gate_passed: bool
    energy_rejected: bool
    q: int
    M: int
    archive_id: str | int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _validate_shared_arrays(
    presence_probabilities: np.ndarray,
    type_logits: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    presence = np.asarray(presence_probabilities, dtype=np.float64)
    logits = np.asarray(type_logits, dtype=np.float64)
    if presence.ndim != 2 or presence.shape[0] == 0 or presence.shape[1] == 0:
        raise ValueError("presence probabilities must have shape [M,q]")
    if logits.shape != presence.shape + (len(KNOWN_TYPES),):
        raise ValueError("type logits must have shape [M,q,4] and share M/q")
    if np.any((presence < 0.0) | (presence > 1.0)):
        raise ValueError("presence probabilities must lie in [0,1]")
    return presence, logits


def _energy(logits: np.ndarray, temperature: float) -> float:
    if temperature <= 0:
        raise ValueError("temperature must be positive")
    values = torch.as_tensor(logits, dtype=torch.float64)
    read_energy = -temperature * torch.logsumexp(values / temperature, dim=-1)
    return float(read_energy.mean(dim=1).mean(dim=0).item())


def compare_shared_logits(
    category: str,
    presence_probabilities: np.ndarray,
    type_logits: np.ndarray,
    thresholds: IncrementalThresholds,
    archive_id: str | int | None = None,
) -> SharedLogitComparison:
    """Compare closed-set and cascade outputs from exactly the same type logits."""

    presence, logits = _validate_shared_arrays(presence_probabilities, type_logits)
    tensor_logits = torch.as_tensor(logits, dtype=torch.float64)
    vote = author_soft_vote(tensor_logits)
    closed_output = KNOWN_TYPES[vote.predicted_index]
    ecc_score = float(presence.mean(axis=1).mean(axis=0))
    unknown_energy = _energy(logits, thresholds.energy_temperature)

    if ecc_score < thresholds.ecc_presence:
        status = "no_ecc"
        code_type = None
        cascade_output = "no_ecc"
    elif unknown_energy > thresholds.unknown_energy:
        status = "unknown_ecc"
        code_type = None
        cascade_output = "unknown_ecc"
    else:
        status = "known_ecc"
        code_type = closed_output
        cascade_output = closed_output

    return SharedLogitComparison(
        category=category,
        closed_set_output=closed_output,
        cascade_output=cascade_output,
        cascade_status=status,
        code_type=code_type,
        code_rate=None,
        code_length=None,
        ecc_score=ecc_score,
        unknown_energy=unknown_energy,
        type_probabilities=tuple(float(value) for value in vote.archive_probabilities.tolist()),
        type_confidence=vote.confidence,
        max_logit=float(tensor_logits.mean(dim=1).mean(dim=0).max().item()),
        ecc_gate_passed=bool(ecc_score >= thresholds.ecc_presence),
        energy_rejected=bool(unknown_energy > thresholds.unknown_energy),
        q=int(presence.shape[1]),
        M=int(presence.shape[0]),
        archive_id=archive_id,
    )


def _binary_macro_f1(labels: np.ndarray, prediction: np.ndarray) -> float:
    scores = []
    for value in (0, 1):
        true_positive = int(np.sum((labels == value) & (prediction == value)))
        false_positive = int(np.sum((labels != value) & (prediction == value)))
        false_negative = int(np.sum((labels == value) & (prediction != value)))
        denominator = 2 * true_positive + false_positive + false_negative
        scores.append(0.0 if denominator == 0 else 2 * true_positive / denominator)
    return float(np.mean(scores))


def _presence_threshold(scores: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    unique = np.unique(scores)
    candidates = [np.nextafter(unique[0], -np.inf)]
    candidates.extend(float((left + right) / 2) for left, right in zip(unique[:-1], unique[1:]))
    candidates.append(np.nextafter(unique[-1], np.inf))
    results = [(float(value), _binary_macro_f1(labels, scores >= value)) for value in candidates]
    best = max(score for _, score in results)
    tied = [value for value, score in results if np.isclose(score, best, atol=1e-12, rtol=0.0)]
    return min(tied, key=lambda value: (abs(value - 0.5), value)), float(best)


def calibrate_thresholds(
    categories: Sequence[str],
    presence_probabilities: np.ndarray,
    type_logits: np.ndarray,
    known_acceptance: float = 0.95,
    energy_temperature: float = 1.0,
) -> tuple[IncrementalThresholds, dict[str, float]]:
    """Calibrate tau1 with supervised inner codes, but tau2 with four known types only."""

    categories = np.asarray(categories, dtype=object)
    presence = np.asarray(presence_probabilities, dtype=np.float64)
    logits = np.asarray(type_logits, dtype=np.float64)
    if presence.ndim != 3 or logits.shape != presence.shape + (4,):
        raise ValueError("validation arrays must have [N,M,q] and [N,M,q,4]")
    if categories.shape != (presence.shape[0],):
        raise ValueError("categories must align with archive axis")
    forbidden = [str(value) for value in categories if value in FORBIDDEN_CALIBRATION_TYPES]
    if forbidden:
        raise ValueError(f"categories forbidden during threshold calibration: {sorted(set(forbidden))}")
    allowed = set(KNOWN_TYPES + NO_ECC_TYPES + STAGE1_SUPERVISED_INNER_CODES)
    unknown_labels = sorted({str(value) for value in categories if value not in allowed})
    if unknown_labels:
        raise ValueError(f"unsupported validation categories: {unknown_labels}")

    archive_presence = presence.mean(axis=2).mean(axis=1)
    binary_labels = np.isin(
        categories, KNOWN_TYPES + STAGE1_SUPERVISED_INNER_CODES
    ).astype(np.int64)
    if np.unique(binary_labels).size != 2:
        raise ValueError("validation requires both known ECC and no-ECC archives")
    tau1, macro_f1 = _presence_threshold(archive_presence, binary_labels)

    known_mask = np.isin(categories, KNOWN_TYPES)
    known_energies = np.asarray(
        [_energy(archive_logits, energy_temperature) for archive_logits in logits[known_mask]],
        dtype=np.float64,
    )
    ordered = np.sort(known_energies)
    rank = max(0, int(np.ceil(known_acceptance * ordered.size)) - 1)
    tau2 = float(ordered[rank])
    thresholds = IncrementalThresholds(
        ecc_presence=tau1,
        unknown_energy=tau2,
        known_acceptance_target=known_acceptance,
        energy_temperature=energy_temperature,
    )
    return thresholds, {
        "presence_validation_macro_f1": macro_f1,
        "known_energy_acceptance": float(np.mean(known_energies <= tau2)),
        "stage1_supervised_inner_code_samples": float(
            np.sum(np.isin(categories, STAGE1_SUPERVISED_INNER_CODES))
        ),
        "stage2_unknown_samples_used": 0.0,
    }


def _truth_label(category: str) -> str:
    if category in NO_ECC_TYPES:
        return "no_ecc"
    if category in TEST_UNKNOWN_TYPES:
        return "unknown_ecc"
    if category in KNOWN_TYPES:
        return category
    raise ValueError(f"unsupported test category {category!r}")


def _confusion(truth: Sequence[str], prediction: Sequence[str]) -> np.ndarray:
    index = {label: position for position, label in enumerate(END_TO_END_LABELS)}
    matrix = np.zeros((len(index), len(index)), dtype=np.int64)
    for expected, observed in zip(truth, prediction):
        matrix[index[expected], index[observed]] += 1
    return matrix


def _macro_f1(truth: Sequence[str], prediction: Sequence[str], labels: Sequence[str]) -> float:
    scores = []
    truth = np.asarray(truth, dtype=object)
    prediction = np.asarray(prediction, dtype=object)
    for label in labels:
        tp = int(np.sum((truth == label) & (prediction == label)))
        fp = int(np.sum((truth != label) & (prediction == label)))
        fn = int(np.sum((truth == label) & (prediction != label)))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(scores))


def summarize_comparisons(comparisons: Sequence[SharedLogitComparison]) -> dict[str, Any]:
    if not comparisons:
        raise ValueError("comparisons must not be empty")
    categories = np.asarray([item.category for item in comparisons], dtype=object)
    closed = np.asarray([item.closed_set_output for item in comparisons], dtype=object)
    cascade = np.asarray([item.cascade_output for item in comparisons], dtype=object)
    truth = np.asarray([_truth_label(str(value)) for value in categories], dtype=object)
    no_ecc = np.isin(categories, NO_ECC_TYPES)
    unknown = np.isin(categories, TEST_UNKNOWN_TYPES)
    known = np.isin(categories, KNOWN_TYPES)
    accepted_known = known & np.isin(cascade, KNOWN_TYPES)
    gate_passed = np.asarray([item.ecc_gate_passed for item in comparisons], dtype=np.bool_)
    energy_rejected = np.asarray([item.energy_rejected for item in comparisons], dtype=np.bool_)
    closed_known_f1 = _macro_f1(categories[known], closed[known], KNOWN_TYPES)
    cascade_known_e2e_f1 = _macro_f1(categories[known], cascade[known], KNOWN_TYPES)
    conditional_f1 = (
        _macro_f1(categories[accepted_known], cascade[accepted_known], KNOWN_TYPES)
        if np.any(accepted_known)
        else float("nan")
    )
    closed_matrix = _confusion(truth, closed)
    cascade_matrix = _confusion(truth, cascade)

    def false_known_rate(mask: np.ndarray, predictions: np.ndarray) -> float:
        return float(np.mean(np.isin(predictions[mask], KNOWN_TYPES))) if np.any(mask) else float("nan")

    return {
        "experiment_name": "基于作者盲识别核心的增量功能验证",
        "shared_type_logits": True,
        "closed_set": {
            "no_ecc_output_as_known_rate": false_known_rate(no_ecc, closed),
            "unknown_ecc_output_as_known_rate": false_known_rate(unknown, closed),
            "known_type_macro_f1": closed_known_f1,
            "labels": list(END_TO_END_LABELS),
            "end_to_end_confusion_matrix": closed_matrix.tolist(),
        },
        "cascade": {
            "no_ecc_output_as_known_rate": false_known_rate(no_ecc, cascade),
            "no_ecc_specificity": float(np.mean(cascade[no_ecc] == "no_ecc")),
            "unknown_ecc_output_as_known_rate": false_known_rate(unknown, cascade),
            "unknown_ecc_gate_recall": float(np.mean(gate_passed[unknown])),
            "unknown_energy_rejection_rate": float(np.mean(energy_rejected[unknown])),
            "unknown_ecc_recall": float(np.mean(cascade[unknown] == "unknown_ecc")),
            "unknown_by_type": {
                label: {
                    "count": int(np.sum(categories == label)),
                    "output_as_known_rate": false_known_rate(categories == label, cascade),
                    "ecc_gate_recall": float(np.mean(gate_passed[categories == label])),
                    "energy_rejection_rate": float(np.mean(energy_rejected[categories == label])),
                    "unknown_recall": float(np.mean(cascade[categories == label] == "unknown_ecc")),
                }
                for label in sorted(set(str(value) for value in categories[unknown]))
            },
            "known_ecc_acceptance_rate": float(np.mean(accepted_known[known])),
            "known_type_macro_f1_end_to_end": cascade_known_e2e_f1,
            "known_type_macro_f1_conditional_accepted": conditional_f1,
            "known_type_macro_f1_change_end_to_end": cascade_known_e2e_f1 - closed_known_f1,
            "labels": list(END_TO_END_LABELS),
            "end_to_end_confusion_matrix": cascade_matrix.tolist(),
        },
        "improvement": {
            "no_ecc_false_known_rate_reduction": false_known_rate(no_ecc, closed)
            - false_known_rate(no_ecc, cascade),
            "unknown_ecc_false_known_rate_reduction": false_known_rate(unknown, closed)
            - false_known_rate(unknown, cascade),
        },
    }


def save_comparisons(
    comparisons: Sequence[SharedLogitComparison],
    output_csv: str | Path,
    output_json: str | Path,
) -> None:
    rows = [item.to_dict() for item in comparisons]
    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    output_json = Path(output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(
        json.dumps(summarize_comparisons(comparisons), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
