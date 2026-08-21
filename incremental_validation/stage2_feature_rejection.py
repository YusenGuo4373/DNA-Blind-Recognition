from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any, Sequence
import argparse
import csv
import json
import platform
import sys

import numpy as np

from hierarchical_ecc.coding import stable_seed
from .comparison import END_TO_END_LABELS, KNOWN_TYPES, NO_ECC_TYPES, IncrementalThresholds


INNER_CODES = ("HEDGES", "DNA-Aeon")
STAT7 = ("mean", "std", "q05", "q25", "median", "q75", "q95")
STAT5 = ("mean", "std", "q05", "median", "q95")


@dataclass(frozen=True)
class FeatureBundle:
    features: np.ndarray
    feature_names: tuple[str, ...]
    energy: np.ndarray
    max_soft_vote_probability: np.ndarray
    max_mean_logit: np.ndarray
    closed_index: np.ndarray


def _quantile_stats(values: np.ndarray, include_quartiles: bool) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    quantiles = (0.05, 0.25, 0.5, 0.75, 0.95) if include_quartiles else (0.05, 0.5, 0.95)
    standard_deviation = (
        values.std(axis=1, ddof=1, keepdims=True)
        if values.shape[1] > 1
        else np.zeros((values.shape[0], 1), dtype=np.float64)
    )
    return np.concatenate(
        (
            values.mean(axis=1, keepdims=True),
            standard_deviation,
            np.quantile(values, quantiles, axis=1).T,
        ),
        axis=1,
    )


def extract_logit_features(logits: np.ndarray) -> FeatureBundle:
    """Extract the preregistered 43 archive features from [N,M,q,4] logits."""

    logits = np.asarray(logits, dtype=np.float64)
    if logits.ndim != 4 or logits.shape[-1] != 4 or min(logits.shape[:3]) <= 0:
        raise ValueError("logits must have shape [N,M,q,4]")
    reads = logits.reshape(logits.shape[0], -1, 4)
    logit_parts = [_quantile_stats(reads[..., index], True) for index in range(4)]
    feature_names = tuple(
        f"logit_{KNOWN_TYPES[index]}_{stat}" for index in range(4) for stat in STAT7
    )

    maximum = reads.max(axis=-1, keepdims=True)
    logsumexp = maximum[..., 0] + np.log(np.exp(reads - maximum).sum(axis=-1))
    energy_reads = -logsumexp
    probabilities = np.exp(reads - maximum)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    entropy_reads = -(probabilities * np.log(np.clip(probabilities, 1e-15, 1.0))).sum(axis=-1)
    ordered = np.sort(probabilities, axis=-1)
    margin_reads = ordered[..., -1] - ordered[..., -2]
    scalar_parts = [
        _quantile_stats(energy_reads, False),
        _quantile_stats(entropy_reads, False),
        _quantile_stats(margin_reads, False),
    ]
    feature_names += tuple(f"energy_{stat}" for stat in STAT5)
    feature_names += tuple(f"softmax_entropy_{stat}" for stat in STAT5)
    feature_names += tuple(f"top1_top2_margin_{stat}" for stat in STAT5)
    features = np.concatenate((*logit_parts, *scalar_parts), axis=1)

    read_probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
    read_probabilities /= read_probabilities.sum(axis=-1, keepdims=True)
    archive_probabilities = read_probabilities.mean(axis=2).mean(axis=1)
    mean_logits = logits.mean(axis=2).mean(axis=1)
    return FeatureBundle(
        features=features,
        feature_names=feature_names,
        energy=energy_reads.mean(axis=1),
        max_soft_vote_probability=archive_probabilities.max(axis=1),
        max_mean_logit=mean_logits.max(axis=1),
        closed_index=archive_probabilities.argmax(axis=1).astype(np.int64),
    )


def split_known_archives(
    categories: Sequence[str], archive_ids: Sequence[str], seed: int
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    categories = np.asarray(categories, dtype=str)
    archive_ids = np.asarray(archive_ids, dtype=str)
    if categories.shape != archive_ids.shape:
        raise ValueError("categories and archive_ids must align")
    unsupported = sorted(set(categories) - set(KNOWN_TYPES))
    if unsupported:
        raise ValueError(f"known-only split received unsupported categories: {unsupported}")
    fit: list[int] = []
    calibration: list[int] = []
    audit: dict[str, Any] = {}
    for category in KNOWN_TYPES:
        indices = np.flatnonzero(categories == category)
        if indices.size < 4:
            raise ValueError(f"{category} needs at least four archives")
        ordered = sorted(
            indices.tolist(),
            key=lambda index: (
                stable_seed("stage2-logit-split", int(seed), category, archive_ids[index]),
                archive_ids[index],
            ),
        )
        midpoint = len(ordered) // 2
        fit.extend(ordered[:midpoint])
        calibration.extend(ordered[midpoint:])
        audit[category] = {
            "fit_count": midpoint,
            "calibration_count": len(ordered) - midpoint,
            "fit_archive_ids": [archive_ids[index] for index in ordered[:midpoint]],
            "calibration_archive_ids": [archive_ids[index] for index in ordered[midpoint:]],
        }
    fit_array = np.asarray(sorted(fit), dtype=np.int64)
    calibration_array = np.asarray(sorted(calibration), dtype=np.int64)
    if set(archive_ids[fit_array]) & set(archive_ids[calibration_array]):
        raise RuntimeError("stage2 fit/calibration archive overlap")
    return fit_array, calibration_array, audit


def _oas_covariance(values: np.ndarray) -> tuple[np.ndarray, float, float]:
    values = np.asarray(values, dtype=np.float64)
    centered = values - values.mean(axis=0, keepdims=True)
    covariance = centered.T @ centered / max(values.shape[0], 1)
    dimension = covariance.shape[0]
    trace = float(np.trace(covariance))
    trace_square = float(np.sum(covariance * covariance))
    mu = trace / max(dimension, 1)
    denominator = (values.shape[0] + 1.0 - 2.0 / dimension) * (
        trace_square - trace * trace / dimension
    )
    numerator = (1.0 - 2.0 / dimension) * trace_square + trace * trace
    alpha = 1.0 if denominator <= 0 else float(np.clip(numerator / denominator, 0.0, 1.0))
    shrunk = (1.0 - alpha) * covariance + alpha * mu * np.eye(dimension)
    ridge = max(mu, 1.0) * 1e-8
    shrunk += ridge * np.eye(dimension)
    return shrunk, alpha, float(np.linalg.cond(shrunk))


@dataclass
class KnownFeatureModel:
    location: np.ndarray
    scale: np.ndarray
    class_means: np.ndarray
    diagonal_variances: np.ndarray
    shrink_inverse_covariances: np.ndarray
    shrinkage: np.ndarray
    condition_numbers: np.ndarray

    @classmethod
    def fit(cls, features: np.ndarray, categories: Sequence[str]) -> "KnownFeatureModel":
        features = np.asarray(features, dtype=np.float64)
        categories = np.asarray(categories, dtype=str)
        if features.ndim != 2 or categories.shape != (features.shape[0],):
            raise ValueError("fit features/categories do not align")
        if set(categories) != set(KNOWN_TYPES):
            raise ValueError("feature model requires all and only four known ECC classes")
        location = features.mean(axis=0)
        scale = features.std(axis=0, ddof=1)
        scale = np.maximum(scale, 1e-8)
        standardized = (features - location) / scale
        means: list[np.ndarray] = []
        diagonal: list[np.ndarray] = []
        inverses: list[np.ndarray] = []
        shrinkage: list[float] = []
        conditions: list[float] = []
        for category in KNOWN_TYPES:
            values = standardized[categories == category]
            means.append(values.mean(axis=0))
            variance = values.var(axis=0, ddof=1)
            diagonal.append(np.maximum(variance, 1e-4))
            covariance, alpha, condition = _oas_covariance(values)
            inverses.append(np.linalg.pinv(covariance, hermitian=True))
            shrinkage.append(alpha)
            conditions.append(condition)
        return cls(
            location,
            scale,
            np.asarray(means),
            np.asarray(diagonal),
            np.asarray(inverses),
            np.asarray(shrinkage),
            np.asarray(conditions),
        )

    def standardized(self, features: np.ndarray) -> np.ndarray:
        return (np.asarray(features, dtype=np.float64) - self.location) / self.scale

    def diagonal_distances(self, features: np.ndarray, batch_size: int = 16384) -> np.ndarray:
        values = self.standardized(features)
        result = np.empty((values.shape[0], 4), dtype=np.float64)
        for start in range(0, values.shape[0], batch_size):
            stop = min(start + batch_size, values.shape[0])
            delta = values[start:stop, None, :] - self.class_means[None, :, :]
            result[start:stop] = np.sqrt(
                np.sum(delta * delta / self.diagonal_variances[None, :, :], axis=-1)
            )
        return result

    def shrinkage_distances(self, features: np.ndarray, batch_size: int = 4096) -> np.ndarray:
        values = self.standardized(features)
        result = np.empty((values.shape[0], 4), dtype=np.float64)
        for start in range(0, values.shape[0], batch_size):
            stop = min(start + batch_size, values.shape[0])
            delta = values[start:stop, None, :] - self.class_means[None, :, :]
            squared = np.einsum(
                "nkd,kde,nke->nk", delta, self.shrink_inverse_covariances, delta
            )
            result[start:stop] = np.sqrt(np.maximum(squared, 0.0))
        return result


def select_class_score(distances: np.ndarray, closed_index: np.ndarray, mode: str) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float64)
    closed_index = np.asarray(closed_index, dtype=np.int64)
    if distances.ndim != 2 or distances.shape[1] != 4:
        raise ValueError("distances must have shape [N,4]")
    if closed_index.shape != (distances.shape[0],):
        raise ValueError("closed_index must align with distances")
    if mode == "predicted":
        return distances[np.arange(distances.shape[0]), closed_index]
    if mode == "minimum":
        return distances.min(axis=1)
    raise ValueError(f"unsupported class distance mode: {mode}")


def conformal_p_values(
    distances: np.ndarray,
    calibration_distances_by_class: Sequence[np.ndarray],
) -> np.ndarray:
    distances = np.asarray(distances, dtype=np.float64)
    if distances.ndim != 2 or distances.shape[1] != 4:
        raise ValueError("distances must have shape [N,4]")
    result = np.empty_like(distances)
    for class_index, calibration in enumerate(calibration_distances_by_class):
        calibration = np.asarray(calibration, dtype=np.float64).reshape(-1)
        if calibration.size == 0:
            raise ValueError("each conformal class needs calibration scores")
        result[:, class_index] = (
            1.0 + (calibration[None, :] >= distances[:, class_index, None]).sum(axis=1)
        ) / (calibration.size + 1.0)
    return result


def leave_one_out_conformal_scores(
    distances: np.ndarray,
    categories: Sequence[str],
    closed_index: np.ndarray,
    mode: str,
) -> np.ndarray:
    """Return rejection scores (-p) for threshold calibration without self-counting."""

    distances = np.asarray(distances, dtype=np.float64)
    categories = np.asarray(categories, dtype=str)
    closed_index = np.asarray(closed_index, dtype=np.int64)
    scores = np.empty(distances.shape[0], dtype=np.float64)
    for row in range(distances.shape[0]):
        pvalues = np.empty(4, dtype=np.float64)
        for class_index, category in enumerate(KNOWN_TYPES):
            mask = categories == category
            reference = distances[mask, class_index]
            if categories[row] == category:
                own = np.flatnonzero(mask)
                own_position = int(np.flatnonzero(own == row)[0])
                reference = np.delete(reference, own_position)
            pvalues[class_index] = (
                1.0 + np.sum(reference >= distances[row, class_index])
            ) / (reference.size + 1.0)
        if mode == "predicted":
            scores[row] = -pvalues[closed_index[row]]
        elif mode == "maximum":
            scores[row] = -pvalues.max()
        else:
            raise ValueError(f"unsupported conformal mode: {mode}")
    return scores


def acceptance_threshold(rejection_scores: np.ndarray, target: float = 0.95) -> float:
    scores = np.sort(np.asarray(rejection_scores, dtype=np.float64).reshape(-1))
    if scores.size == 0 or not 0.0 < target <= 1.0:
        raise ValueError("non-empty scores and target in (0,1] required")
    rank = max(0, int(np.ceil(target * scores.size)) - 1)
    return float(scores[rank])


def _auroc(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    return float(
        (positive[:, None] > negative[None, :]).mean()
        + 0.5 * (positive[:, None] == negative[None, :]).mean()
    )


def _average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=np.bool_)
    order = np.argsort(-np.asarray(scores, dtype=np.float64), kind="stable")
    ordered = labels[order]
    positives = int(ordered.sum())
    if positives == 0:
        return float("nan")
    precision = np.cumsum(ordered) / np.arange(1, ordered.size + 1)
    return float(precision[ordered].sum() / positives)


def _fpr_at_95_tpr(positive: np.ndarray, negative: np.ndarray) -> float:
    positive = np.asarray(positive, dtype=np.float64)
    negative = np.asarray(negative, dtype=np.float64)
    candidates = np.unique(np.concatenate((positive, negative)))
    best = 1.0
    found = False
    for threshold in candidates:
        tpr = float(np.mean(positive >= threshold))
        if tpr >= 0.95:
            best = min(best, float(np.mean(negative >= threshold)))
            found = True
    return best if found else 1.0


def _macro_f1(truth: np.ndarray, prediction: np.ndarray) -> float:
    scores: list[float] = []
    for label in KNOWN_TYPES:
        tp = int(np.sum((truth == label) & (prediction == label)))
        fp = int(np.sum((truth != label) & (prediction == label)))
        fn = int(np.sum((truth == label) & (prediction != label)))
        denominator = 2 * tp + fp + fn
        scores.append(0.0 if denominator == 0 else 2 * tp / denominator)
    return float(np.mean(scores))


def _truth(category: str) -> str:
    if category in NO_ECC_TYPES:
        return "no_ecc"
    if category in INNER_CODES:
        return "unknown_ecc"
    if category in KNOWN_TYPES:
        return category
    raise ValueError(category)


def _confusion(truth: np.ndarray, prediction: np.ndarray) -> list[list[int]]:
    index = {label: position for position, label in enumerate(END_TO_END_LABELS)}
    matrix = np.zeros((len(index), len(index)), dtype=np.int64)
    for expected, observed in zip(truth, prediction):
        matrix[index[str(expected)], index[str(observed)]] += 1
    return matrix.tolist()


def evaluate_method(
    categories: np.ndarray,
    ecc_scores: np.ndarray,
    closed_index: np.ndarray,
    rejection_scores: np.ndarray,
    threshold: float,
    tau1: float,
    closed_macro_f1: float,
) -> tuple[dict[str, Any], np.ndarray]:
    categories = np.asarray(categories, dtype=str)
    rejection_scores = np.asarray(rejection_scores, dtype=np.float64)
    closed_labels = np.asarray(KNOWN_TYPES, dtype=object)[closed_index]
    outputs = closed_labels.copy()
    outputs[rejection_scores > threshold] = "unknown_ecc"
    outputs[ecc_scores < tau1] = "no_ecc"
    known = np.isin(categories, KNOWN_TYPES)
    unknown = np.isin(categories, INNER_CODES)
    truth = np.asarray([_truth(category) for category in categories], dtype=object)
    known_prediction = outputs[known]
    macro_f1 = _macro_f1(categories[known], known_prediction)
    binary_labels = unknown[np.logical_or(known, unknown)]
    binary_scores = rejection_scores[np.logical_or(known, unknown)]
    metrics = {
        "threshold": float(threshold),
        "known_acceptance_rate": float(np.mean(np.isin(outputs[known], KNOWN_TYPES))),
        "HEDGES_unknown_recall": float(np.mean(outputs[categories == "HEDGES"] == "unknown_ecc")),
        "DNA_Aeon_unknown_recall": float(
            np.mean(outputs[categories == "DNA-Aeon"] == "unknown_ecc")
        ),
        "combined_unknown_recall": float(np.mean(outputs[unknown] == "unknown_ecc")),
        "unknown_output_as_known_rate": float(np.mean(np.isin(outputs[unknown], KNOWN_TYPES))),
        "unknown_misclassified_as_BCH_rate": float(np.mean(outputs[unknown] == "BCH")),
        "known_type_macro_f1": macro_f1,
        "known_type_macro_f1_change_from_closed": macro_f1 - closed_macro_f1,
        "AUROC": _auroc(rejection_scores[unknown], rejection_scores[known]),
        "AUPR": _average_precision(binary_labels, binary_scores),
        "FPR_at_95_TPR": _fpr_at_95_tpr(rejection_scores[unknown], rejection_scores[known]),
        "labels": list(END_TO_END_LABELS),
        "end_to_end_confusion_matrix": _confusion(truth, outputs),
    }
    metrics["success"] = bool(
        metrics["known_acceptance_rate"] >= 0.93
        and metrics["combined_unknown_recall"] >= 0.70
        and metrics["unknown_output_as_known_rate"] <= 0.30
        and metrics["known_type_macro_f1_change_from_closed"] >= -0.02
    )
    return metrics, outputs.astype(str)


def _sha256(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _distribution(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(values.mean()),
        "std": float(values.std(ddof=1)) if values.size > 1 else 0.0,
        "q05": float(np.quantile(values, 0.05)),
        "median": float(np.quantile(values, 0.5)),
        "q95": float(np.quantile(values, 0.95)),
    }


def _write_score_svg(
    path: Path,
    method_scores: dict[str, np.ndarray],
    thresholds: dict[str, float],
    categories: np.ndarray,
) -> None:
    methods = list(method_scores)
    width = 1100
    row_height = 72
    height = 70 + row_height * len(methods)
    left, right = 250, 40
    colors = {"Known ECC": "#1f77b4", "HEDGES": "#d62728", "DNA-Aeon": "#9467bd"}
    groups = {
        "Known ECC": np.isin(categories, KNOWN_TYPES),
        "HEDGES": categories == "HEDGES",
        "DNA-Aeon": categories == "DNA-Aeon",
    }
    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="550" y="25" text-anchor="middle" font-family="sans-serif" font-size="17">Stage-2 rejection-score distributions (threshold-normalized)</text>',
    ]
    for row, method in enumerate(methods):
        score = method_scores[method]
        threshold = thresholds[method]
        known_std = max(float(score[groups["Known ECC"]].std(ddof=1)), 1e-8)
        normalized = np.clip((score - threshold) / known_std, -5.0, 5.0)
        y = 55 + row * row_height
        svg.append(f'<text x="10" y="{y+20}" font-family="sans-serif" font-size="12">{method}</text>')
        svg.append(f'<line x1="{left}" y1="{y+20}" x2="{width-right}" y2="{y+20}" stroke="#cccccc"/>')
        zero_x = left + (5.0 / 10.0) * (width - left - right)
        svg.append(f'<line x1="{zero_x}" y1="{y}" x2="{zero_x}" y2="{y+42}" stroke="black" stroke-dasharray="3,3"/>')
        for group_index, (group, mask) in enumerate(groups.items()):
            values = normalized[mask]
            q05, median, q95 = np.quantile(values, (0.05, 0.5, 0.95))
            scale = (width - left - right) / 10.0
            x05, xm, x95 = left + (q05 + 5) * scale, left + (median + 5) * scale, left + (q95 + 5) * scale
            gy = y + 8 + group_index * 13
            svg.append(f'<line x1="{x05:.1f}" y1="{gy}" x2="{x95:.1f}" y2="{gy}" stroke="{colors[group]}" stroke-width="3"/>')
            svg.append(f'<circle cx="{xm:.1f}" cy="{gy}" r="4" fill="{colors[group]}"/>')
        if row == 0:
            for index, group in enumerate(groups):
                svg.append(f'<text x="{left+index*160}" y="45" fill="{colors[group]}" font-family="sans-serif" font-size="11">{group}</text>')
    svg.append(f'<text x="{(left+width-right)/2}" y="{height-8}" text-anchor="middle" font-family="sans-serif" font-size="12">(score - calibrated threshold) / known-score SD; positive means rejected</text>')
    svg.append("</svg>")
    path.write_text("\n".join(svg), encoding="utf-8")


def run_stage1(
    source: str | Path,
    output: str | Path,
    seed: int = 42,
    known_acceptance: float = 0.95,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    source = Path(source).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    calibration_path = source / "calibration_shared_logits.npz"
    test_path = source / "test_shared_logits.npz"
    threshold_path = source / "thresholds.json"
    calibration_payload = np.load(calibration_path, allow_pickle=False)
    calibration_categories_all = calibration_payload["categories"].astype(str)
    calibration_ids_all = calibration_payload["archive_ids"].astype(str)
    known_mask = np.isin(calibration_categories_all, KNOWN_TYPES)
    known_categories = calibration_categories_all[known_mask]
    known_ids = calibration_ids_all[known_mask]
    known_logits = calibration_payload["type_logits"][known_mask]
    if set(known_categories) != set(KNOWN_TYPES):
        raise RuntimeError("calibration does not contain all four known classes")
    fit_indices, threshold_indices, split_audit = split_known_archives(
        known_categories, known_ids, seed
    )
    known_bundle = extract_logit_features(known_logits)
    model = KnownFeatureModel.fit(
        known_bundle.features[fit_indices], known_categories[fit_indices]
    )
    diagonal_cal = model.diagonal_distances(known_bundle.features[threshold_indices])
    shrink_cal = model.shrinkage_distances(known_bundle.features[threshold_indices])
    cal_categories = known_categories[threshold_indices]
    cal_closed = known_bundle.closed_index[threshold_indices]

    method_cal_scores: dict[str, np.ndarray] = {
        "max_soft_vote_probability": -known_bundle.max_soft_vote_probability[threshold_indices],
        "max_mean_logit": -known_bundle.max_mean_logit[threshold_indices],
        "diagonal_predicted_class": select_class_score(diagonal_cal, cal_closed, "predicted"),
        "diagonal_minimum_class": select_class_score(diagonal_cal, cal_closed, "minimum"),
        "shrinkage_predicted_class": select_class_score(shrink_cal, cal_closed, "predicted"),
        "shrinkage_minimum_class": select_class_score(shrink_cal, cal_closed, "minimum"),
    }
    calibration_by_class = [
        shrink_cal[cal_categories == category, class_index]
        for class_index, category in enumerate(KNOWN_TYPES)
    ]
    method_cal_scores["conformal_predicted_class"] = leave_one_out_conformal_scores(
        shrink_cal, cal_categories, cal_closed, "predicted"
    )
    method_cal_scores["conformal_maximum_pvalue"] = leave_one_out_conformal_scores(
        shrink_cal, cal_categories, cal_closed, "maximum"
    )
    method_thresholds = {
        method: acceptance_threshold(scores, known_acceptance)
        for method, scores in method_cal_scores.items()
    }
    original_thresholds = IncrementalThresholds.load(threshold_path)
    method_thresholds = {
        "global_energy_original": original_thresholds.unknown_energy,
        **method_thresholds,
    }

    test_payload = np.load(test_path, allow_pickle=False)
    test_categories = test_payload["categories"].astype(str)
    test_ids = test_payload["archive_ids"].astype(str)
    test_presence = np.asarray(test_payload["presence_probabilities"], dtype=np.float64)
    test_bundle = extract_logit_features(test_payload["type_logits"])
    diagonal_test = model.diagonal_distances(test_bundle.features)
    shrink_test = model.shrinkage_distances(test_bundle.features)
    pvalues = conformal_p_values(shrink_test, calibration_by_class)
    method_scores: dict[str, np.ndarray] = {
        "global_energy_original": test_bundle.energy,
        "max_soft_vote_probability": -test_bundle.max_soft_vote_probability,
        "max_mean_logit": -test_bundle.max_mean_logit,
        "diagonal_predicted_class": select_class_score(
            diagonal_test, test_bundle.closed_index, "predicted"
        ),
        "diagonal_minimum_class": select_class_score(
            diagonal_test, test_bundle.closed_index, "minimum"
        ),
        "shrinkage_predicted_class": select_class_score(
            shrink_test, test_bundle.closed_index, "predicted"
        ),
        "shrinkage_minimum_class": select_class_score(
            shrink_test, test_bundle.closed_index, "minimum"
        ),
        "conformal_predicted_class": -pvalues[
            np.arange(pvalues.shape[0]), test_bundle.closed_index
        ],
        "conformal_maximum_pvalue": -pvalues.max(axis=1),
    }
    ecc_scores = test_presence.mean(axis=2).mean(axis=1)
    known_test = np.isin(test_categories, KNOWN_TYPES)
    closed_labels = np.asarray(KNOWN_TYPES, dtype=object)[test_bundle.closed_index]
    closed_macro_f1 = _macro_f1(test_categories[known_test], closed_labels[known_test])
    metrics: dict[str, Any] = {}
    outputs_by_method: dict[str, np.ndarray] = {}
    for method, scores in method_scores.items():
        metrics[method], outputs_by_method[method] = evaluate_method(
            test_categories,
            ecc_scores,
            test_bundle.closed_index,
            scores,
            method_thresholds[method],
            original_thresholds.ecc_presence,
            closed_macro_f1,
        )

    feature_definitions = {
        "feature_count": len(test_bundle.feature_names),
        "feature_names": list(test_bundle.feature_names),
        "logit_statistics": list(STAT7),
        "energy_entropy_margin_statistics": list(STAT5),
        "aggregation": "flatten M*q reads within each archive, then compute fixed statistics",
        "controls": ["global_energy_original", "max_soft_vote_probability", "max_mean_logit"],
        "author_core_modified": False,
    }
    (output / "logit_feature_definitions.json").write_text(
        json.dumps(feature_definitions, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    calibration_report = {
        "seed": seed,
        "known_acceptance_target": known_acceptance,
        "fit_and_calibration_classes": list(KNOWN_TYPES),
        "forbidden_classes_absent_from_fit_and_threshold_selection": list(INNER_CODES),
        "no_ecc_absent_from_fit_and_threshold_selection": list(NO_ECC_TYPES),
        "split": split_audit,
        "thresholds": method_thresholds,
        "calibration_actual_acceptance": {
            method: float(np.mean(scores <= method_thresholds[method]))
            for method, scores in method_cal_scores.items()
        },
        "original_energy_threshold_source": str(threshold_path),
        "covariance": {
            category: {
                "oas_shrinkage": float(model.shrinkage[index]),
                "condition_number": float(model.condition_numbers[index]),
                "ridge_relative_floor": 1e-8,
            }
            for index, category in enumerate(KNOWN_TYPES)
        },
        "conformal_calibration_count_per_class": {
            category: int(calibration_by_class[index].size)
            for index, category in enumerate(KNOWN_TYPES)
        },
    }
    (output / "stage2_calibration.json").write_text(
        json.dumps(calibration_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    metric_report = {
        "protocol": "frozen_author_core_external_logit_feature_open_set_validation",
        "closed_set_known_type_macro_f1": closed_macro_f1,
        "methods": metrics,
        "any_method_meets_all_success_criteria": any(
            value["success"] for value in metrics.values()
        ),
        "success_criteria": {
            "known_acceptance_minimum": 0.93,
            "combined_unknown_recall_minimum": 0.70,
            "unknown_output_as_known_maximum": 0.30,
            "known_macro_f1_drop_maximum": 0.02,
        },
    }
    (output / "stage2_metrics.json").write_text(
        json.dumps(metric_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    prediction_fields = [
        "archive_id", "category", "method", "closed_set_output", "ecc_score", "tau1",
        "rejection_score", "threshold", "stage2_rejected", "cascade_output",
        "code_rate", "code_length", *[f"conformal_p_{label}" for label in KNOWN_TYPES],
    ]
    with (output / "stage2_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=prediction_fields)
        writer.writeheader()
        for method, scores in method_scores.items():
            for index, archive_id in enumerate(test_ids):
                writer.writerow({
                    "archive_id": archive_id,
                    "category": test_categories[index],
                    "method": method,
                    "closed_set_output": closed_labels[index],
                    "ecc_score": ecc_scores[index],
                    "tau1": original_thresholds.ecc_presence,
                    "rejection_score": scores[index],
                    "threshold": method_thresholds[method],
                    "stage2_rejected": bool(scores[index] > method_thresholds[method]),
                    "cascade_output": outputs_by_method[method][index],
                    "code_rate": "null",
                    "code_length": "null",
                    **{
                        f"conformal_p_{label}": pvalues[index, class_index]
                        for class_index, label in enumerate(KNOWN_TYPES)
                    },
                })

    with (output / "stage2_score_distributions.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        fields = ("method", "category", "count", "threshold", "mean", "std", "q05", "median", "q95")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, scores in method_scores.items():
            for category in (*KNOWN_TYPES, *NO_ECC_TYPES, *INNER_CODES):
                values = scores[test_categories == category]
                if values.size == 0:
                    continue
                writer.writerow({
                    "method": method, "category": category, "count": values.size,
                    "threshold": method_thresholds[method], **_distribution(values),
                })
    _write_score_svg(
        output / "stage2_score_distributions.svg",
        method_scores,
        method_thresholds,
        test_categories,
    )
    audit = {
        "source_directory": str(source),
        "output_directory": str(output),
        "source_hashes": {
            path.name: _sha256(path)
            for path in (calibration_path, test_path, threshold_path)
        },
        "fit_categories": sorted(set(known_categories[fit_indices])),
        "threshold_calibration_categories": sorted(set(known_categories[threshold_indices])),
        "test_categories": sorted(set(test_categories)),
        "fit_calibration_archive_overlap": 0,
        "calibration_final_test_archive_id_overlap": len(set(known_ids) & set(test_ids)),
        "unknown_used_for_fit": False,
        "unknown_used_for_feature_selection": False,
        "unknown_used_for_threshold_selection": False,
        "python_hash_used": False,
        "stable_split_seed": seed,
        "command": list(command or []),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": np.__version__,
        },
    }
    (output / "stage2_run_audit.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metric_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Known-only logit-feature stage-2 rejection")
    parser.add_argument("--source", default="outputs/inner_codes_formal_seed42")
    parser.add_argument("--output", default="outputs/stage2_feature_rejection_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--known-acceptance", type=float, default=0.95)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    command = [sys.executable, "-m", "incremental_validation.stage2_feature_rejection", *(argv or sys.argv[1:])]
    report = run_stage1(
        args.source, args.output, args.seed, args.known_acceptance, command=command
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
