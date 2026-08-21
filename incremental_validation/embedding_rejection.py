from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence
import argparse
import csv
import json
import platform
import sys

import numpy as np
import torch

from author_baseline.recognizer import OneHotArchive, OriginalTaskModel
from author_baseline.weights import DEFAULT_WEIGHT_ROOT, EXPECTED_SHA256, build_primary_type_recognizer
from hierarchical_ecc.config import ExperimentConfig, KNOWN_CODE_TYPES, NO_ECC_TYPES
from hierarchical_ecc.data import ReferenceFactory
from .comparison import IncrementalThresholds, KNOWN_TYPES
from .inner_codes import UNKNOWN_INNER_CODES, archives_from_references, generate_inner_code_references
from .simulation import SimulationRunConfig, _make_archives, audit_molecular_references
from .stage2_feature_rejection import (
    INNER_CODES,
    KnownFeatureModel,
    _macro_f1,
    _sha256,
    _write_score_svg,
    acceptance_threshold,
    conformal_p_values,
    evaluate_method,
    leave_one_out_conformal_scores,
    select_class_score,
)


@torch.inference_mode()
def extract_archive_embeddings(
    task_model: OriginalTaskModel,
    archive: OneHotArchive,
) -> tuple[np.ndarray, np.ndarray]:
    """Capture the unchanged Transformer's fc input with an external pre-hook."""

    fc = getattr(task_model.model, "fc", None)
    if not isinstance(fc, torch.nn.Linear) or fc.in_features != 128:
        raise ValueError("author type model must expose a 128-input final fc layer")
    captured: list[torch.Tensor] = []

    def capture(_module: torch.nn.Module, inputs: tuple[torch.Tensor, ...]) -> None:
        if len(inputs) != 1 or inputs[0].ndim != 2 or inputs[0].shape[1] != 128:
            raise RuntimeError("unexpected author fc input")
        captured.append(inputs[0].detach().cpu().clone())

    handle = fc.register_forward_pre_hook(capture)
    try:
        logits = task_model.read_logits(archive).detach().cpu().numpy()
    finally:
        handle.remove()
    molecules, reads, _ = archive.validate()
    embeddings = torch.cat(captured, dim=0).numpy().reshape(molecules, reads, 128)
    return logits, embeddings


def fit_pca(read_embeddings: np.ndarray, dimensions: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(read_embeddings, dtype=np.float64).reshape(-1, 128)
    if not 0 < dimensions <= 128:
        raise ValueError("PCA dimensions must be in [1,128]")
    mean = values.mean(axis=0)
    centered = values - mean
    covariance = centered.T @ centered / max(values.shape[0] - 1, 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    selected = order[:dimensions]
    return mean, eigenvectors[:, selected].T, eigenvalues[order]


def transform_embeddings(
    embeddings: np.ndarray,
    mean: np.ndarray | None,
    components: np.ndarray | None,
) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    if components is None:
        return values
    return np.matmul(values - mean, components.T)


def archive_mean_std(embeddings: np.ndarray) -> np.ndarray:
    values = np.asarray(embeddings, dtype=np.float64)
    if values.ndim != 4:
        raise ValueError("embeddings must have shape [N,M,q,D]")
    flattened = values.reshape(values.shape[0], -1, values.shape[-1])
    return np.concatenate((flattened.mean(axis=1), flattened.std(axis=1, ddof=1)), axis=1)


def aggregate_read_distances(distances: np.ndarray, shape: tuple[int, int, int]) -> np.ndarray:
    archives, molecules, reads = shape
    values = np.asarray(distances, dtype=np.float64)
    if values.shape != (archives * molecules * reads, 4):
        raise ValueError("read distances do not match N/M/q")
    return values.reshape(archives, molecules, reads, 4).mean(axis=2).mean(axis=1)


def _closed_indices(logits: np.ndarray) -> np.ndarray:
    values = np.asarray(logits, dtype=np.float64)
    probabilities = np.exp(values - values.max(axis=-1, keepdims=True))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    return probabilities.mean(axis=2).mean(axis=1).argmax(axis=1).astype(np.int64)


def _fingerprints(reference_sets: dict[str, np.ndarray]) -> set[bytes]:
    result: set[bytes] = set()
    for references in reference_sets.values():
        for reference in np.asarray(references, dtype=np.uint8):
            encoded = reference.tobytes()
            if encoded in result:
                raise RuntimeError("duplicate molecular reference within split")
            result.add(encoded)
    return result


def _known_reference_sets(
    experiment: ExperimentConfig,
    split: str,
    seed: int,
    archives_per_category: int,
    molecules: int,
    categories: Sequence[str],
) -> dict[str, np.ndarray]:
    factory = ReferenceFactory(experiment)
    namespace = f"{split}-seed-{seed}"
    return {
        category: np.stack(
            [
                factory.make_reference(category, namespace, archive_id, molecule_id)
                for archive_id in range(archives_per_category)
                for molecule_id in range(molecules)
            ]
        )
        for category in categories
    }


def _extract_many(
    task_model: OriginalTaskModel,
    archives: Sequence[OneHotArchive],
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    logits: list[np.ndarray] = []
    embeddings: list[np.ndarray] = []
    for index, archive in enumerate(archives):
        archive_logits, archive_embeddings = extract_archive_embeddings(task_model, archive)
        logits.append(archive_logits)
        embeddings.append(archive_embeddings)
        if (index + 1) % 10 == 0 or index + 1 == len(archives):
            print(f"embedding extraction {label}: {index + 1}/{len(archives)}", flush=True)
    return np.stack(logits), np.stack(embeddings)


def _calibrate_embedding_family(
    prefix: str,
    fit_values: np.ndarray,
    calibration_values: np.ndarray,
    test_values: np.ndarray,
    fit_categories: np.ndarray,
    calibration_categories: np.ndarray,
    calibration_closed: np.ndarray,
    test_closed: np.ndarray,
    known_acceptance: float,
) -> tuple[dict[str, np.ndarray], dict[str, float], dict[str, Any], dict[str, np.ndarray]]:
    model = KnownFeatureModel.fit(fit_values, fit_categories)
    diagonal_cal = model.diagonal_distances(calibration_values)
    diagonal_test = model.diagonal_distances(test_values)
    shrink_cal = model.shrinkage_distances(calibration_values)
    shrink_test = model.shrinkage_distances(test_values)
    calibration_by_class = [
        shrink_cal[calibration_categories == category, class_index]
        for class_index, category in enumerate(KNOWN_TYPES)
    ]
    pvalues = conformal_p_values(shrink_test, calibration_by_class)
    cal_scores = {
        f"{prefix}_diagonal_predicted": select_class_score(
            diagonal_cal, calibration_closed, "predicted"
        ),
        f"{prefix}_diagonal_minimum": select_class_score(
            diagonal_cal, calibration_closed, "minimum"
        ),
        f"{prefix}_shrinkage_predicted": select_class_score(
            shrink_cal, calibration_closed, "predicted"
        ),
        f"{prefix}_shrinkage_minimum": select_class_score(
            shrink_cal, calibration_closed, "minimum"
        ),
        f"{prefix}_conformal_predicted": leave_one_out_conformal_scores(
            shrink_cal, calibration_categories, calibration_closed, "predicted"
        ),
        f"{prefix}_conformal_maximum_pvalue": leave_one_out_conformal_scores(
            shrink_cal, calibration_categories, calibration_closed, "maximum"
        ),
    }
    scores = {
        f"{prefix}_diagonal_predicted": select_class_score(
            diagonal_test, test_closed, "predicted"
        ),
        f"{prefix}_diagonal_minimum": select_class_score(
            diagonal_test, test_closed, "minimum"
        ),
        f"{prefix}_shrinkage_predicted": select_class_score(
            shrink_test, test_closed, "predicted"
        ),
        f"{prefix}_shrinkage_minimum": select_class_score(
            shrink_test, test_closed, "minimum"
        ),
        f"{prefix}_conformal_predicted": -pvalues[
            np.arange(pvalues.shape[0]), test_closed
        ],
        f"{prefix}_conformal_maximum_pvalue": -pvalues.max(axis=1),
    }
    thresholds = {
        method: acceptance_threshold(values, known_acceptance)
        for method, values in cal_scores.items()
    }
    audit = {
        "fit_sample_count": int(fit_values.shape[0]),
        "feature_dimension": int(fit_values.shape[1]),
        "covariance": {
            category: {
                "oas_shrinkage": float(model.shrinkage[index]),
                "condition_number": float(model.condition_numbers[index]),
            }
            for index, category in enumerate(KNOWN_TYPES)
        },
        "calibration_actual_acceptance": {
            method: float(np.mean(values <= thresholds[method]))
            for method, values in cal_scores.items()
        },
    }
    pvalue_columns = {
        f"{prefix}_p_{category}": pvalues[:, index]
        for index, category in enumerate(KNOWN_TYPES)
    }
    return scores, thresholds, audit, pvalue_columns


def run_embedding_stage(
    source: str | Path,
    stage1_output: str | Path,
    output: str | Path,
    seed: int = 42,
    fit_archives_per_category: int = 20,
    calibration_archives_per_category: int = 20,
    known_acceptance: float = 0.95,
    weight_root: str | Path = DEFAULT_WEIGHT_ROOT,
    device: str | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    source = Path(source).resolve()
    stage1_output = Path(stage1_output).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    device_value = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    experiment = ExperimentConfig()
    run = SimulationRunConfig(
        seed=seed,
        test_archives_per_category=50,
        molecules=20,
        reads_per_molecule=50,
        test_error_rate=0.05,
    )
    models_path = Path("vendor/zhouph0313_DNA/models.py").resolve()
    checkpoint_path = Path(weight_root).resolve() / "type" / "transformer_model_f10.6033.pt"
    models_hash_before = _sha256(models_path)
    weight_hash_before = _sha256(checkpoint_path)
    recognizer = build_primary_type_recognizer(
        weight_root=weight_root, device=device_value, batch_size=64
    )
    task_model = recognizer.code_type
    for parameter in task_model.model.parameters():
        parameter.requires_grad_(False)

    fit_archives, fit_categories_list, fit_ids = _make_archives(
        experiment, "embedding-fit", KNOWN_CODE_TYPES, fit_archives_per_category, run
    )
    calibration_archives, calibration_categories_list, calibration_ids = _make_archives(
        experiment,
        "embedding-calibration",
        KNOWN_CODE_TYPES,
        calibration_archives_per_category,
        run,
    )
    test_categories_base = KNOWN_CODE_TYPES + NO_ECC_TYPES
    test_archives, test_categories_list, test_ids = _make_archives(
        experiment, "test", test_categories_base, 50, run
    )

    reference_sets_by_split: dict[str, dict[str, np.ndarray]] = {
        "stage2-fit": _known_reference_sets(
            experiment, "embedding-fit", seed, fit_archives_per_category, 20, KNOWN_CODE_TYPES
        ),
        "stage2-calibration": _known_reference_sets(
            experiment,
            "embedding-calibration",
            seed,
            calibration_archives_per_category,
            20,
            KNOWN_CODE_TYPES,
        ),
        "final-test": _known_reference_sets(
            experiment, "test", seed, 50, 20, test_categories_base
        ),
    }
    inner_validations: list[dict[str, Any]] = []
    for category in UNKNOWN_INNER_CODES:
        references, validation = generate_inner_code_references(
            category,
            50 * 20,
            seed,
            output / "references" / f"{category.lower().replace('-', '_')}.fasta",
            namespace="final-test",
        )
        reference_sets_by_split["final-test"][category] = references
        inner_validations.append(asdict(validation))
        test_archives.extend(
            archives_from_references(
                experiment, category, f"test-seed-{seed}", references, 50, 20, 50, 0.05
            )
        )
        test_categories_list.extend([category] * 50)
        test_ids.extend([f"test:{category}:{archive_id}" for archive_id in range(50)])

    fingerprints_by_split = {
        split: _fingerprints(reference_sets)
        for split, reference_sets in reference_sets_by_split.items()
    }
    for left, right in (
        ("stage2-fit", "stage2-calibration"),
        ("stage2-fit", "final-test"),
        ("stage2-calibration", "final-test"),
    ):
        if fingerprints_by_split[left] & fingerprints_by_split[right]:
            raise RuntimeError(f"molecular reference overlap: {left}/{right}")
    reference_audits = {
        split: audit_molecular_references(reference_sets)
        for split, reference_sets in reference_sets_by_split.items()
    }

    baseline_logits = task_model.read_logits(fit_archives[0]).detach().cpu().numpy()
    hooked_logits, _ = extract_archive_embeddings(task_model, fit_archives[0])
    if not np.array_equal(baseline_logits, hooked_logits):
        raise RuntimeError("fc pre-hook changed author logits")
    fit_logits, fit_embeddings = _extract_many(task_model, fit_archives, "stage2-fit")
    calibration_logits, calibration_embeddings = _extract_many(
        task_model, calibration_archives, "stage2-calibration"
    )
    test_logits, test_embeddings = _extract_many(task_model, test_archives, "final-test")

    source_test = np.load(source / "test_shared_logits.npz", allow_pickle=False)
    source_ids = source_test["archive_ids"].astype(str)
    if not np.array_equal(source_ids, np.asarray(test_ids, dtype=str)):
        raise RuntimeError("regenerated final-test archive IDs differ from formal source")
    source_logits = np.asarray(source_test["type_logits"], dtype=np.float32)
    source_max_abs_difference = float(np.max(np.abs(source_logits - test_logits)))
    source_logits_allclose = bool(np.allclose(source_logits, test_logits, atol=1e-6, rtol=1e-6))
    if not source_logits_allclose:
        raise RuntimeError("regenerated final-test logits differ from formal source")

    np.savez_compressed(
        output / "embedding_features.npz",
        fit_categories=np.asarray(fit_categories_list, dtype=str),
        fit_archive_ids=np.asarray(fit_ids, dtype=str),
        fit_embeddings=fit_embeddings.astype(np.float32),
        fit_logits=fit_logits.astype(np.float32),
        calibration_categories=np.asarray(calibration_categories_list, dtype=str),
        calibration_archive_ids=np.asarray(calibration_ids, dtype=str),
        calibration_embeddings=calibration_embeddings.astype(np.float32),
        calibration_logits=calibration_logits.astype(np.float32),
        test_categories=np.asarray(test_categories_list, dtype=str),
        test_archive_ids=np.asarray(test_ids, dtype=str),
        test_embeddings=test_embeddings.astype(np.float32),
        test_logits=test_logits.astype(np.float32),
    )

    fit_categories = np.asarray(fit_categories_list, dtype=str)
    calibration_categories = np.asarray(calibration_categories_list, dtype=str)
    test_categories = np.asarray(test_categories_list, dtype=str)
    fit_closed = _closed_indices(fit_logits)
    calibration_closed = _closed_indices(calibration_logits)
    test_closed = _closed_indices(test_logits)
    source_presence = np.asarray(source_test["presence_probabilities"], dtype=np.float64)
    ecc_scores = source_presence.mean(axis=2).mean(axis=1)
    thresholds_stage1 = IncrementalThresholds.load(source / "thresholds.json")
    known_test = np.isin(test_categories, KNOWN_TYPES)
    closed_labels = np.asarray(KNOWN_TYPES, dtype=object)[test_closed]
    closed_macro_f1 = _macro_f1(test_categories[known_test], closed_labels[known_test])

    pca_models: dict[str, tuple[np.ndarray | None, np.ndarray | None]] = {
        "raw128": (None, None)
    }
    pca_audit: dict[str, Any] = {}
    for dimensions in (16, 32):
        mean, components, eigenvalues = fit_pca(fit_embeddings, dimensions)
        name = f"pca{dimensions}"
        pca_models[name] = (mean, components)
        pca_audit[name] = {
            "dimensions": dimensions,
            "explained_variance_ratio": float(
                eigenvalues[:dimensions].sum() / eigenvalues.sum()
            ),
            "fit_classes": list(KNOWN_TYPES),
        }

    all_scores: dict[str, np.ndarray] = {}
    all_thresholds: dict[str, float] = {}
    detector_audit: dict[str, Any] = {}
    all_pvalues: dict[str, np.ndarray] = {}
    m, q = 20, 50
    for representation, (pca_mean, components) in pca_models.items():
        print(f"fitting embedding detectors: {representation}", flush=True)
        fit_rep = transform_embeddings(fit_embeddings, pca_mean, components)
        calibration_rep = transform_embeddings(calibration_embeddings, pca_mean, components)
        test_rep = transform_embeddings(test_embeddings, pca_mean, components)
        read_fit = fit_rep.reshape(-1, fit_rep.shape[-1])
        read_fit_categories = np.repeat(fit_categories, m * q)
        read_model = KnownFeatureModel.fit(read_fit, read_fit_categories)
        read_cal_diag = aggregate_read_distances(
            read_model.diagonal_distances(calibration_rep.reshape(-1, calibration_rep.shape[-1])),
            (calibration_rep.shape[0], m, q),
        )
        read_test_diag = aggregate_read_distances(
            read_model.diagonal_distances(test_rep.reshape(-1, test_rep.shape[-1])),
            (test_rep.shape[0], m, q),
        )
        read_cal_shrink = aggregate_read_distances(
            read_model.shrinkage_distances(calibration_rep.reshape(-1, calibration_rep.shape[-1])),
            (calibration_rep.shape[0], m, q),
        )
        read_test_shrink = aggregate_read_distances(
            read_model.shrinkage_distances(test_rep.reshape(-1, test_rep.shape[-1])),
            (test_rep.shape[0], m, q),
        )
        read_prefix = f"{representation}_read_mean"
        read_calibration_by_class = [
            read_cal_shrink[calibration_categories == category, index]
            for index, category in enumerate(KNOWN_TYPES)
        ]
        read_pvalues = conformal_p_values(read_test_shrink, read_calibration_by_class)
        read_cal_scores = {
            f"{read_prefix}_diagonal_predicted": select_class_score(read_cal_diag, calibration_closed, "predicted"),
            f"{read_prefix}_diagonal_minimum": select_class_score(read_cal_diag, calibration_closed, "minimum"),
            f"{read_prefix}_shrinkage_predicted": select_class_score(read_cal_shrink, calibration_closed, "predicted"),
            f"{read_prefix}_shrinkage_minimum": select_class_score(read_cal_shrink, calibration_closed, "minimum"),
            f"{read_prefix}_conformal_predicted": leave_one_out_conformal_scores(read_cal_shrink, calibration_categories, calibration_closed, "predicted"),
            f"{read_prefix}_conformal_maximum_pvalue": leave_one_out_conformal_scores(read_cal_shrink, calibration_categories, calibration_closed, "maximum"),
        }
        read_scores = {
            f"{read_prefix}_diagonal_predicted": select_class_score(read_test_diag, test_closed, "predicted"),
            f"{read_prefix}_diagonal_minimum": select_class_score(read_test_diag, test_closed, "minimum"),
            f"{read_prefix}_shrinkage_predicted": select_class_score(read_test_shrink, test_closed, "predicted"),
            f"{read_prefix}_shrinkage_minimum": select_class_score(read_test_shrink, test_closed, "minimum"),
            f"{read_prefix}_conformal_predicted": -read_pvalues[np.arange(read_pvalues.shape[0]), test_closed],
            f"{read_prefix}_conformal_maximum_pvalue": -read_pvalues.max(axis=1),
        }
        read_thresholds = {
            method: acceptance_threshold(values, known_acceptance)
            for method, values in read_cal_scores.items()
        }
        all_scores.update(read_scores)
        all_thresholds.update(read_thresholds)
        all_pvalues.update({
            f"{read_prefix}_p_{category}": read_pvalues[:, index]
            for index, category in enumerate(KNOWN_TYPES)
        })
        detector_audit[read_prefix] = {
            "aggregation": "distance per read, mean_q, then mean_M",
            "feature_dimension": int(read_fit.shape[1]),
            "fit_read_count": int(read_fit.shape[0]),
            "covariance": {
                category: {
                    "oas_shrinkage": float(read_model.shrinkage[index]),
                    "condition_number": float(read_model.condition_numbers[index]),
                }
                for index, category in enumerate(KNOWN_TYPES)
            },
            "calibration_actual_acceptance": {
                method: float(np.mean(values <= read_thresholds[method]))
                for method, values in read_cal_scores.items()
            },
        }

        archive_fit = archive_mean_std(fit_rep)
        archive_calibration = archive_mean_std(calibration_rep)
        archive_test = archive_mean_std(test_rep)
        archive_scores, archive_thresholds, archive_audit, archive_pvalues = _calibrate_embedding_family(
            f"{representation}_archive_mean_std",
            archive_fit,
            archive_calibration,
            archive_test,
            fit_categories,
            calibration_categories,
            calibration_closed,
            test_closed,
            known_acceptance,
        )
        archive_audit["aggregation"] = "archive embedding mean concatenated with archive embedding std"
        all_scores.update(archive_scores)
        all_thresholds.update(archive_thresholds)
        all_pvalues.update(archive_pvalues)
        detector_audit[f"{representation}_archive_mean_std"] = archive_audit
        del fit_rep, calibration_rep, test_rep

    metrics: dict[str, Any] = {}
    outputs: dict[str, np.ndarray] = {}
    for method, scores in all_scores.items():
        metrics[method], outputs[method] = evaluate_method(
            test_categories,
            ecc_scores,
            test_closed,
            scores,
            all_thresholds[method],
            thresholds_stage1.ecc_presence,
            closed_macro_f1,
        )
    metrics_report = {
        "protocol": "frozen_author_transformer_128d_external_embedding_open_set_validation",
        "closed_set_known_type_macro_f1": closed_macro_f1,
        "methods": metrics,
        "any_method_meets_all_success_criteria": any(value["success"] for value in metrics.values()),
        "success_criteria": {
            "known_acceptance_minimum": 0.93,
            "combined_unknown_recall_minimum": 0.70,
            "unknown_output_as_known_maximum": 0.30,
            "known_macro_f1_drop_maximum": 0.02,
        },
    }
    (output / "embedding_metrics.json").write_text(
        json.dumps(metrics_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    detector_config = {
        "seed": seed,
        "known_acceptance_target": known_acceptance,
        "representations": ["raw128", "pca16", "pca32"],
        "aggregations": ["read_distance_mean_q_then_mean_M", "archive_embedding_mean_std"],
        "detectors": ["standardized_diagonal_mahalanobis", "OAS_class_conditional_mahalanobis", "class_conditional_conformal"],
        "PCA": pca_audit,
        "detector_details": detector_audit,
        "fit_classes": list(KNOWN_TYPES),
        "calibration_classes": list(KNOWN_TYPES),
        "unknown_used_for_fit_or_calibration": False,
    }
    (output / "embedding_detector_config.json").write_text(
        json.dumps(detector_config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    embedding_calibration = {
        "thresholds": all_thresholds,
        "known_acceptance_target": known_acceptance,
        "calibration_archive_count_per_class": calibration_archives_per_category,
        "classes": list(KNOWN_TYPES),
        "HEDGES_or_DNA_Aeon_used": False,
    }
    (output / "embedding_calibration.json").write_text(
        json.dumps(embedding_calibration, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    with (output / "embedding_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("archive_id", "category", "method", "closed_set_output", "ecc_score", "rejection_score", "threshold", "cascade_output", "code_rate", "code_length")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for method, scores in all_scores.items():
            for index, archive_id in enumerate(test_ids):
                writer.writerow({
                    "archive_id": archive_id, "category": test_categories[index],
                    "method": method, "closed_set_output": closed_labels[index],
                    "ecc_score": ecc_scores[index], "rejection_score": scores[index],
                    "threshold": all_thresholds[method], "cascade_output": outputs[method][index],
                    "code_rate": "null", "code_length": "null",
                })
    _write_score_svg(
        output / "embedding_score_distributions.svg", all_scores, all_thresholds, test_categories
    )

    stage1_metrics = json.loads((stage1_output / "stage2_metrics.json").read_text(encoding="utf-8"))
    stage1_methods = stage1_metrics["methods"]
    stage1_best_name = max(
        stage1_methods,
        key=lambda name: (
            stage1_methods[name]["combined_unknown_recall"],
            stage1_methods[name]["known_acceptance_rate"],
        ),
    )
    embedding_best_name = max(
        metrics,
        key=lambda name: (
            metrics[name]["success"],
            metrics[name]["combined_unknown_recall"],
            metrics[name]["known_acceptance_rate"],
        ),
    )
    comparison = {
        "selection_note": "Best-observed labels are descriptive only; no threshold was retuned on unknown tests.",
        "original_energy": stage1_methods["global_energy_original"],
        "stage1_best_observed": {"method": stage1_best_name, **stage1_methods[stage1_best_name]},
        "embedding_best_observed": {"method": embedding_best_name, **metrics[embedding_best_name]},
    }
    (output / "method_comparison.json").write_text(
        json.dumps(comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    models_hash_after = _sha256(models_path)
    weight_hash_after = _sha256(checkpoint_path)
    extraction_audit = {
        "author_models_path": str(models_path),
        "author_models_sha256_before": models_hash_before,
        "author_models_sha256_after": models_hash_after,
        "author_models_unchanged": models_hash_before == models_hash_after,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256_before": weight_hash_before,
        "checkpoint_sha256_after": weight_hash_after,
        "checkpoint_expected_sha256": EXPECTED_SHA256["type/transformer_model_f10.6033.pt"],
        "checkpoint_unchanged": weight_hash_before == weight_hash_after,
        "all_transformer_parameters_frozen": all(not parameter.requires_grad for parameter in task_model.model.parameters()),
        "hook_layer": "fc forward_pre_hook",
        "hook_embedding_dimension": 128,
        "hooked_logits_elementwise_equal_to_unhooked": True,
        "formal_source_logits_allclose": source_logits_allclose,
        "formal_source_logits_max_abs_difference": source_max_abs_difference,
        "fit_calibration_molecular_overlap": len(fingerprints_by_split["stage2-fit"] & fingerprints_by_split["stage2-calibration"]),
        "fit_test_molecular_overlap": len(fingerprints_by_split["stage2-fit"] & fingerprints_by_split["final-test"]),
        "calibration_test_molecular_overlap": len(fingerprints_by_split["stage2-calibration"] & fingerprints_by_split["final-test"]),
        "reference_audits": reference_audits,
        "inner_code_validation": inner_validations,
        "fit_categories": sorted(set(fit_categories)),
        "calibration_categories": sorted(set(calibration_categories)),
        "test_categories": sorted(set(test_categories)),
        "unknown_used_for_fit_or_calibration": False,
        "no_ecc_used_for_fit_or_calibration": False,
        "split_namespaces": {
            "fit": f"embedding-fit-seed-{seed}",
            "calibration": f"embedding-calibration-seed-{seed}",
            "test": f"test-seed-{seed}",
        },
        "stable_seed_only": True,
        "python_hash_used": False,
        "data_contract": {"one_hot": "[M,q,4,400]", "mask": "[M,q,400]"},
        "code_rate": None,
        "code_length": None,
        "command": list(command or []),
        "environment": {
            "python": sys.version, "platform": platform.platform(),
            "numpy": np.__version__, "torch": torch.__version__, "device": str(device_value),
            "cuda_device": torch.cuda.get_device_name(0) if device_value.type == "cuda" else None,
        },
    }
    (output / "embedding_extraction_audit.json").write_text(
        json.dumps(extraction_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return metrics_report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Frozen 128D author embedding rejection")
    parser.add_argument("--source", default="outputs/inner_codes_formal_seed42")
    parser.add_argument("--stage1-output", default="outputs/stage2_feature_rejection_seed42")
    parser.add_argument("--output", default="outputs/stage2_feature_rejection_seed42/embedding_detector")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fit-archives-per-category", type=int, default=20)
    parser.add_argument("--calibration-archives-per-category", type=int, default=20)
    parser.add_argument("--known-acceptance", type=float, default=0.95)
    parser.add_argument("--weights-root", default=str(DEFAULT_WEIGHT_ROOT))
    parser.add_argument("--device", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    command = [sys.executable, "-m", "incremental_validation.embedding_rejection", *(argv or sys.argv[1:])]
    report = run_embedding_stage(
        source=args.source,
        stage1_output=args.stage1_output,
        output=args.output,
        seed=args.seed,
        fit_archives_per_category=args.fit_archives_per_category,
        calibration_archives_per_category=args.calibration_archives_per_category,
        known_acceptance=args.known_acceptance,
        weight_root=args.weights_root,
        device=args.device,
        command=command,
    )
    print(json.dumps(report, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
