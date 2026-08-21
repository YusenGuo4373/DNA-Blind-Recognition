from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence
import argparse
import csv
import json
import platform
import sys

import numpy as np
import torch

from author_baseline.weights import DEFAULT_WEIGHT_ROOT, EXPECTED_SHA256, build_primary_type_recognizer
from hierarchical_ecc.coding import stable_seed
from hierarchical_ecc.config import ExperimentConfig, KNOWN_CODE_TYPES, NO_ECC_TYPES
from hierarchical_ecc.data import ReferenceFactory
from incremental_validation.collector import TorchPresenceDetector
from incremental_validation.comparison import IncrementalThresholds, KNOWN_TYPES
from incremental_validation.inner_codes import archives_from_references, generate_inner_code_references
from incremental_validation.simulation import ExternalPresenceCNN, audit_molecular_references
from incremental_validation.stage2_feature_rejection import _macro_f1, _sha256, acceptance_threshold


SEEDS = (43, 44, 45)
ERROR_RATES = (0.0, 0.01, 0.05, 0.10, 0.15, 0.20)
Q_VALUES = (1, 5, 10, 20, 50)
M_VALUES = (1, 5, 10, 20, 50)
TEST_CATEGORIES = (*KNOWN_CODE_TYPES, *NO_ECC_TYPES, "HEDGES", "DNA-Aeon")
# Keep the Stage-3 seven-output schema.  Conservative mode never emits
# uncertain_ecc, so its row/column is expected to remain zero.
SEVEN_LABELS = ("no_ecc", "uncertain_ecc", "unknown_ecc", *KNOWN_TYPES)
NULL_PARAMETER_OUTPUT = {"code_rate": None, "code_length": None}


def freeze_module(module: torch.nn.Module) -> torch.nn.Module:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)
    return module


def noise_seed(
    seed: int,
    error_rate: float,
    category: str,
    archive_id: int,
    molecule_id: int,
    read_id: int,
) -> int:
    return stable_seed(
        "stage4-noise", seed, error_rate, category, archive_id, molecule_id, read_id
    )


def prefix_soft_vote(logits: np.ndarray, molecules: int, reads: int) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(logits, dtype=np.float64)
    if values.ndim != 4 or values.shape[-1] != 4:
        raise ValueError("logits must have shape [N,M,q,4]")
    if not 0 < molecules <= values.shape[1] or not 0 < reads <= values.shape[2]:
        raise ValueError("prefix M/q exceeds available logits")
    prefix = values[:, :molecules, :reads]
    maximum = prefix.max(axis=-1, keepdims=True)
    probabilities = np.exp(prefix - maximum)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    archive_probabilities = probabilities.mean(axis=2).mean(axis=1)
    energy_reads = -(maximum[..., 0] + np.log(np.exp(prefix - maximum).sum(axis=-1)))
    energy = energy_reads.mean(axis=2).mean(axis=1)
    return archive_probabilities, energy


def prefix_presence(probabilities: np.ndarray, molecules: int, reads: int) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 3:
        raise ValueError("presence probabilities must have shape [N,M,q]")
    return values[:, :molecules, :reads].mean(axis=2).mean(axis=1)


def conservative_outputs(
    categories: np.ndarray,
    presence: np.ndarray,
    logits: np.ndarray,
    tau1: float,
    tau2: float,
    molecules: int,
    reads: int,
) -> dict[str, np.ndarray]:
    categories = np.asarray(categories, dtype=str)
    archive_probabilities, energy = prefix_soft_vote(logits, molecules, reads)
    ecc = prefix_presence(presence, molecules, reads)
    closed_index = archive_probabilities.argmax(axis=1)
    closed = np.asarray(KNOWN_TYPES, dtype=object)[closed_index]
    outputs = closed.copy()
    outputs[energy > tau2] = "unknown_ecc"
    outputs[ecc < tau1] = "no_ecc"
    return {
        "ecc_score": ecc,
        "energy": energy,
        "closed_index": closed_index,
        "closed_output": closed.astype(str),
        "output": outputs.astype(str),
        "type_probabilities": archive_probabilities,
    }


def seven_class_confusion(categories: Sequence[str], outputs: Sequence[str]) -> list[list[int]]:
    categories = np.asarray(categories, dtype=str)
    outputs = np.asarray(outputs, dtype=str)
    truth = np.asarray([
        category if category in KNOWN_TYPES else (
            "no_ecc" if category in NO_ECC_TYPES else "unknown_ecc"
        )
        for category in categories
    ])
    index = {label: position for position, label in enumerate(SEVEN_LABELS)}
    matrix = np.zeros((len(SEVEN_LABELS), len(SEVEN_LABELS)), dtype=np.int64)
    for expected, observed in zip(truth, outputs):
        matrix[index[str(expected)], index[str(observed)]] += 1
    return matrix.tolist()


def condition_metrics(categories: np.ndarray, result: dict[str, np.ndarray]) -> dict[str, Any]:
    categories = np.asarray(categories, dtype=str)
    outputs = result["output"]
    known = np.isin(categories, KNOWN_TYPES)
    no_ecc = np.isin(categories, NO_ECC_TYPES)
    unknown = np.isin(categories, ("HEDGES", "DNA-Aeon"))
    closed_macro_f1 = _macro_f1(categories[known], result["closed_output"][known])
    cascade_macro_f1 = _macro_f1(categories[known], outputs[known])
    metrics = {
        "known_acceptance_rate": float(np.mean(np.isin(outputs[known], KNOWN_TYPES))),
        "known_type_acceptance": {
            category: float(np.mean(np.isin(outputs[categories == category], KNOWN_TYPES)))
            for category in KNOWN_TYPES
        },
        "closed_set_known_type_macro_f1": closed_macro_f1,
        "known_type_macro_f1": cascade_macro_f1,
        "known_type_macro_f1_change_from_closed": cascade_macro_f1 - closed_macro_f1,
        "no_ecc_specificity": float(np.mean(outputs[no_ecc] == "no_ecc")),
        "HEDGES_unknown_recall": float(np.mean(outputs[categories == "HEDGES"] == "unknown_ecc")),
        "DNA_Aeon_unknown_recall": float(np.mean(outputs[categories == "DNA-Aeon"] == "unknown_ecc")),
        "combined_unknown_recall": float(np.mean(outputs[unknown] == "unknown_ecc")),
        "unknown_misclassified_as_BCH_rate": float(np.mean(outputs[unknown] == "BCH")),
        "sample_count_per_category": {
            category: int(np.sum(categories == category)) for category in TEST_CATEGORIES
        },
        "labels": list(SEVEN_LABELS),
        "seven_class_confusion_matrix": seven_class_confusion(categories, outputs),
    }
    return metrics


def bootstrap_archive_metrics(
    categories: np.ndarray,
    result: dict[str, np.ndarray],
    seed: int,
    repetitions: int = 2000,
) -> dict[str, Any]:
    """Bootstrap complete archives, never individual reads."""

    categories = np.asarray(categories, dtype=str)
    keys = (
        "known_acceptance_rate", "known_type_macro_f1",
        "known_type_macro_f1_change_from_closed", "no_ecc_specificity",
        "HEDGES_unknown_recall", "DNA_Aeon_unknown_recall",
        "combined_unknown_recall", "unknown_misclassified_as_BCH_rate",
    )
    samples = {key: np.empty(repetitions, dtype=np.float64) for key in keys}
    category_indices = {
        category: np.flatnonzero(categories == category) for category in TEST_CATEGORIES
    }
    for repetition in range(repetitions):
        rng = np.random.default_rng(stable_seed("stage4-bootstrap", seed, repetition))
        selected = np.concatenate([
            rng.choice(indices, size=indices.size, replace=True)
            for indices in category_indices.values()
        ])
        subset_result = {
            key: value[selected] if isinstance(value, np.ndarray) and value.shape[0] == categories.size else value
            for key, value in result.items()
        }
        values = condition_metrics(categories[selected], subset_result)
        for key in keys:
            samples[key][repetition] = values[key]
    return {
        "unit": "archive",
        "repetitions": repetitions,
        "metrics": {
            key: {
                "mean": float(values.mean()),
                "lower_95": float(np.quantile(values, 0.025)),
                "upper_95": float(np.quantile(values, 0.975)),
            }
            for key, values in samples.items()
        },
    }


def _reference_fingerprints(reference_sets: dict[str, np.ndarray]) -> set[bytes]:
    result: set[bytes] = set()
    for references in reference_sets.values():
        for reference in np.asarray(references, dtype=np.uint8):
            encoded = reference.tobytes()
            if encoded in result:
                raise RuntimeError("duplicate reference within split")
            result.add(encoded)
    return result


def generate_test_reference_pool(
    experiment: ExperimentConfig,
    seed: int,
    output: Path,
    archives: int = 50,
    molecules: int = 50,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    factory = ReferenceFactory(experiment)
    namespace = f"stage4-test-seed-{seed}"
    reference_sets = {
        category: np.stack([
            factory.make_reference(category, namespace, archive_id, molecule_id)
            for archive_id in range(archives)
            for molecule_id in range(molecules)
        ])
        for category in (*KNOWN_CODE_TYPES, *NO_ECC_TYPES)
    }
    validations: list[dict[str, Any]] = []
    for category in ("HEDGES", "DNA-Aeon"):
        references, validation = generate_inner_code_references(
            category, archives * molecules, seed,
            output / f"{category.lower().replace('-', '_')}.fasta",
            namespace="stage4-robustness-test",
        )
        reference_sets[category] = references
        validations.append(validation.__dict__)
    return reference_sets, validations


def generate_calibration_reference_pool(
    experiment: ExperimentConfig,
    seed: int,
    archives: int = 20,
    molecules: int = 20,
) -> dict[str, np.ndarray]:
    factory = ReferenceFactory(experiment)
    namespace = f"stage4-calibration-seed-{seed}"
    return {
        category: np.stack([
            factory.make_reference(category, namespace, archive_id, molecule_id)
            for archive_id in range(archives)
            for molecule_id in range(molecules)
        ])
        for category in KNOWN_CODE_TYPES
    }


def score_condition(
    experiment: ExperimentConfig,
    reference_sets: dict[str, np.ndarray],
    seed: int,
    error_rate: float,
    molecules: int,
    reads: int,
    author_task: Any,
    presence_detector: TorchPresenceDetector,
) -> dict[str, np.ndarray]:
    categories: list[str] = []
    archive_ids: list[str] = []
    presence_values: list[np.ndarray] = []
    logits_values: list[np.ndarray] = []
    for category in TEST_CATEGORIES:
        pool = reference_sets[category].reshape(50, 50, 384)[:, :molecules].reshape(-1, 384)
        archives = archives_from_references(
            experiment, category, f"stage4-test-seed-{seed}", pool,
            50, molecules, reads, error_rate,
        )
        for archive_index, archive in enumerate(archives):
            presence_values.append(presence_detector.predict_probabilities(archive))
            logits_values.append(author_task.read_logits(archive).numpy())
            categories.append(category)
            archive_ids.append(f"seed{seed}:error{error_rate}:{category}:{archive_index}")
        print(
            f"stage4 seed={seed} error={error_rate} category={category} archives=50 M={molecules} q={reads}",
            flush=True,
        )
    return {
        "categories": np.asarray(categories, dtype=str),
        "archive_ids": np.asarray(archive_ids, dtype=str),
        "presence": np.stack(presence_values).astype(np.float32),
        "logits": np.stack(logits_values).astype(np.float32),
    }


def recalibrate_energy_threshold(
    experiment: ExperimentConfig,
    references: dict[str, np.ndarray],
    seed: int,
    author_task: Any,
    target: float = 0.98,
) -> tuple[float, dict[str, Any]]:
    categories: list[str] = []
    logits: list[np.ndarray] = []
    forbidden = {"HEDGES", "DNA-Aeon", *NO_ECC_TYPES}
    if set(references) & forbidden:
        raise ValueError("recalibration must contain only four known ECC classes")
    for category in KNOWN_CODE_TYPES:
        archives = archives_from_references(
            experiment, category, f"stage4-calibration-seed-{seed}", references[category],
            20, 20, 50, 0.05,
        )
        for archive in archives:
            logits.append(author_task.read_logits(archive).numpy())
            categories.append(category)
    bundle = np.stack(logits)
    _, energy = prefix_soft_vote(bundle, 20, 50)
    threshold = acceptance_threshold(energy, target)
    return threshold, {
        "target_known_acceptance": target,
        "actual_calibration_acceptance": float(np.mean(energy <= threshold)),
        "archive_count": len(categories),
        "categories": sorted(set(categories)),
        "forbidden_categories_used": False,
    }


def _write_curves(path: Path, title: str, x_label: str, series: dict[str, tuple[np.ndarray, np.ndarray]]) -> None:
    width, height = 900, 520
    left, top, plot_width, plot_height = 70, 45, 790, 390
    colors = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e")
    all_x = np.concatenate([value[0] for value in series.values()]).astype(float)
    x_min, x_max = float(all_x.min()), float(all_x.max())
    if x_min == x_max:
        x_max += 1.0
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="450" y="24" text-anchor="middle" font-family="sans-serif" font-size="17">{title}</text>']
    svg.append(f'<line x1="{left}" y1="{top+plot_height}" x2="{left+plot_width}" y2="{top+plot_height}" stroke="black"/>')
    svg.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_height}" stroke="black"/>')
    for index, (name, (x_values, y_values)) in enumerate(series.items()):
        points = []
        for x_value, y_value in zip(x_values, y_values):
            x = left + (float(x_value) - x_min) / (x_max - x_min) * plot_width
            y = top + (1.0 - float(y_value)) * plot_height
            points.append(f"{x:.1f},{y:.1f}")
            svg.append(f'<circle cx="{x:.1f}" cy="{y:.1f}" r="3" fill="{colors[index % len(colors)]}"/>')
        svg.append(f'<polyline points="{" ".join(points)}" fill="none" stroke="{colors[index % len(colors)]}" stroke-width="2"/>')
        svg.append(f'<text x="{left+20+index*150}" y="{height-35}" fill="{colors[index % len(colors)]}" font-family="sans-serif" font-size="11">{name}</text>')
    svg.append(f'<text x="450" y="{height-8}" text-anchor="middle" font-family="sans-serif" font-size="12">{x_label}</text>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def run_stage4(
    source: str | Path,
    stage3: str | Path,
    output: str | Path,
    device: str | None = None,
    resume: bool = False,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    source = Path(source).resolve()
    stage3 = Path(stage3).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    score_root = output / "condition_scores"
    score_root.mkdir(exist_ok=True)
    experiment = ExperimentConfig()
    device_value = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    fixed_tau1 = IncrementalThresholds.load(source / "thresholds.json").ecc_presence
    calibration = json.loads((stage3 / "multidetector_calibration.json").read_text(encoding="utf-8"))
    fixed_tau2 = float(calibration["thresholds"]["0.98"]["global_energy"])
    models_path = Path("vendor/zhouph0313_DNA/models.py").resolve()
    weight_path = Path(DEFAULT_WEIGHT_ROOT).resolve() / "type" / "transformer_model_f10.6033.pt"
    models_hash_before, weight_hash_before = _sha256(models_path), _sha256(weight_path)
    author = build_primary_type_recognizer(device=device_value, batch_size=64)
    freeze_module(author.code_type.model)
    presence_checkpoint = torch.load(
        source / "models" / "external_presence_cnn.pt", map_location="cpu", weights_only=True
    )
    presence_model = ExternalPresenceCNN()
    presence_model.load_state_dict(presence_checkpoint["state_dict"], strict=True)
    presence = TorchPresenceDetector(presence_model, device=device_value, batch_size=64)

    protocol = {
        "experiment_positioning": "冻结作者盲识别核心条件下，保守开放集工作点的跨随机种子、信道错误率及软投票规模稳健性验证。",
        "seeds": list(SEEDS), "error_rates": list(ERROR_RATES),
        "q_values": list(Q_VALUES), "M_values": list(M_VALUES),
        "fixed_tau1": fixed_tau1, "fixed_tau2": fixed_tau2,
        "main_rule": "t98_energy_only", "thresholds_never_selected_on_test": True,
        "HEDGES_DNA_Aeon_descriptive_only": True,
    }
    (output / "conservative_protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    all_test_fingerprints: dict[int, set[bytes]] = {}
    reference_audits: dict[str, Any] = {}
    seed_split_audit: dict[str, Any] = {}
    recalibrated_thresholds: dict[int, float] = {}
    recalibration_audits: dict[str, Any] = {}
    for seed in SEEDS:
        reference_sets, inner_validation = generate_test_reference_pool(
            experiment, seed, output / "reference_pools" / f"seed{seed}" / "test"
        )
        calibration_references = generate_calibration_reference_pool(experiment, seed)
        test_fingerprints = _reference_fingerprints(reference_sets)
        calibration_fingerprints = _reference_fingerprints(calibration_references)
        overlap = len(test_fingerprints & calibration_fingerprints)
        if overlap:
            raise RuntimeError(f"seed {seed} calibration/test molecule overlap")
        all_test_fingerprints[seed] = test_fingerprints
        reference_audits[f"seed{seed}_test"] = audit_molecular_references(reference_sets)
        reference_audits[f"seed{seed}_calibration"] = audit_molecular_references(calibration_references)
        seed_split_audit[str(seed)] = {
            "test_namespace": f"stage4-test-seed-{seed}",
            "calibration_namespace": f"stage4-calibration-seed-{seed}",
            "calibration_test_overlap": overlap,
            "inner_code_validation": inner_validation,
        }
        tau2, recalibration_audit = recalibrate_energy_threshold(
            experiment, calibration_references, seed, author.code_type, 0.98
        )
        recalibrated_thresholds[seed] = tau2
        recalibration_audits[str(seed)] = recalibration_audit
        np.savez_compressed(
            output / "reference_pools" / f"seed{seed}" / "test_references.npz",
            **{category: references for category, references in reference_sets.items()},
        )
        for error_rate in ERROR_RATES:
            condition_path = score_root / f"seed{seed}_error{error_rate:.2f}.npz"
            if condition_path.is_file() and resume:
                continue
            molecules = 50 if np.isclose(error_rate, 0.05) else 20
            scored = score_condition(
                experiment, reference_sets, seed, error_rate, molecules, 50,
                author.code_type, presence,
            )
            np.savez_compressed(condition_path, **scored)
    for left_index, left_seed in enumerate(SEEDS):
        for right_seed in SEEDS[left_index + 1:]:
            overlap = len(all_test_fingerprints[left_seed] & all_test_fingerprints[right_seed])
            seed_split_audit[f"seed{left_seed}|seed{right_seed}"] = {"test_reference_overlap": overlap}
            if overlap:
                raise RuntimeError("reference overlap across experiment seeds")
    (output / "seed_split_audit.json").write_text(
        json.dumps(seed_split_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "reference_molecule_audit.json").write_text(
        json.dumps(reference_audits, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    fixed_prediction_rows: list[dict[str, Any]] = []
    recal_prediction_rows: list[dict[str, Any]] = []
    error_rows: list[dict[str, Any]] = []
    q_rows: list[dict[str, Any]] = []
    m_rows: list[dict[str, Any]] = []
    per_code_rows: list[dict[str, Any]] = []
    per_seed_metrics: dict[str, Any] = {}
    confusion: dict[str, Any] = {}
    bootstrap: dict[str, Any] = {}
    main_seed_values: dict[str, list[float]] = {}
    pooled_categories: list[np.ndarray] = []
    pooled_results: list[dict[str, np.ndarray]] = []
    for seed in SEEDS:
        per_seed_metrics[str(seed)] = {"fixed": {}, "recalibrated": {}}
        for error_rate in ERROR_RATES:
            payload = np.load(score_root / f"seed{seed}_error{error_rate:.2f}.npz", allow_pickle=False)
            categories = payload["categories"].astype(str)
            max_m = int(payload["presence"].shape[1])
            fixed = conservative_outputs(
                categories, payload["presence"], payload["logits"],
                fixed_tau1, fixed_tau2, min(20, max_m), 50,
            )
            recal = conservative_outputs(
                categories, payload["presence"], payload["logits"],
                fixed_tau1, recalibrated_thresholds[seed], min(20, max_m), 50,
            )
            fixed_metrics = condition_metrics(categories, fixed)
            recal_metrics = condition_metrics(categories, recal)
            per_seed_metrics[str(seed)]["fixed"][str(error_rate)] = fixed_metrics
            per_seed_metrics[str(seed)]["recalibrated"][str(error_rate)] = recal_metrics
            for mode, metrics, result, rows in (
                ("fixed", fixed_metrics, fixed, fixed_prediction_rows),
                ("recalibrated", recal_metrics, recal, recal_prediction_rows),
            ):
                error_rows.append({
                    "seed": seed, "error_rate": error_rate, "threshold_mode": mode,
                    "tau2": fixed_tau2 if mode == "fixed" else recalibrated_thresholds[seed],
                    **{key: metrics[key] for key in (
                        "known_acceptance_rate", "closed_set_known_type_macro_f1",
                        "known_type_macro_f1", "known_type_macro_f1_change_from_closed",
                        "no_ecc_specificity", "HEDGES_unknown_recall",
                        "DNA_Aeon_unknown_recall", "combined_unknown_recall",
                        "unknown_misclassified_as_BCH_rate",
                    )},
                })
                for index, archive_id in enumerate(payload["archive_ids"].astype(str)):
                    rows.append({
                        "archive_id": archive_id, "seed": seed, "error_rate": error_rate,
                        "category": categories[index], "threshold_mode": mode,
                        "q": 50, "M": 20, "ecc_score": result["ecc_score"][index],
                        "energy": result["energy"][index], "tau1": fixed_tau1,
                        "tau2": fixed_tau2 if mode == "fixed" else recalibrated_thresholds[seed],
                        "closed_set_output": result["closed_output"][index],
                        "cascade_output": result["output"][index],
                        "code_rate": "null", "code_length": "null",
                    })
                confusion[f"seed{seed}_error{error_rate}_{mode}"] = {
                    "labels": list(SEVEN_LABELS),
                    "matrix": metrics["seven_class_confusion_matrix"],
                }
            for category, value in fixed_metrics["known_type_acceptance"].items():
                per_code_rows.append({
                    "seed": seed, "error_rate": error_rate, "threshold_mode": "fixed",
                    "category": category, "acceptance_rate": value,
                })
            if np.isclose(error_rate, 0.05):
                bootstrap[str(seed)] = bootstrap_archive_metrics(categories, fixed, seed)
                pooled_categories.append(categories)
                pooled_results.append(fixed)
                for key in (
                    "known_acceptance_rate", "known_type_macro_f1_change_from_closed",
                    "no_ecc_specificity", "combined_unknown_recall",
                ):
                    main_seed_values.setdefault(key, []).append(float(fixed_metrics[key]))
                for q_value in Q_VALUES:
                    result = conservative_outputs(
                        categories, payload["presence"], payload["logits"],
                        fixed_tau1, fixed_tau2, 20, q_value,
                    )
                    metrics = condition_metrics(categories, result)
                    q_rows.append({
                        "seed": seed, "q": q_value, "M": 20,
                        **{key: metrics[key] for key in (
                            "known_acceptance_rate", "known_type_macro_f1",
                            "known_type_macro_f1_change_from_closed", "no_ecc_specificity",
                            "HEDGES_unknown_recall", "DNA_Aeon_unknown_recall",
                            "unknown_misclassified_as_BCH_rate",
                        )},
                    })
                for m_value in M_VALUES:
                    result = conservative_outputs(
                        categories, payload["presence"], payload["logits"],
                        fixed_tau1, fixed_tau2, m_value, 50,
                    )
                    metrics = condition_metrics(categories, result)
                    m_rows.append({
                        "seed": seed, "q": 50, "M": m_value,
                        **{key: metrics[key] for key in (
                            "known_acceptance_rate", "known_type_macro_f1",
                            "known_type_macro_f1_change_from_closed", "no_ecc_specificity",
                            "HEDGES_unknown_recall", "DNA_Aeon_unknown_recall",
                            "unknown_misclassified_as_BCH_rate",
                        )},
                    })
                repeated = conservative_outputs(
                    categories, payload["presence"], payload["logits"],
                    fixed_tau1, fixed_tau2, 20, 50,
                )
                if not np.array_equal(fixed["output"], repeated["output"]):
                    raise RuntimeError("fixed condition is not deterministic")

    combined_categories = np.concatenate(pooled_categories)
    combined_result = {
        key: np.concatenate([result[key] for result in pooled_results], axis=0)
        for key in pooled_results[0]
    }
    bootstrap["three_seed_pooled_archive_bootstrap"] = bootstrap_archive_metrics(
        combined_categories, combined_result, stable_seed("stage4", "pooled", 43, 44, 45)
    )
    pooled_intervals = bootstrap["three_seed_pooled_archive_bootstrap"]["metrics"]
    bootstrap["three_seed_summary"] = {
        key: {
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)),
            "minimum": float(np.min(values)),
            "maximum": float(np.max(values)),
            "pooled_archive_bootstrap_lower_95": pooled_intervals[key]["lower_95"],
            "pooled_archive_bootstrap_upper_95": pooled_intervals[key]["upper_95"],
        }
        for key, values in main_seed_values.items()
    }
    (output / "per_seed_metrics.json").write_text(
        json.dumps(per_seed_metrics, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "bootstrap_confidence_intervals.json").write_text(
        json.dumps(bootstrap, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "seven_class_confusion_matrices.json").write_text(
        json.dumps(confusion, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    fields_predictions = (
        "archive_id", "seed", "error_rate", "category", "threshold_mode", "q", "M",
        "ecc_score", "energy", "tau1", "tau2", "closed_set_output", "cascade_output",
        "code_rate", "code_length",
    )
    for filename, rows in (
        ("fixed_threshold_predictions.csv", fixed_prediction_rows),
        ("recalibrated_threshold_predictions.csv", recal_prediction_rows),
    ):
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields_predictions)
            writer.writeheader(); writer.writerows(rows)
    for filename, rows in (
        ("error_rate_metrics.csv", error_rows),
        ("q_sensitivity_metrics.csv", q_rows),
        ("M_sensitivity_metrics.csv", m_rows),
        ("per_code_type_acceptance.csv", per_code_rows),
    ):
        with (output / filename).open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)
    transfer_rows = [{
        "seed": seed, "error_rate": error_rate,
        "fixed_tau2": fixed_tau2, "recalibrated_tau2": recalibrated_thresholds[seed],
        "tau2_difference": recalibrated_thresholds[seed] - fixed_tau2,
        "fixed_known_acceptance": per_seed_metrics[str(seed)]["fixed"][str(error_rate)]["known_acceptance_rate"],
        "recalibrated_known_acceptance": per_seed_metrics[str(seed)]["recalibrated"][str(error_rate)]["known_acceptance_rate"],
        "acceptance_difference": per_seed_metrics[str(seed)]["recalibrated"][str(error_rate)]["known_acceptance_rate"] - per_seed_metrics[str(seed)]["fixed"][str(error_rate)]["known_acceptance_rate"],
    } for seed in SEEDS for error_rate in ERROR_RATES]
    with (output / "threshold_transfer_analysis.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(transfer_rows[0])); writer.writeheader(); writer.writerows(transfer_rows)

    fixed_error_rows = [row for row in error_rows if row["threshold_mode"] == "fixed"]
    error_x = np.asarray(ERROR_RATES)
    _write_curves(output / "conservative_error_rate_curves.svg", "Conservative fixed-threshold robustness across IDS rates", "IDS error rate", {
        "known acceptance": (error_x, np.asarray([np.mean([row["known_acceptance_rate"] for row in fixed_error_rows if np.isclose(row["error_rate"], rate)]) for rate in ERROR_RATES])),
        "NoECC specificity": (error_x, np.asarray([np.mean([row["no_ecc_specificity"] for row in fixed_error_rows if np.isclose(row["error_rate"], rate)]) for rate in ERROR_RATES])),
        "known macro F1": (error_x, np.asarray([np.mean([row["known_type_macro_f1"] for row in fixed_error_rows if np.isclose(row["error_rate"], rate)]) for rate in ERROR_RATES])),
    })
    _write_curves(output / "conservative_q_M_curves.svg", "Conservative q/M prefix sensitivity", "q or M prefix", {
        "vary q (M=20)": (np.asarray(Q_VALUES), np.asarray([np.mean([row["known_acceptance_rate"] for row in q_rows if row["q"] == value]) for value in Q_VALUES])),
        "vary M (q=50)": (np.asarray(M_VALUES), np.asarray([np.mean([row["known_acceptance_rate"] for row in m_rows if row["M"] == value]) for value in M_VALUES])),
    })

    models_hash_after, weight_hash_after = _sha256(models_path), _sha256(weight_path)
    environment = {
        "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
        "torch": torch.__version__, "device": str(device_value),
        "gpu": torch.cuda.get_device_name(0) if device_value.type == "cuda" else None,
        "models_py_sha256_before": models_hash_before, "models_py_sha256_after": models_hash_after,
        "author_weight_sha256_before": weight_hash_before, "author_weight_sha256_after": weight_hash_after,
        "author_weight_expected_sha256": EXPECTED_SHA256["type/transformer_model_f10.6033.pt"],
        "models_py_unchanged": models_hash_before == models_hash_after,
        "author_weight_unchanged": weight_hash_before == weight_hash_after,
        "author_transformer_frozen": all(not parameter.requires_grad for parameter in author.code_type.model.parameters()),
    }
    (output / "environment_audit.json").write_text(
        json.dumps(environment, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output / "test_commands.json").write_text(
        json.dumps({"experiment": list(command or []), "pytest": [sys.executable, "-m", "pytest", "-q"]}, indent=2), encoding="utf-8"
    )
    input_files = [
        source / "thresholds.json", source / "models" / "external_presence_cnn.pt",
        stage3 / "multidetector_calibration.json", stage3 / "multidetector_metrics.json",
        weight_path,
    ]
    manifest = {
        **protocol,
        "command": list(command or []),
        "recalibrated_tau2": {str(seed): value for seed, value in recalibrated_thresholds.items()},
        "recalibration_audits": recalibration_audits,
        "input_sha256": {str(path): _sha256(path) for path in input_files},
        "namespaces": seed_split_audit,
        "environment": environment,
        "HEDGES_DNA_Aeon_used_for_threshold_selection": False,
        "code_rate": None, "code_length": None,
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return {
        "fixed_tau1": fixed_tau1, "fixed_tau2": fixed_tau2,
        "recalibrated_tau2": recalibrated_thresholds,
        "three_seed_main_summary": bootstrap["three_seed_summary"],
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-4 conservative robustness validation")
    parser.add_argument("--source", default="outputs/inner_codes_formal_seed42")
    parser.add_argument("--stage3", default="outputs/stage3_multidetector_proxy_exposure_seed42")
    parser.add_argument("--output", default="outputs/stage4_conservative_robustness")
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    command = [sys.executable, "-m", "incremental_validation.stage4_conservative_robustness", *(argv or sys.argv[1:])]
    result = run_stage4(args.source, args.stage3, args.output, args.device, args.resume, command)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
