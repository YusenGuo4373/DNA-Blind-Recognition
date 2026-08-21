from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterable
import csv
import json

import numpy as np
import torch

from .coding import stable_seed
from .config import ExperimentConfig, KNOWN_CODE_TYPES, NO_ECC_TYPES, UNKNOWN_CODE_TYPE
from .data import generate_archive_reads
from .inference import ArchiveModelScores, score_archive
from .metrics import (
    accuracy_score,
    binary_aupr,
    binary_auroc,
    confusion_matrix,
    f1_from_confusion,
    fpr_at_tpr,
    macro_f1_score,
)
from .models import DNAReadTransformer
from .reporting import write_curve_svgs
from .voting import (
    Thresholds,
    energy_score,
    hierarchical_decision,
    select_presence_threshold,
    select_unknown_threshold,
    softmax,
    two_level_soft_vote,
)


END_TO_END_LABELS = ("no_ecc", "unknown_ecc") + KNOWN_CODE_TYPES


def _write_rows(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")


def calibrate_thresholds(
    config: ExperimentConfig,
    seed: int,
    presence_model: DNAReadTransformer,
    type_model: DNAReadTransformer,
    output_directory: str | Path,
    device: str | torch.device,
    batch_size: int = 256,
    archives_per_category: int | None = None,
    molecules: int | None = None,
    reads_per_molecule: int | None = None,
) -> Thresholds:
    """Calibrate tau1/tau2 without ever constructing a fountain sample."""

    output_directory = Path(output_directory)
    count = archives_per_category or config.training.calibration_archives_per_category
    molecules = molecules or config.voting.default_molecules
    reads_per_molecule = reads_per_molecule or config.voting.default_reads
    rows: list[dict[str, Any]] = []
    presence_scores: list[float] = []
    presence_labels: list[int] = []
    known_energies: list[float] = []
    rates = config.channel.train_error_rates

    for category in KNOWN_CODE_TYPES + NO_ECC_TYPES:
        for archive_id in range(count):
            error_rate = float(rates[archive_id % len(rates)])
            archive = generate_archive_reads(
                config,
                category=category,
                split=f"calibration-seed-{seed}",
                archive_id=archive_id,
                error_rate=error_rate,
                molecules=molecules,
                reads_per_molecule=reads_per_molecule,
            )
            scored = score_archive(
                presence_model,
                archive,
                device=device,
                batch_size=batch_size,
                type_model=type_model if category in KNOWN_CODE_TYPES else None,
            )
            ecc_score = float(two_level_soft_vote(scored.presence_probabilities))
            presence_scores.append(ecc_score)
            presence_labels.append(int(category in KNOWN_CODE_TYPES))
            archive_energy: float | None = None
            if category in KNOWN_CODE_TYPES:
                if scored.type_logits is None:
                    raise RuntimeError("type logits missing for known calibration sample")
                archive_energy = float(two_level_soft_vote(energy_score(scored.type_logits)))
                known_energies.append(archive_energy)
            rows.append(
                {
                    "seed": seed,
                    "category": category,
                    "archive_id": archive_id,
                    "error_rate": error_rate,
                    "M": molecules,
                    "q": reads_per_molecule,
                    "presence_label": int(category in KNOWN_CODE_TYPES),
                    "ecc_score": ecc_score,
                    "known_energy": archive_energy,
                }
            )
        print(
            f"calibration seed={seed} category={category} archives={count}",
            flush=True,
        )

    tau1, validation_macro_f1 = select_presence_threshold(
        np.asarray(presence_scores), np.asarray(presence_labels)
    )
    tau2 = select_unknown_threshold(
        np.asarray(known_energies), known_acceptance=config.voting.known_acceptance
    )
    thresholds = Thresholds(
        presence=tau1,
        unknown_energy=tau2,
        known_acceptance=config.voting.known_acceptance,
    )
    thresholds.save(output_directory / f"thresholds_seed_{seed}.json")
    _write_rows(output_directory / f"calibration_seed_{seed}.csv", rows)
    _save_json(
        output_directory / f"calibration_seed_{seed}_summary.json",
        {
            "seed": seed,
            "thresholds": asdict(thresholds),
            "presence_validation_macro_f1": validation_macro_f1,
            "known_validation_acceptance": float(np.mean(np.asarray(known_energies) <= tau2)),
            "fountain_samples_used": 0,
            "M": molecules,
            "q": reads_per_molecule,
        },
    )
    return thresholds


def _settings(config: ExperimentConfig) -> list[tuple[str, int, int]]:
    result = [
        (
            "default",
            config.voting.default_molecules,
            config.voting.default_reads,
        )
    ]
    result.extend(
        ("q_sweep", config.voting.default_molecules, q) for q in config.voting.read_sweep
    )
    result.extend(
        ("M_sweep", molecules, config.voting.default_reads)
        for molecules in config.voting.molecule_sweep
    )
    return result


def _true_final_label(category: str) -> str:
    if category in NO_ECC_TYPES:
        return "no_ecc"
    if category == UNKNOWN_CODE_TYPE:
        return "unknown_ecc"
    return category


def _decision_label(status: str, code_type: str | None) -> str:
    return code_type if status == "known_ecc" and code_type is not None else status


def _record_for_prefix(
    seed: int,
    category: str,
    archive_id: int,
    error_rate: float,
    setting: str,
    molecules: int,
    reads: int,
    scores: ArchiveModelScores,
    thresholds: Thresholds,
) -> dict[str, Any]:
    prefix = scores.prefix(molecules, reads)
    if prefix.type_logits is None:
        raise RuntimeError("evaluation requires type logits")
    decision = hierarchical_decision(
        prefix.presence_probabilities,
        prefix.type_logits,
        thresholds,
    )
    stage2_energy = float(two_level_soft_vote(energy_score(prefix.type_logits)))
    type_probabilities = np.asarray(two_level_soft_vote(softmax(prefix.type_logits)))
    mean_logits = np.asarray(two_level_soft_vote(prefix.type_logits))
    standalone_type = KNOWN_CODE_TYPES[int(np.argmax(type_probabilities))]
    row: dict[str, Any] = {
        "seed": seed,
        "setting": setting,
        "category": category,
        "archive_id": archive_id,
        "error_rate": error_rate,
        "M": molecules,
        "q": reads,
        "true_presence": int(category not in NO_ECC_TYPES),
        "ecc_score": decision.ecc_score,
        "predicted_presence": int(decision.ecc_score >= thresholds.presence),
        "stage2_energy": stage2_energy,
        "stage2_predicted_unknown": int(stage2_energy > thresholds.unknown_energy),
        "standalone_type": standalone_type,
        "status": decision.status,
        "code_type": decision.code_type,
        "code_rate": None,
        "code_length": None,
        "unknown_score": decision.unknown_score,
        "max_type_probability": float(np.max(type_probabilities)),
        "max_logit": float(np.max(mean_logits)),
        "true_final_label": _true_final_label(category),
        "predicted_final_label": _decision_label(decision.status, decision.code_type),
        "tau1": thresholds.presence,
        "tau2": thresholds.unknown_energy,
    }
    for index, code_type in enumerate(KNOWN_CODE_TYPES):
        row[f"prob_{code_type}"] = float(type_probabilities[index])
    return row


def evaluate_seed(
    config: ExperimentConfig,
    seed: int,
    presence_model: DNAReadTransformer,
    type_model: DNAReadTransformer,
    thresholds: Thresholds,
    output_directory: str | Path,
    device: str | torch.device,
    batch_size: int = 256,
    archives_per_category: int | None = None,
    test_error_rates: Iterable[float] | None = None,
    max_molecules: int | None = None,
    max_reads: int | None = None,
) -> dict[str, Any]:
    output_directory = Path(output_directory)
    count = archives_per_category or config.training.test_archives_per_category
    errors = tuple(config.channel.test_error_rates if test_error_rates is None else test_error_rates)
    required_molecules = max(molecules for _, molecules, _ in _settings(config))
    required_reads = max(reads for _, _, reads in _settings(config))
    maximum_molecules = max_molecules or required_molecules
    maximum_reads = max_reads or required_reads
    if maximum_molecules < required_molecules or maximum_reads < required_reads:
        raise ValueError("generated archive must cover all configured q/M sweep prefixes")

    rows: list[dict[str, Any]] = []
    for category in KNOWN_CODE_TYPES + NO_ECC_TYPES + (UNKNOWN_CODE_TYPE,):
        for error_rate in errors:
            for archive_id in range(count):
                archive = generate_archive_reads(
                    config,
                    category=category,
                    split=f"test-seed-{seed}",
                    archive_id=archive_id,
                    error_rate=float(error_rate),
                    molecules=maximum_molecules,
                    reads_per_molecule=maximum_reads,
                )
                scores = score_archive(
                    presence_model,
                    archive,
                    device=device,
                    batch_size=batch_size,
                    type_model=type_model,
                )
                for setting, molecules, reads in _settings(config):
                    rows.append(
                        _record_for_prefix(
                            seed,
                            category,
                            archive_id,
                            float(error_rate),
                            setting,
                            molecules,
                            reads,
                            scores,
                            thresholds,
                        )
                    )
            print(
                f"evaluation seed={seed} category={category} error_rate={float(error_rate):.2f} "
                f"archives={count}",
                flush=True,
            )

    prediction_path = output_directory / f"predictions_seed_{seed}.csv"
    _write_rows(prediction_path, rows)
    default_rows = [row for row in rows if row["setting"] == "default"]
    summary = summarize_predictions(default_rows, thresholds)
    summary.update(
        {
            "seed": seed,
            "thresholds": asdict(thresholds),
            "archives_per_category_per_error": count,
            "prediction_csv": str(prediction_path),
        }
    )
    _save_json(output_directory / f"metrics_seed_{seed}.json", summary)
    curve_rows = build_curve_rows(rows, thresholds)
    _write_rows(output_directory / f"curves_seed_{seed}.csv", curve_rows)
    write_curve_svgs(curve_rows, output_directory, seed)
    return summary


def _safe_rate(numerator: int, denominator: int) -> float:
    return float("nan") if denominator == 0 else numerator / denominator


def summarize_predictions(rows: list[dict[str, Any]], thresholds: Thresholds) -> dict[str, Any]:
    if not rows:
        raise ValueError("cannot summarize empty predictions")
    categories = np.asarray([row["category"] for row in rows], dtype=object)
    presence_truth = np.asarray([row["true_presence"] for row in rows], dtype=np.int64)
    presence_prediction = np.asarray([row["predicted_presence"] for row in rows], dtype=np.int64)
    ecc_scores = np.asarray([row["ecc_score"] for row in rows], dtype=np.float64)

    no_ecc_mask = np.isin(categories, NO_ECC_TYPES)
    fountain_mask = categories == UNKNOWN_CODE_TYPE
    known_mask = np.isin(categories, KNOWN_CODE_TYPES)
    stage2_mask = known_mask | fountain_mask
    stage2_unknown_truth = fountain_mask[stage2_mask].astype(np.int64)
    stage2_energy = np.asarray([row["stage2_energy"] for row in rows], dtype=np.float64)[stage2_mask]
    known_detection_truth = known_mask[stage2_mask].astype(np.int64)
    known_detection_score = -stage2_energy

    type_truth = categories[known_mask]
    type_prediction = np.asarray([row["standalone_type"] for row in rows], dtype=object)[known_mask]
    type_confusion = confusion_matrix(type_truth, type_prediction, KNOWN_CODE_TYPES)
    _, type_macro_f1 = f1_from_confusion(type_confusion)

    final_truth = np.asarray([row["true_final_label"] for row in rows], dtype=object)
    final_prediction = np.asarray([row["predicted_final_label"] for row in rows], dtype=object)
    final_confusion = confusion_matrix(final_truth, final_prediction, END_TO_END_LABELS)
    _, final_macro_f1 = f1_from_confusion(final_confusion)
    false_known_mask = no_ecc_mask | fountain_mask
    false_known = np.isin(final_prediction[false_known_mask], KNOWN_CODE_TYPES)

    no_ecc_specificity = _safe_rate(
        int(np.sum(presence_prediction[no_ecc_mask] == 0)), int(np.sum(no_ecc_mask))
    )
    fountain_ecc_recall = _safe_rate(
        int(np.sum(presence_prediction[fountain_mask] == 1)), int(np.sum(fountain_mask))
    )
    stage2_unknown_prediction = stage2_energy > thresholds.unknown_energy
    fountain_unknown_recall = _safe_rate(
        int(np.sum(stage2_unknown_prediction[stage2_unknown_truth == 1])),
        int(np.sum(stage2_unknown_truth == 1)),
    )
    known_acceptance = _safe_rate(
        int(np.sum(~stage2_unknown_prediction[known_detection_truth == 1])),
        int(np.sum(known_detection_truth == 1)),
    )
    false_known_rate = float(np.mean(false_known)) if false_known.size else float("nan")
    stage2_auroc = binary_auroc(stage2_unknown_truth, stage2_energy)

    return {
        "stage1": {
            "accuracy": accuracy_score(presence_truth, presence_prediction),
            "macro_f1": macro_f1_score(presence_truth, presence_prediction, (0, 1)),
            "auroc": binary_auroc(presence_truth, ecc_scores),
            "no_ecc_specificity": no_ecc_specificity,
            "fountain_ecc_recall": fountain_ecc_recall,
        },
        "stage2": {
            "unknown_auroc": stage2_auroc,
            "unknown_aupr": binary_aupr(stage2_unknown_truth, stage2_energy),
            "fpr_at_95_tpr_known_detection": fpr_at_tpr(
                known_detection_truth, known_detection_score, target_tpr=0.95
            ),
            "fountain_unknown_recall": fountain_unknown_recall,
            "known_acceptance": known_acceptance,
        },
        "known_type": {
            "macro_f1": type_macro_f1,
            "labels": list(KNOWN_CODE_TYPES),
            "confusion_matrix": type_confusion.tolist(),
        },
        "end_to_end": {
            "accuracy": accuracy_score(final_truth, final_prediction),
            "macro_f1": final_macro_f1,
            "labels": list(END_TO_END_LABELS),
            "confusion_matrix": final_confusion.tolist(),
            "no_ecc_or_fountain_output_as_known_rate": false_known_rate,
        },
        "feasibility": {
            "stage1_fountain_recall_at_least_0_70": bool(fountain_ecc_recall >= 0.70),
            "no_ecc_specificity_at_least_0_80": bool(no_ecc_specificity >= 0.80),
            "stage2_fountain_unknown_recall_at_least_0_70": bool(
                fountain_unknown_recall >= 0.70
            ),
            "stage2_auroc_at_least_0_80": bool(stage2_auroc >= 0.80),
            "false_known_rate_at_most_0_20": bool(false_known_rate <= 0.20),
            "fountain_failure_interpretation": (
                None
                if fountain_ecc_recall >= 0.70
                else "原文逐 read 投票不足以检测跨分子喷泉结构"
            ),
        },
    }


def _compact_metrics(rows: list[dict[str, Any]], thresholds: Thresholds) -> dict[str, float]:
    summary = summarize_predictions(rows, thresholds)
    return {
        "stage1_macro_f1": float(summary["stage1"]["macro_f1"]),
        "no_ecc_specificity": float(summary["stage1"]["no_ecc_specificity"]),
        "fountain_ecc_recall": float(summary["stage1"]["fountain_ecc_recall"]),
        "stage2_unknown_auroc": float(summary["stage2"]["unknown_auroc"]),
        "fountain_unknown_recall": float(summary["stage2"]["fountain_unknown_recall"]),
        "known_type_macro_f1": float(summary["known_type"]["macro_f1"]),
        "end_to_end_accuracy": float(summary["end_to_end"]["accuracy"]),
    }


def build_curve_rows(rows: list[dict[str, Any]], thresholds: Thresholds) -> list[dict[str, Any]]:
    curves: list[dict[str, Any]] = []
    default = [row for row in rows if row["setting"] == "default"]
    for error_rate in sorted({float(row["error_rate"]) for row in default}):
        subset = [row for row in default if float(row["error_rate"]) == error_rate]
        curves.append({"curve": "error_rate", "x": error_rate, **_compact_metrics(subset, thresholds)})
    q_rows = [row for row in rows if row["setting"] == "q_sweep"]
    for reads in sorted({int(row["q"]) for row in q_rows}):
        subset = [row for row in q_rows if int(row["q"]) == reads]
        curves.append({"curve": "q", "x": reads, **_compact_metrics(subset, thresholds)})
    m_rows = [row for row in rows if row["setting"] == "M_sweep"]
    for molecules in sorted({int(row["M"]) for row in m_rows}):
        subset = [row for row in m_rows if int(row["M"]) == molecules]
        curves.append({"curve": "M", "x": molecules, **_compact_metrics(subset, thresholds)})
    return curves


def aggregate_seed_metrics(
    metric_paths: Iterable[str | Path],
    output_path: str | Path,
    bootstrap_resamples: int = 10_000,
) -> dict[str, Any]:
    payloads = [json.loads(Path(path).read_text(encoding="utf-8")) for path in metric_paths]
    if not payloads:
        raise ValueError("no seed metrics were supplied")
    keys = {
        "stage1.accuracy": lambda item: item["stage1"]["accuracy"],
        "stage1.macro_f1": lambda item: item["stage1"]["macro_f1"],
        "stage1.auroc": lambda item: item["stage1"]["auroc"],
        "stage1.no_ecc_specificity": lambda item: item["stage1"]["no_ecc_specificity"],
        "stage1.fountain_ecc_recall": lambda item: item["stage1"]["fountain_ecc_recall"],
        "stage2.unknown_auroc": lambda item: item["stage2"]["unknown_auroc"],
        "stage2.unknown_aupr": lambda item: item["stage2"]["unknown_aupr"],
        "stage2.fpr_at_95_tpr_known_detection": lambda item: item["stage2"][
            "fpr_at_95_tpr_known_detection"
        ],
        "stage2.fountain_unknown_recall": lambda item: item["stage2"][
            "fountain_unknown_recall"
        ],
        "stage2.known_acceptance": lambda item: item["stage2"]["known_acceptance"],
        "known_type.macro_f1": lambda item: item["known_type"]["macro_f1"],
        "end_to_end.accuracy": lambda item: item["end_to_end"]["accuracy"],
        "end_to_end.macro_f1": lambda item: item["end_to_end"]["macro_f1"],
        "end_to_end.false_known_rate": lambda item: item["end_to_end"][
            "no_ecc_or_fountain_output_as_known_rate"
        ],
    }
    rng = np.random.default_rng(stable_seed("aggregate-bootstrap", len(payloads)))
    metrics: dict[str, Any] = {}
    for name, getter in keys.items():
        values = np.asarray([float(getter(payload)) for payload in payloads], dtype=np.float64)
        bootstrap = np.empty(bootstrap_resamples, dtype=np.float64)
        for index in range(bootstrap_resamples):
            bootstrap[index] = np.mean(rng.choice(values, size=values.size, replace=True))
        metrics[name] = {
            "values": values.tolist(),
            "mean": float(np.mean(values)),
            "std": float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
            "bootstrap_95_ci": [
                float(np.quantile(bootstrap, 0.025)),
                float(np.quantile(bootstrap, 0.975)),
            ],
        }
    result = {
        "seeds": [int(payload["seed"]) for payload in payloads],
        "bootstrap_unit": "training seed",
        "bootstrap_resamples": bootstrap_resamples,
        "metrics": metrics,
    }
    _save_json(Path(output_path), result)
    return result
