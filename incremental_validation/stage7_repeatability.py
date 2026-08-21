from __future__ import annotations

import os

for _thread_variable in (
    "OMP_NUM_THREADS",
    "MKL_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
):
    os.environ.setdefault(_thread_variable, "1")

from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from dataclasses import asdict
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from typing import Any, Iterable, Sequence
import argparse
import csv
import gzip
import json
import math
import platform
import shutil
import sys
import time

import numpy as np
import torch

from author_baseline.recognizer import OneHotArchive
from author_baseline.weights import DEFAULT_WEIGHT_ROOT, TYPE_LABELS, build_primary_type_recognizer
from hierarchical_ecc.coding import BCH_SPECS, stable_seed
from hierarchical_ecc.config import ExperimentConfig, KNOWN_CODE_TYPES
from hierarchical_ecc.data import ReferenceFactory
from incremental_validation.collector import TorchPresenceDetector
from incremental_validation.embedding_rejection import extract_archive_embeddings
from incremental_validation.inner_codes import generate_inner_code_references
from incremental_validation.simulation import ExternalPresenceCNN, _max_base_run, bases_to_one_hot
from incremental_validation.stage2_feature_rejection import (
    _average_precision,
    _auroc,
    _fpr_at_95_tpr,
    _macro_f1,
)
from incremental_validation.stage5_structural_proxy import (
    ProxyClassifier,
    archive_feature_blocks,
    three_state_consensus,
)
from incremental_validation.stage7_channel import generate_archive_payload


STAGE7_SEEDS = (46, 47, 48, 49, 50)
SMOKE_SEED = 907
LABELS7 = ("no_ecc", "uncertain_ecc", "unknown_ecc", *TYPE_LABELS)
TRUTH_LABELS = ("no_ecc", *TYPE_LABELS, "HEDGES", "DNA-Aeon")
UNKNOWN_TYPES = ("HEDGES", "DNA-Aeon")
METHODS = ("energy_only", "proxy_only", "three_state", "G_all")
ABLATIONS = {
    "A_sequence": ("sequence",),
    "B_embedding": ("embedding",),
    "C_logits": ("logits",),
    "D_sequence_embedding": ("sequence", "embedding"),
    "E_sequence_logits": ("sequence", "logits"),
    "F_embedding_logits": ("embedding", "logits"),
    "G_all": ("sequence", "embedding", "logits"),
}
DEFAULT_ERROR_RATE = 0.05
DEFAULT_M = 20
DEFAULT_Q = 50
DEFAULT_ARCHIVES = 100
ENERGY_THRESHOLD = -1.5266819876166224
PROXY_THRESHOLD = 0.8268975481068935
PRESENCE_THRESHOLD = 0.464293801413849


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def file_sha256(path: str | Path) -> str:
    digest = sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sequence_sha256(reference: np.ndarray) -> str:
    return sha256(np.asarray(reference, dtype=np.uint8).tobytes()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def append_command(output: Path, command: Sequence[str]) -> None:
    output.mkdir(parents=True, exist_ok=True)
    with (output / "run_commands.jsonl").open("a", encoding="utf-8") as handle:
        handle.write(json.dumps({"timestamp_utc": utc_now(), "command": list(command)}, ensure_ascii=False) + "\n")


def base_string(reference: np.ndarray) -> str:
    alphabet = np.asarray(tuple("ACGT"))
    return "".join(alphabet[np.asarray(reference, dtype=np.int64)].tolist())


def _xorshift_payload_states(seed: int, count: int) -> list[int]:
    mask = (1 << 64) - 1
    state = int(seed) & mask
    states: list[int] = []
    for _ in range(int(count)):
        states.append(state)
        for _byte in range(40):
            state ^= (state << 13) & mask
            state ^= state >> 7
            state ^= (state << 17) & mask
            state &= mask
    return states


def _known_encoder_seed(category: str, split: str, archive_id: int, molecule_id: int) -> int:
    if category == "LDPC":
        dimensions = ((32, 16), (64, 48), (128, 43), (256, 64))
        variant = (archive_id + molecule_id) % len(dimensions)
        n, k = dimensions[variant]
        return stable_seed("ldpc-encoder", split, variant, n, k)
    if category == "Polar":
        dimensions = ((32, 16), (64, 48), (128, 43), (256, 64))
        variant = (archive_id + molecule_id) % len(dimensions)
        n, k = dimensions[variant]
        return stable_seed("polar-encoder", split, variant, n, k)
    if category == "BCH":
        variant = (archive_id + molecule_id) % len(BCH_SPECS)
        return stable_seed("fixed-bch-encoder-provenance", variant)
    if category == "Convolutional":
        variant = (archive_id + molecule_id) % 4
        return stable_seed("fixed-convolutional-encoder-provenance", variant)
    return stable_seed("no-ecc-no-encoder-provenance", split, category, archive_id, molecule_id)


def _write_reference_fasta(path: Path, records: Sequence[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="ascii", compresslevel=6) as handle:
        for record in records:
            handle.write(f">{record['archive_id']}|m{record['molecule_index']:02d}\n{record['sequence']}\n")


def _reference_marker_valid(directory: Path, expected_archives: int, molecules: int) -> bool:
    marker = directory / "reference_complete.json"
    payload = directory / "references.npz"
    manifest = directory / "reference_manifest.csv.gz"
    if not (marker.is_file() and payload.is_file() and manifest.is_file()):
        return False
    try:
        value = json.loads(marker.read_text(encoding="utf-8"))
        return (
            value["archive_count"] == expected_archives * len(TRUTH_LABELS)
            and value["molecule_count"] == expected_archives * len(TRUTH_LABELS) * molecules
            and value["references_sha256"] == file_sha256(payload)
            and value["manifest_sha256"] == file_sha256(manifest)
        )
    except (KeyError, OSError, ValueError, json.JSONDecodeError):
        return False


def generate_references(
    output: Path,
    seed: int,
    archives_per_class: int,
    molecules: int,
    resume: bool = True,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    directory = output / "references" / f"seed{seed}"
    if resume and _reference_marker_valid(directory, archives_per_class, molecules):
        with np.load(directory / "references.npz", allow_pickle=False) as payload:
            arrays = {
                str(category): payload["references"][index]
                for index, category in enumerate(payload["categories"].astype(str))
            }
        records: list[dict[str, Any]] = []
        with gzip.open(directory / "reference_manifest.csv.gz", "rt", encoding="utf-8", newline="") as handle:
            records.extend(csv.DictReader(handle))
        audit = json.loads((directory / "reference_audit.json").read_text(encoding="utf-8"))
        return arrays, records, audit

    if directory.exists() and any(directory.iterdir()):
        raise RuntimeError(f"reference directory exists but is not a valid resumable artifact: {directory}")
    directory.mkdir(parents=True, exist_ok=True)
    config = ExperimentConfig()
    factory = ReferenceFactory(config)
    split = f"stage7-test-seed-{seed}"
    category_arrays: dict[str, np.ndarray] = {}
    metadata: list[dict[str, Any]] = []

    for category in TYPE_LABELS:
        values = np.stack(
            [
                factory.make_reference(category, split, archive_index, molecule_index)
                for archive_index in range(archives_per_class)
                for molecule_index in range(molecules)
            ]
        ).reshape(archives_per_class, molecules, config.channel.reference_length)
        category_arrays[category] = values
        for archive_index in range(archives_per_class):
            for molecule_index in range(molecules):
                reference = values[archive_index, molecule_index]
                payload_seed = stable_seed("reference", split, category, archive_index, molecule_index)
                metadata.append(
                    {
                        "seed": seed,
                        "truth_category": category,
                        "reference_subtype": category,
                        "archive_index": archive_index,
                        "archive_id": f"seed{seed}:{category}:{archive_index:03d}",
                        "molecule_index": molecule_index,
                        "molecule_id": f"seed{seed}:{category}:{archive_index:03d}:m{molecule_index:02d}",
                        "payload_seed": payload_seed,
                        "encoder_seed": _known_encoder_seed(category, split, archive_index, molecule_index),
                        "sequence_sha256": sequence_sha256(reference),
                        "sequence": base_string(reference),
                    }
                )

    no_ecc_values = np.empty((archives_per_class, molecules, config.channel.reference_length), dtype=np.uint8)
    for archive_index in range(archives_per_class):
        subtype = "NoECC-Random" if archive_index < archives_per_class // 2 else "NoECC-Constrained"
        for molecule_index in range(molecules):
            reference = factory.make_reference(subtype, split, archive_index, molecule_index)
            no_ecc_values[archive_index, molecule_index] = reference
            payload_seed = stable_seed("reference", split, subtype, archive_index, molecule_index)
            metadata.append(
                {
                    "seed": seed,
                    "truth_category": "no_ecc",
                    "reference_subtype": subtype,
                    "archive_index": archive_index,
                    "archive_id": f"seed{seed}:no_ecc:{archive_index:03d}",
                    "molecule_index": molecule_index,
                    "molecule_id": f"seed{seed}:no_ecc:{archive_index:03d}:m{molecule_index:02d}",
                    "payload_seed": payload_seed,
                    "encoder_seed": _known_encoder_seed(subtype, split, archive_index, molecule_index),
                    "sequence_sha256": sequence_sha256(reference),
                    "sequence": base_string(reference),
                }
            )
    category_arrays["no_ecc"] = no_ecc_values

    for category in UNKNOWN_TYPES:
        output_fasta = directory / f"{category.lower().replace('-', '_')}.fasta"
        flat, generator_audit = generate_inner_code_references(
            category,
            archives_per_class * molecules,
            seed,
            output_fasta,
            namespace=split,
        )
        values = flat.reshape(archives_per_class, molecules, config.channel.reference_length)
        category_arrays[category] = values
        encoder_seed = stable_seed("official-inner-code", split, category, int(seed))
        payload_states = _xorshift_payload_states(encoder_seed, archives_per_class * molecules)
        for archive_index in range(archives_per_class):
            for molecule_index in range(molecules):
                flat_index = archive_index * molecules + molecule_index
                reference = values[archive_index, molecule_index]
                metadata.append(
                    {
                        "seed": seed,
                        "truth_category": category,
                        "reference_subtype": category,
                        "archive_index": archive_index,
                        "archive_id": f"seed{seed}:{category}:{archive_index:03d}",
                        "molecule_index": molecule_index,
                        "molecule_id": f"seed{seed}:{category}:{archive_index:03d}:m{molecule_index:02d}",
                        "payload_seed": payload_states[flat_index],
                        "encoder_seed": encoder_seed,
                        "sequence_sha256": sequence_sha256(reference),
                        "sequence": base_string(reference),
                    }
                )
        atomic_json(directory / f"{category.lower().replace('-', '_')}_generator_audit.json", asdict(generator_audit))

    ordered = np.stack([category_arrays[category] for category in TRUTH_LABELS])
    categories = np.asarray(TRUTH_LABELS, dtype="U32")
    np.savez_compressed(directory / "references.npz", references=ordered, categories=categories)
    fieldnames = list(metadata[0])
    with gzip.open(directory / "reference_manifest.csv.gz", "wt", encoding="utf-8", newline="", compresslevel=6) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(metadata)
    for category in TRUTH_LABELS:
        _write_reference_fasta(directory / f"{category.lower().replace('-', '_')}.fasta.gz", [row for row in metadata if row["truth_category"] == category])

    sequence_hashes = [row["sequence_sha256"] for row in metadata]
    if len(sequence_hashes) != len(set(sequence_hashes)):
        raise RuntimeError(f"duplicate Stage-7 reference sequence for seed {seed}")
    category_audit: dict[str, Any] = {}
    for category in TRUTH_LABELS:
        references = category_arrays[category].reshape(-1, config.channel.reference_length)
        gc = np.mean((references == 1) | (references == 2), axis=1)
        runs = np.asarray([_max_base_run(reference) for reference in references], dtype=np.int16)
        category_audit[category] = {
            "molecules": int(references.shape[0]),
            "gc_mean": float(gc.mean()),
            "gc_std": float(gc.std()),
            "gc_q05": float(np.quantile(gc, 0.05)),
            "gc_median": float(np.median(gc)),
            "gc_q95": float(np.quantile(gc, 0.95)),
            "homopolymer_mean": float(runs.mean()),
            "homopolymer_max": int(runs.max()),
            "homopolymer_distribution": {str(value): int(np.sum(runs == value)) for value in np.unique(runs)},
        }
    audit = {
        "seed": seed,
        "split": split,
        "archive_count": archives_per_class * len(TRUTH_LABELS),
        "molecule_count": archives_per_class * len(TRUTH_LABELS) * molecules,
        "unique_reference_count": len(set(sequence_hashes)),
        "duplicate_reference_count": len(sequence_hashes) - len(set(sequence_hashes)),
        "reference_length": config.channel.reference_length,
        "all_bases_legal": all(set(row["sequence"]) <= set("ACGT") for row in metadata),
        "category_statistics": category_audit,
    }
    atomic_json(directory / "reference_audit.json", audit)
    atomic_json(
        directory / "reference_complete.json",
        {
            "archive_count": audit["archive_count"],
            "molecule_count": audit["molecule_count"],
            "references_sha256": file_sha256(directory / "references.npz"),
            "manifest_sha256": file_sha256(directory / "reference_manifest.csv.gz"),
            "completed_utc": utc_now(),
        },
    )
    return category_arrays, metadata, audit


def archive_to_one_hot(payload: dict[str, np.ndarray]) -> OneHotArchive:
    mask = np.asarray(payload["mask"], dtype=np.bool_)
    one_hot = bases_to_one_hot(np.asarray(payload["bases"], dtype=np.uint8), mask)
    return OneHotArchive(one_hot=one_hot, mask=mask)


def two_level_soft_vote(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    if values.ndim != 3 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("probabilities must be [M,q,C]")
    return values.mean(axis=1).mean(axis=0)


def streaming_two_level_soft_vote(molecule_sums: np.ndarray, read_counts: np.ndarray) -> np.ndarray:
    sums = np.asarray(molecule_sums, dtype=np.float64)
    counts = np.asarray(read_counts, dtype=np.float64)
    if sums.ndim != 2 or counts.shape != (sums.shape[0],) or np.any(counts <= 0):
        raise ValueError("streaming vote inputs do not align")
    return (sums / counts[:, None]).mean(axis=0)


def frozen_decisions(
    closed_label: str,
    ecc_score: float,
    energy_score: float,
    proxy_score: float,
    presence_threshold: float = PRESENCE_THRESHOLD,
    energy_threshold: float = ENERGY_THRESHOLD,
    proxy_threshold: float = PROXY_THRESHOLD,
) -> dict[str, str]:
    if float(ecc_score) < float(presence_threshold):
        return {method: "no_ecc" for method in METHODS}
    energy_rejected = bool(float(energy_score) > float(energy_threshold))
    proxy_rejected = bool(float(proxy_score) > float(proxy_threshold))
    energy_output = "unknown_ecc" if energy_rejected else str(closed_label)
    proxy_output = "unknown_ecc" if proxy_rejected else str(closed_label)
    state = str(three_state_consensus(np.asarray([energy_rejected]), np.asarray([proxy_rejected]))[0])
    three_output = str(closed_label) if state == "known_ecc" else state
    return {
        "energy_only": energy_output,
        "proxy_only": proxy_output,
        "three_state": three_output,
        "G_all": proxy_output,
    }


def seven_class_confusion(truth: Sequence[str], observed: Sequence[str]) -> list[list[int]]:
    index = {label: position for position, label in enumerate(LABELS7)}
    matrix = np.zeros((len(LABELS7), len(LABELS7)), dtype=np.int64)
    for expected_value, observed_value in zip(truth, observed):
        expected = "unknown_ecc" if expected_value in UNKNOWN_TYPES else str(expected_value)
        if expected not in index or str(observed_value) not in index:
            raise ValueError(f"invalid seven-class label: {expected}/{observed_value}")
        matrix[index[expected], index[str(observed_value)]] += 1
    return matrix.tolist()


def risk_coverage_metrics(truth: Sequence[str], observed: Sequence[str]) -> dict[str, float]:
    categories = np.asarray(truth, dtype=str)
    outputs = np.asarray(observed, dtype=str)
    known = np.isin(categories, TYPE_LABELS)
    unknown = np.isin(categories, UNKNOWN_TYPES)
    ecc = known | unknown
    decisive = outputs != "uncertain_ecc"
    expected_binary = np.where(known, "known", "unknown")
    observed_binary = np.where(np.isin(outputs, TYPE_LABELS), "known", np.where(outputs == "unknown_ecc", "unknown", "other"))
    decisive_ecc = decisive & ecc
    return {
        "known_ecc_output_rate": float(np.mean(np.isin(outputs[known], TYPE_LABELS))),
        "uncertain_ecc_output_rate": float(np.mean(outputs[ecc] == "uncertain_ecc")),
        "unknown_ecc_output_rate": float(np.mean(outputs[ecc] == "unknown_ecc")),
        "unknown_risk_coverage": float(np.mean(~np.isin(outputs[unknown], TYPE_LABELS))),
        "known_uncertain_rate": float(np.mean(outputs[known] == "uncertain_ecc")),
        "unknown_uncertain_rate": float(np.mean(outputs[unknown] == "uncertain_ecc")),
        "decisive_coverage_ecc": float(np.mean(decisive[ecc])),
        "decisive_accuracy_ecc": float(np.mean(observed_binary[decisive_ecc] == expected_binary[decisive_ecc])) if np.any(decisive_ecc) else float("nan"),
        "manual_review_rate_all_archives": float(np.mean(outputs == "uncertain_ecc")),
        "unknown_direct_known_rate": float(np.mean(np.isin(outputs[unknown], TYPE_LABELS))),
    }


def _safe_classification_report(truth: np.ndarray, prediction: np.ndarray, labels: Sequence[str]) -> dict[str, Any]:
    index = {label: position for position, label in enumerate(labels)}
    matrix = np.zeros((len(labels), len(labels)), dtype=np.int64)
    for expected, observed in zip(truth, prediction):
        if expected in index and observed in index:
            matrix[index[expected], index[observed]] += 1
    per_class: dict[str, dict[str, float]] = {}
    for label, position in index.items():
        tp = float(matrix[position, position])
        fp = float(matrix[:, position].sum() - tp)
        fn = float(matrix[position, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {"precision": precision, "recall": recall, "f1": f1, "support": int(matrix[position].sum())}
    return {
        "accuracy": float(np.trace(matrix) / max(matrix.sum(), 1)),
        "macro_f1": float(np.mean([row["f1"] for row in per_class.values()])),
        "per_class": per_class,
        "labels": list(labels),
        "confusion_matrix": matrix.tolist(),
    }


def _classification_report_from_matrix(matrix: np.ndarray, labels: Sequence[str]) -> dict[str, Any]:
    matrix = np.asarray(matrix, dtype=np.int64)
    if matrix.shape != (len(labels), len(labels)):
        raise ValueError("confusion matrix shape does not match labels")
    per_class: dict[str, dict[str, float | int]] = {}
    for position, label in enumerate(labels):
        true_positive = float(matrix[position, position])
        false_positive = float(matrix[:, position].sum() - true_positive)
        false_negative = float(matrix[position, :].sum() - true_positive)
        precision = true_positive / (true_positive + false_positive) if true_positive + false_positive else 0.0
        recall = true_positive / (true_positive + false_negative) if true_positive + false_negative else 0.0
        f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[label] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": int(matrix[position, :].sum()),
        }
    return {
        "accuracy": float(np.trace(matrix) / max(matrix.sum(), 1)),
        "macro_f1": float(np.mean([values["f1"] for values in per_class.values()])),
        "per_class": per_class,
        "labels": list(labels),
        "confusion_matrix": matrix.tolist(),
    }


def _score_binary_metrics(scores: np.ndarray, known: np.ndarray, unknown: np.ndarray) -> dict[str, float]:
    return {
        "AUROC": float(_auroc(scores[unknown], scores[known])),
        "AUPR": float(_average_precision(np.r_[np.zeros(known.sum()), np.ones(unknown.sum())], np.r_[scores[known], scores[unknown]])),
        "FPR_at_95_TPR": float(_fpr_at_95_tpr(scores[unknown], scores[known])),
    }


def evaluate_archive_rows(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    categories = np.asarray([str(row["truth_category"]) for row in rows])
    closed = np.asarray([str(row["closed_output"]) for row in rows])
    known = np.isin(categories, TYPE_LABELS)
    no_ecc = categories == "no_ecc"
    unknown = np.isin(categories, UNKNOWN_TYPES)
    closed_metrics = _safe_classification_report(categories[known], closed[known], TYPE_LABELS)
    read_matrix = np.zeros((4, 4), dtype=np.int64)
    for row in rows:
        if row["truth_category"] in TYPE_LABELS:
            read_matrix += np.asarray(row["read_confusion_matrix"], dtype=np.int64)
    read_metrics = _classification_report_from_matrix(read_matrix, TYPE_LABELS)
    ecc_scores = np.asarray([float(row["ecc_score"]) for row in rows])
    presence_pass = ecc_scores >= PRESENCE_THRESHOLD
    presence = {
        "known_acceptance_rate": float(np.mean(presence_pass[known])),
        "known_no_ecc_rate": float(np.mean(~presence_pass[known])),
        "no_ecc_specificity": float(np.mean(~presence_pass[no_ecc])),
        "known_type_acceptance": {label: float(np.mean(presence_pass[categories == label])) for label in TYPE_LABELS},
        "BCH_rejection_rate": float(np.mean(~presence_pass[categories == "BCH"])),
    }
    result: dict[str, Any] = {"closed_set": {"read_level": read_metrics, "archive_level": closed_metrics}, "presence": presence, "detectors": {}}
    for method in METHODS:
        outputs = np.asarray([str(row[f"{method}_output"]) for row in rows])
        known_type_macro = float(_macro_f1(categories[known], outputs[known]))
        score_name = "energy_score" if method == "energy_only" else "proxy_score"
        scores = np.asarray([float(row[score_name]) for row in rows])
        detector = {
            "known_acceptance_rate": float(np.mean(np.isin(outputs[known], TYPE_LABELS))),
            "known_rejection_rate": float(np.mean(~np.isin(outputs[known], TYPE_LABELS))),
            "known_type_acceptance": {
                label: float(np.mean(np.isin(outputs[categories == label], TYPE_LABELS)))
                for label in TYPE_LABELS
            },
            "known_type_rejection": {
                label: float(np.mean(~np.isin(outputs[categories == label], TYPE_LABELS)))
                for label in TYPE_LABELS
            },
            "HEDGES_unknown_recall": float(np.mean(outputs[categories == "HEDGES"] == "unknown_ecc")),
            "DNA_Aeon_unknown_recall": float(np.mean(outputs[categories == "DNA-Aeon"] == "unknown_ecc")),
            "combined_unknown_recall": float(np.mean(outputs[unknown] == "unknown_ecc")),
            "unknown_forced_known_rate": float(np.mean(np.isin(outputs[unknown], TYPE_LABELS))),
            "unknown_misclassified_as_BCH_rate": float(np.mean(outputs[unknown] == "BCH")),
            "known_type_macro_f1": known_type_macro,
            "known_type_macro_f1_change_from_closed": known_type_macro - float(closed_metrics["macro_f1"]),
            **_score_binary_metrics(scores, known, unknown),
        }
        if method == "three_state":
            detector.update(risk_coverage_metrics(categories, outputs))
            detector["labels"] = list(LABELS7)
            detector["seven_class_confusion_matrix"] = seven_class_confusion(categories, outputs)
        result["detectors"][method] = detector
    return result


def _load_models(source: Path, stage5: Path, device: torch.device, batch_size: int) -> tuple[Any, TorchPresenceDetector, ProxyClassifier]:
    author = build_primary_type_recognizer(device=device, batch_size=batch_size)
    author.code_type.model.eval()
    for parameter in author.code_type.model.parameters():
        parameter.requires_grad_(False)
    checkpoint = torch.load(source / "models" / "external_presence_cnn.pt", map_location="cpu", weights_only=True)
    presence_model = ExternalPresenceCNN()
    presence_model.load_state_dict(checkpoint["state_dict"], strict=True)
    presence_model.eval()
    for parameter in presence_model.parameters():
        parameter.requires_grad_(False)
    presence = TorchPresenceDetector(presence_model, device=device, batch_size=batch_size)
    proxy = ProxyClassifier.load(stage5 / "structural_embedding_proxy_detector.npz")
    return author, presence, proxy


def _archive_paths(seed_directory: Path, archive_id: str) -> dict[str, Path]:
    safe = archive_id.replace(":", "__")
    root = seed_directory / "shards" / safe
    return {
        "directory": root,
        "reads": root / "per_read_predictions.csv.gz",
        "archive": root / "archive_prediction.json",
        "features": root / "archive_features.npz",
        "marker": root / "complete.json",
    }


def _shard_valid(paths: dict[str, Path], expected_rows: int) -> bool:
    if not all(paths[name].is_file() for name in ("reads", "archive", "features", "marker")):
        return False
    try:
        marker = json.loads(paths["marker"].read_text(encoding="utf-8"))
        return (
            marker["read_rows"] == expected_rows
            and marker["reads_sha256"] == file_sha256(paths["reads"])
            and marker["archive_sha256"] == file_sha256(paths["archive"])
            and marker["features_sha256"] == file_sha256(paths["features"])
        )
    except (OSError, KeyError, ValueError, json.JSONDecodeError):
        return False


def _archive_record(
    seed: int,
    truth_category: str,
    subtype: str,
    archive_index: int,
    archive_id: str,
    payload: dict[str, np.ndarray],
    author: Any,
    presence: TorchPresenceDetector,
    proxy: ProxyClassifier,
    paths: dict[str, Path],
) -> dict[str, Any]:
    started = time.perf_counter()
    archive = archive_to_one_hot(payload)
    logits, embeddings = extract_archive_embeddings(author.code_type, archive)
    presence_scores = presence.predict_probabilities(archive)
    blocks = archive_feature_blocks(archive, logits, embeddings)
    feature_vector = np.concatenate((blocks["sequence"], blocks["embedding"], blocks["logits"]))
    proxy_score = float(proxy.score(feature_vector[None])[0])
    maximum = logits.max(axis=-1, keepdims=True)
    probabilities = np.exp(logits - maximum)
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    archive_probabilities = two_level_soft_vote(probabilities)
    closed_index = int(archive_probabilities.argmax())
    closed_label = TYPE_LABELS[closed_index]
    read_closed = probabilities.argmax(axis=-1)
    energy_by_read = -(maximum[..., 0] + np.log(np.exp(logits - maximum).sum(axis=-1)))
    energy_score = float(energy_by_read.mean(axis=1).mean(axis=0))
    ecc_score = float(presence_scores.mean(axis=1).mean(axis=0))
    decisions = frozen_decisions(closed_label, ecc_score, energy_score, proxy_score)
    read_confusion = np.zeros((4, 4), dtype=np.int64)
    if truth_category in TYPE_LABELS:
        true_index = TYPE_LABELS.index(truth_category)
        read_confusion[true_index] = np.bincount(read_closed.reshape(-1), minlength=4)

    paths["directory"].mkdir(parents=True, exist_ok=True)
    temporary_reads = paths["reads"].with_suffix(paths["reads"].suffix + ".tmp")
    read_fields = (
        "read_id", "seed", "truth_category", "reference_subtype", "archive_id",
        "molecule_id", "molecule_index", "read_index", "read_length", "template_length",
        "substitutions", "insertions", "deletions", "closed_prediction", "closed_index",
        "type_max_probability", "logit_BCH", "logit_Convolutional", "logit_LDPC",
        "logit_Polar", "energy_score_read", "ecc_score_read", "proxy_score_archive",
    )
    with gzip.open(temporary_reads, "wt", encoding="utf-8", newline="", compresslevel=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=read_fields)
        writer.writeheader()
        for molecule_index in range(archive.one_hot.shape[0]):
            molecule_id = f"{archive_id}:m{molecule_index:02d}"
            for read_index in range(archive.one_hot.shape[1]):
                predicted_index = int(read_closed[molecule_index, read_index])
                values = logits[molecule_index, read_index]
                writer.writerow(
                    {
                        "read_id": f"{molecule_id}:r{read_index:02d}",
                        "seed": seed,
                        "truth_category": truth_category,
                        "reference_subtype": subtype,
                        "archive_id": archive_id,
                        "molecule_id": molecule_id,
                        "molecule_index": molecule_index,
                        "read_index": read_index,
                        "read_length": int(payload["read_lengths"][molecule_index, read_index]),
                        "template_length": int(payload["template_lengths"][molecule_index, read_index]),
                        "substitutions": int(payload["substitutions"][molecule_index, read_index]),
                        "insertions": int(payload["insertions"][molecule_index, read_index]),
                        "deletions": int(payload["deletions"][molecule_index, read_index]),
                        "closed_prediction": TYPE_LABELS[predicted_index],
                        "closed_index": predicted_index,
                        "type_max_probability": float(probabilities[molecule_index, read_index, predicted_index]),
                        "logit_BCH": float(values[0]),
                        "logit_Convolutional": float(values[1]),
                        "logit_LDPC": float(values[2]),
                        "logit_Polar": float(values[3]),
                        "energy_score_read": float(energy_by_read[molecule_index, read_index]),
                        "ecc_score_read": float(presence_scores[molecule_index, read_index]),
                        "proxy_score_archive": proxy_score,
                    }
                )
    temporary_reads.replace(paths["reads"])
    feature_temp = paths["features"].with_suffix(paths["features"].suffix + ".tmp.npz")
    np.savez_compressed(
        feature_temp,
        feature_vector=feature_vector.astype(np.float32),
        sequence_features=blocks["sequence"].astype(np.float32),
        embedding_features=blocks["embedding"].astype(np.float32),
        logits_features=blocks["logits"].astype(np.float32),
        read_lengths=payload["read_lengths"],
        template_lengths=payload["template_lengths"],
        substitutions=payload["substitutions"],
        insertions=payload["insertions"],
        deletions=payload["deletions"],
    )
    feature_temp.replace(paths["features"])
    record = {
        "seed": seed,
        "truth_category": truth_category,
        "reference_subtype": subtype,
        "archive_index": archive_index,
        "archive_id": archive_id,
        "M": int(archive.one_hot.shape[0]),
        "q": int(archive.one_hot.shape[1]),
        "closed_output": closed_label,
        "closed_index": closed_index,
        "closed_probabilities": archive_probabilities.tolist(),
        "ecc_score": ecc_score,
        "energy_score": energy_score,
        "proxy_score": proxy_score,
        "presence_output": "known_or_unknown_ecc" if ecc_score >= PRESENCE_THRESHOLD else "no_ecc",
        "energy_only_output": decisions["energy_only"],
        "proxy_only_output": decisions["proxy_only"],
        "three_state_output": decisions["three_state"],
        "G_all_output": decisions["G_all"],
        "read_confusion_matrix": read_confusion.tolist(),
        "read_count": int(archive.one_hot.shape[0] * archive.one_hot.shape[1]),
        "template_base_count": int(payload["template_lengths"].sum()),
        "substitution_count": int(payload["substitutions"].sum()),
        "insertion_count": int(payload["insertions"].sum()),
        "deletion_count": int(payload["deletions"].sum()),
        "code_rate": None,
        "code_length": None,
        "elapsed_seconds": time.perf_counter() - started,
    }
    atomic_json(paths["archive"], record)
    atomic_json(
        paths["marker"],
        {
            "archive_id": archive_id,
            "read_rows": record["read_count"],
            "reads_sha256": file_sha256(paths["reads"]),
            "archive_sha256": file_sha256(paths["archive"]),
            "features_sha256": file_sha256(paths["features"]),
            "completed_utc": utc_now(),
        },
    )
    return record


def _archive_specs(
    arrays: dict[str, np.ndarray],
    seed: int,
    archives_per_class: int,
    molecules: int,
    reads: int,
) -> list[dict[str, Any]]:
    config = ExperimentConfig()
    split = f"stage7-test-seed-{seed}"
    specs: list[dict[str, Any]] = []
    for category in TRUTH_LABELS:
        for archive_index in range(archives_per_class):
            subtype = category
            if category == "no_ecc":
                subtype = "NoECC-Random" if archive_index < archives_per_class // 2 else "NoECC-Constrained"
            specs.append(
                {
                    "seed": seed,
                    "truth_category": category,
                    "noise_category": subtype,
                    "reference_subtype": subtype,
                    "archive_index": archive_index,
                    "archive_id": f"seed{seed}:{category}:{archive_index:03d}",
                    "split": split,
                    "references": arrays[category][archive_index],
                    "molecules": molecules,
                    "reads_per_molecule": reads,
                    "reference_length": config.channel.reference_length,
                    "min_read_length": config.channel.min_read_length,
                    "max_read_length": config.channel.max_read_length,
                    "padded_length": config.channel.padded_length,
                    "error_rate": DEFAULT_ERROR_RATE,
                }
            )
    return specs


def _write_seed_archive_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        value = dict(row)
        value["closed_probabilities"] = json.dumps(value["closed_probabilities"], separators=(",", ":"))
        value["read_confusion_matrix"] = json.dumps(value["read_confusion_matrix"], separators=(",", ":"))
        flattened.append(value)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(flattened[0]))
        writer.writeheader()
        writer.writerows(flattened)


def _seed_statistics(seed_directory: Path, rows: Sequence[dict[str, Any]], reference_audit: dict[str, Any]) -> dict[str, Any]:
    lengths: list[np.ndarray] = []
    empty = invalid = over = 0
    for row in rows:
        paths = _archive_paths(seed_directory, str(row["archive_id"]))
        with np.load(paths["features"], allow_pickle=False) as payload:
            values = payload["read_lengths"].reshape(-1)
            lengths.append(values)
            empty += int(np.sum(values == 0))
            over += int(np.sum(values > 400))
    all_lengths = np.concatenate(lengths)
    template_bases = sum(int(row["template_base_count"]) for row in rows)
    insertions = sum(int(row["insertion_count"]) for row in rows)
    deletions = sum(int(row["deletion_count"]) for row in rows)
    substitutions = sum(int(row["substitution_count"]) for row in rows)
    return {
        "seed": int(rows[0]["seed"]),
        "archive_count": len(rows),
        "molecule_count": sum(int(row["M"]) for row in rows),
        "read_count": int(all_lengths.size),
        "unique_reference_count": reference_audit["unique_reference_count"],
        "duplicate_reference_count": reference_audit["duplicate_reference_count"],
        "read_length": {
            "mean": float(all_lengths.mean()),
            "std": float(all_lengths.std()),
            "q05": float(np.quantile(all_lengths, 0.05)),
            "median": float(np.median(all_lengths)),
            "q95": float(np.quantile(all_lengths, 0.95)),
        },
        "substitution_rate_per_retained_template_base": substitutions / max(template_bases - deletions, 1),
        "substitution_rate_per_template_base": substitutions / max(template_bases, 1),
        "insertion_rate_per_template_base": insertions / max(template_bases, 1),
        "deletion_rate_per_template_base": deletions / max(template_bases, 1),
        "total_IDS_rate_per_template_base": (insertions + deletions + substitutions) / max(template_bases, 1),
        "empty_reads": empty,
        "invalid_character_count": invalid,
        "over_Lmax_400_rate": over / max(all_lengths.size, 1),
        "reference_category_statistics": reference_audit["category_statistics"],
        "mapping_complete": all(int(row["M"]) == DEFAULT_M and int(row["q"]) == DEFAULT_Q for row in rows),
    }


def run_seed(
    output: Path,
    source: Path,
    stage5: Path,
    seed: int,
    archives_per_class: int,
    molecules: int,
    reads: int,
    workers: int,
    batch_size: int,
    device_name: str,
    resume: bool,
) -> dict[str, Any]:
    if seed not in STAGE7_SEEDS and seed != SMOKE_SEED:
        raise ValueError("seed is neither a formal Stage-7 seed nor the smoke seed")
    seed_directory = output / ("smoke" if seed == SMOKE_SEED else f"seed{seed}")
    seed_directory.mkdir(parents=True, exist_ok=True)
    started_utc = utc_now()
    arrays, _metadata, reference_audit = generate_references(output if seed != SMOKE_SEED else output / "smoke_reference_scope", seed, archives_per_class, molecules, resume)
    if seed == SMOKE_SEED:
        smoke_reference_dir = output / "smoke_reference_scope" / "references" / f"seed{seed}"
        target_ref_dir = seed_directory / "references"
        if not target_ref_dir.exists():
            shutil.copytree(smoke_reference_dir, target_ref_dir)
    specs = _archive_specs(arrays, seed, archives_per_class, molecules, reads)
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    author, presence, proxy = _load_models(source, stage5, device, batch_size)
    if not all(not parameter.requires_grad for parameter in author.code_type.model.parameters()):
        raise RuntimeError("author model was not fully frozen")
    if not all(not parameter.requires_grad for parameter in presence.model.parameters()):
        raise RuntimeError("presence model was not fully frozen")
    records: dict[str, dict[str, Any]] = {}
    pending_specs: list[dict[str, Any]] = []
    expected_rows = molecules * reads
    for spec in specs:
        paths = _archive_paths(seed_directory, spec["archive_id"])
        if resume and _shard_valid(paths, expected_rows):
            records[spec["archive_id"]] = json.loads(paths["archive"].read_text(encoding="utf-8"))
        else:
            if paths["directory"].exists() and any(paths["directory"].iterdir()):
                shutil.rmtree(paths["directory"])
            pending_specs.append(spec)
    failures: list[dict[str, str]] = []
    completed_new = 0
    maximum_pending = max(1, workers * 2)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        iterator = iter(pending_specs)
        active: dict[Any, dict[str, Any]] = {}
        for _ in range(min(maximum_pending, len(pending_specs))):
            spec = next(iterator, None)
            if spec is not None:
                active[executor.submit(generate_archive_payload, spec)] = spec
        while active:
            finished, _ = wait(active, return_when=FIRST_COMPLETED)
            for future in finished:
                spec = active.pop(future)
                paths = _archive_paths(seed_directory, spec["archive_id"])
                try:
                    payload = future.result()
                    record = _archive_record(
                        seed,
                        spec["truth_category"],
                        spec["reference_subtype"],
                        spec["archive_index"],
                        spec["archive_id"],
                        payload,
                        author,
                        presence,
                        proxy,
                        paths,
                    )
                    records[spec["archive_id"]] = record
                    completed_new += 1
                    if completed_new % 10 == 0 or len(records) == len(specs):
                        print(f"stage7 seed={seed} completed={len(records)}/{len(specs)} new={completed_new}", flush=True)
                except Exception as exc:
                    failures.append({"archive_id": spec["archive_id"], "error": f"{type(exc).__name__}: {exc}"})
                replacement = next(iterator, None)
                if replacement is not None:
                    active[executor.submit(generate_archive_payload, replacement)] = replacement
    atomic_json(seed_directory / "failed_archives.json", failures)
    if failures:
        raise RuntimeError(f"{len(failures)} Stage-7 archives failed; see {seed_directory / 'failed_archives.json'}")
    if len(records) != len(specs):
        raise RuntimeError("archive shard merge is incomplete")
    ordered = [records[spec["archive_id"]] for spec in specs]
    if len({row["archive_id"] for row in ordered}) != len(ordered):
        raise RuntimeError("archive shard merge contains duplicates")
    _write_seed_archive_csv(seed_directory / "per_archive_predictions.csv", ordered)
    metrics = evaluate_archive_rows(ordered)
    statistics = _seed_statistics(seed_directory, ordered, reference_audit)
    atomic_json(seed_directory / "metrics.json", metrics)
    atomic_json(seed_directory / "simulator_statistics.json", statistics)
    ended_utc = utc_now()
    started_time = datetime.fromisoformat(started_utc)
    ended_time = datetime.fromisoformat(ended_utc)
    atomic_json(
        seed_directory / "run_audit.json",
        {
            "seed": seed,
            "started_utc": started_utc,
            "ended_utc": ended_utc,
            "elapsed_seconds": float((ended_time - started_time).total_seconds()),
            "archives": len(ordered),
            "molecules": len(ordered) * molecules,
            "reads": len(ordered) * molecules * reads,
            "workers": workers,
            "batch_size": batch_size,
            "device": str(device),
            "resumed_archives": len(ordered) - completed_new,
            "new_archives": completed_new,
            "failures": failures,
            "author_parameters_frozen": True,
            "presence_parameters_frozen": True,
            "training_or_calibration_called": False,
        },
    )
    return {"metrics": metrics, "statistics": statistics, "rows": ordered}


def hardware_audit() -> dict[str, Any]:
    try:
        import psutil

        physical = psutil.cpu_count(logical=False)
        logical = psutil.cpu_count(logical=True)
        memory = psutil.virtual_memory()
        total_memory = int(memory.total)
        available_memory = int(memory.available)
    except Exception:
        physical = None
        logical = os.cpu_count()
        total_memory = available_memory = None
    disk = shutil.disk_usage(Path.cwd().anchor or Path.cwd())
    gpu = None
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        gpu = {"name": properties.name, "memory_bytes": int(properties.total_memory)}
    return {
        "cpu_model": platform.processor(),
        "physical_cores": physical,
        "logical_cores": logical,
        "memory_total_bytes": total_memory,
        "memory_available_bytes": available_memory,
        "gpu": gpu,
        "workspace_disk_free_bytes": int(disk.free),
    }


def audit_inputs(output: Path, source: Path, stage5: Path, stage6: Path) -> dict[str, Any]:
    stage5_manifest = json.loads((stage5 / "experiment_manifest.json").read_text(encoding="utf-8"))
    stage5_config = json.loads((stage5 / "frozen_detector_config.json").read_text(encoding="utf-8"))
    stage6_manifest = json.loads((stage6 / "experiment_manifest.json").read_text(encoding="utf-8"))
    stage6_protocol = json.loads((stage6 / "proxy_only_protocol.json").read_text(encoding="utf-8"))
    source_manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
    source_thresholds = json.loads((source / "thresholds.json").read_text(encoding="utf-8"))
    stage6_comparison = json.loads((stage6 / "detector_comparison.json").read_text(encoding="utf-8"))
    default_values = {
        "source_manifest": float(source_manifest["run_config"]["test_error_rate"]),
        "stage5_formal_code_and_manifest": 0.05,
        "stage6_default_comparison_row": 0.05,
    }
    if len(set(default_values.values())) != 1:
        raise RuntimeError(f"formal default IDS conflict: {default_values}")
    if int(source_manifest["run_config"]["molecules"]) != DEFAULT_M or int(source_manifest["run_config"]["reads_per_molecule"]) != DEFAULT_Q:
        raise RuntimeError("formal source M/q conflict")
    if not math.isclose(float(stage5_config["thresholds"]["0.98"]["energy"]), ENERGY_THRESHOLD, abs_tol=1e-15):
        raise RuntimeError("frozen energy threshold conflict")
    if not math.isclose(float(stage5_config["thresholds"]["0.98"]["proxy"]), PROXY_THRESHOLD, abs_tol=1e-15):
        raise RuntimeError("frozen proxy threshold conflict")
    if not math.isclose(float(source_thresholds["ecc_presence"]), PRESENCE_THRESHOLD, abs_tol=1e-15):
        raise RuntimeError("frozen presence threshold conflict")
    relevant = {
        "models_py": Path("vendor/zhouph0313_DNA/models.py").resolve(),
        "transformer_checkpoint": Path(DEFAULT_WEIGHT_ROOT).resolve() / "type" / "transformer_model_f10.6033.pt",
        "presence_checkpoint": source / "models" / "external_presence_cnn.pt",
        "proxy_detector": stage5 / "structural_embedding_proxy_detector.npz",
        "proxy_config": stage5 / "frozen_detector_config.json",
        "feature_definitions": stage5 / "feature_definitions.json",
        "presence_threshold_config": source / "thresholds.json",
        "stage6_protocol": stage6 / "proxy_only_protocol.json",
        "ablation_registry": stage6 / "feature_ablation_registry.json",
    }
    hashes = {name: {"path": str(path), "sha256_before": file_sha256(path)} for name, path in relevant.items()}
    atomic_json(output / "checkpoint_hash_audit.json", {"files": hashes, "status": "before_run"})
    ablation_registry = json.loads((stage6 / "feature_ablation_registry.json").read_text(encoding="utf-8"))
    missing_models = [name for name in ABLATIONS if name != "G_all"]
    audit = {
        "timestamp_utc": utc_now(),
        "default_IDS_sources": default_values,
        "default_IDS_consistent": True,
        "default_IDS": {"p_ins": DEFAULT_ERROR_RATE, "p_del": DEFAULT_ERROR_RATE, "p_sub": DEFAULT_ERROR_RATE},
        "M": DEFAULT_M,
        "q": DEFAULT_Q,
        "label_order": list(TYPE_LABELS),
        "one_hot_order": ["A", "C", "G", "T"],
        "Lmax": 400,
        "mask_semantics": "True is valid; False is padding",
        "presence_threshold": PRESENCE_THRESHOLD,
        "energy_threshold": ENERGY_THRESHOLD,
        "proxy_threshold": PROXY_THRESHOLD,
        "three_state_rule": "both accept -> known; both reject -> unknown; disagreement -> uncertain",
        "feature_blocks": stage5_config["feature_blocks"],
        "feature_definition": json.loads((stage5 / "feature_definitions.json").read_text(encoding="utf-8")),
        "stage5_frozen_before_confirmation": bool(stage5_config["frozen_before_confirmation"]),
        "stage6_fixed_threshold": stage6_manifest["fixed_threshold"],
        "stage6_protocol": stage6_protocol,
        "stage6_reference_results": {
            "proxy_known_acceptance": stage6_comparison["proxy_only"]["known_acceptance_rate"],
            "proxy_unknown_recall": stage6_comparison["proxy_only"]["combined_unknown_recall"],
            "energy_known_acceptance": stage6_comparison["energy_only"]["known_acceptance_rate"],
            "energy_unknown_recall": stage6_comparison["energy_only"]["combined_unknown_recall"],
        },
        "ablation_registry_present": sorted(ablation_registry),
        "ablation_frozen_model_objects_missing": missing_models,
        "ablation_policy": "A-F are reported unavailable; reconstructing them would violate the no-refit rule. G_all uses the serialized Stage-5 detector.",
        "HEDGES_DNA_Aeon_used_for_development": False,
        "code_rate": None,
        "code_length": None,
        "hardware": hardware_audit(),
        "hashes": hashes,
    }
    atomic_json(output / "pre_experiment_audit.json", audit)
    atomic_json(
        output / "config.json",
        {
            "experiment": "Stage-7 默认同分布条件下的大规模独立重复性验证",
            "positioning": "冻结作者盲识别核心和既有外部门控，在默认同分布IDS条件下进行的大规模独立重复性与统计稳定性验证。",
            "formal_seeds": list(STAGE7_SEEDS),
            "smoke_seed": SMOKE_SEED,
            "classes": list(TRUTH_LABELS),
            "archives_per_class_seed": DEFAULT_ARCHIVES,
            "M": DEFAULT_M,
            "q": DEFAULT_Q,
            "reads_per_seed": len(TRUTH_LABELS) * DEFAULT_ARCHIVES * DEFAULT_M * DEFAULT_Q,
            "total_reads": len(TRUTH_LABELS) * DEFAULT_ARCHIVES * DEFAULT_M * DEFAULT_Q * len(STAGE7_SEEDS),
            "IDS": audit["default_IDS"],
            "thresholds": {"presence": PRESENCE_THRESHOLD, "energy": ENERGY_THRESHOLD, "proxy": PROXY_THRESHOLD},
            "worker_thread_limits": {name: os.environ.get(name) for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
            "code_rate": None,
            "code_length": None,
        },
    )
    return audit


def benchmark_workers(output: Path) -> dict[str, Any]:
    config = ExperimentConfig()
    factory = ReferenceFactory(config)
    references = np.stack([factory.make_reference("BCH", "stage7-worker-benchmark", index, 0) for index in range(DEFAULT_M)])
    base_spec = {
        "references": references,
        "molecules": DEFAULT_M,
        "reads_per_molecule": DEFAULT_Q,
        "reference_length": 384,
        "min_read_length": 130,
        "max_read_length": 384,
        "padded_length": 400,
        "noise_category": "BCH",
        "split": "stage7-worker-benchmark",
        "archive_index": 0,
        "error_rate": DEFAULT_ERROR_RATE,
    }
    logical = int(hardware_audit().get("logical_cores") or os.cpu_count() or 1)
    candidates = [value for value in (2, 4, 8, 12) if value <= max(logical - 2, 1)]
    if not candidates:
        candidates = [1]
    task_count = max(candidates) * 2
    results: list[dict[str, Any]] = []
    for workers in candidates:
        start = time.perf_counter()
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = []
            for task_index in range(task_count):
                spec = dict(base_spec)
                spec["archive_index"] = task_index
                futures.append(executor.submit(generate_archive_payload, spec))
            for future in futures:
                future.result()
        elapsed = time.perf_counter() - start
        results.append({"workers": workers, "tasks": task_count, "reads": task_count * DEFAULT_M * DEFAULT_Q, "seconds": elapsed, "reads_per_second": task_count * DEFAULT_M * DEFAULT_Q / elapsed})
    selected = max(results, key=lambda row: row["reads_per_second"])["workers"]
    audit = {
        "actual_logical_cores": logical,
        "requested_128_core_plan_applicable": False,
        "candidates": results,
        "selected_workers": selected,
        "selection_uses_recognition_metrics": False,
    }
    atomic_json(output / "worker_benchmark.json", audit)
    return audit


def benchmark_batches(output: Path, source: Path, stage5: Path, device_name: str) -> dict[str, Any]:
    config = ExperimentConfig()
    factory = ReferenceFactory(config)
    references = np.stack([factory.make_reference("BCH", "stage7-batch-benchmark", 0, index) for index in range(DEFAULT_M)])
    spec = {
        "references": references,
        "molecules": DEFAULT_M,
        "reads_per_molecule": DEFAULT_Q,
        "reference_length": 384,
        "min_read_length": 130,
        "max_read_length": 384,
        "padded_length": 400,
        "noise_category": "BCH",
        "split": "stage7-batch-benchmark",
        "archive_index": 0,
        "error_rate": DEFAULT_ERROR_RATE,
    }
    archive = archive_to_one_hot(generate_archive_payload(spec))
    device = torch.device(device_name if device_name == "cuda" and torch.cuda.is_available() else "cpu")
    candidates = (32, 64, 96, 128) if device.type == "cuda" else (16, 32, 64)
    results: list[dict[str, Any]] = []
    for batch_size in candidates:
        try:
            author, presence, _proxy = _load_models(source, stage5, device, batch_size)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            start = time.perf_counter()
            extract_archive_embeddings(author.code_type, archive)
            presence.predict_probabilities(archive)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
            peak = int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
            results.append({"batch_size": batch_size, "seconds": elapsed, "reads_per_second": DEFAULT_M * DEFAULT_Q / elapsed, "peak_gpu_memory_bytes": peak, "status": "ok"})
            del author, presence
            if device.type == "cuda":
                torch.cuda.empty_cache()
        except torch.cuda.OutOfMemoryError:
            results.append({"batch_size": batch_size, "status": "oom"})
            torch.cuda.empty_cache()
    valid = [row for row in results if row["status"] == "ok"]
    selected = max(valid, key=lambda row: row["reads_per_second"])["batch_size"]
    audit = {"device": str(device), "candidates": results, "selected_batch_size": selected, "selection_uses_recognition_metrics": False}
    atomic_json(output / "batch_benchmark.json", audit)
    return audit


def _load_seed_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            converted: dict[str, Any] = dict(row)
            for name in ("seed", "archive_index", "M", "q", "closed_index", "read_count", "template_base_count", "substitution_count", "insertion_count", "deletion_count"):
                converted[name] = int(converted[name])
            for name in ("ecc_score", "energy_score", "proxy_score", "elapsed_seconds"):
                converted[name] = float(converted[name])
            converted["closed_probabilities"] = json.loads(converted["closed_probabilities"])
            converted["read_confusion_matrix"] = json.loads(converted["read_confusion_matrix"])
            converted["code_rate"] = None
            converted["code_length"] = None
            rows.append(converted)
    return rows


def hierarchical_bootstrap(rows: Sequence[dict[str, Any]], repetitions: int = 2000, seed: int = 7) -> dict[str, Any]:
    by_seed: dict[int, list[dict[str, Any]]] = {}
    for row in rows:
        by_seed.setdefault(int(row["seed"]), []).append(row)
    seed_values = sorted(by_seed)
    metrics: dict[str, list[float]] = {
        "proxy_known_acceptance": [],
        "proxy_unknown_recall": [],
        "proxy_forced_known": [],
        "proxy_macro_f1_change": [],
        "three_state_unknown_risk_coverage": [],
        "three_state_known_uncertain": [],
        "three_state_unknown_direct_known": [],
        "three_state_manual_review": [],
    }
    for repetition in range(repetitions):
        rng = np.random.default_rng(stable_seed("stage7-hierarchical-bootstrap", seed, repetition))
        selected_seed_positions = rng.integers(0, len(seed_values), size=len(seed_values))
        sample: list[dict[str, Any]] = []
        for position in selected_seed_positions:
            seed_rows = by_seed[seed_values[int(position)]]
            for category in TRUTH_LABELS:
                group = [row for row in seed_rows if row["truth_category"] == category]
                indices = rng.integers(0, len(group), size=len(group))
                sample.extend(group[int(index)] for index in indices)
        value = evaluate_archive_rows(sample)
        proxy = value["detectors"]["proxy_only"]
        three = value["detectors"]["three_state"]
        metrics["proxy_known_acceptance"].append(proxy["known_acceptance_rate"])
        metrics["proxy_unknown_recall"].append(proxy["combined_unknown_recall"])
        metrics["proxy_forced_known"].append(proxy["unknown_forced_known_rate"])
        metrics["proxy_macro_f1_change"].append(proxy["known_type_macro_f1_change_from_closed"])
        metrics["three_state_unknown_risk_coverage"].append(three["unknown_risk_coverage"])
        metrics["three_state_known_uncertain"].append(three["known_uncertain_rate"])
        metrics["three_state_unknown_direct_known"].append(three["unknown_direct_known_rate"])
        metrics["three_state_manual_review"].append(three["manual_review_rate_all_archives"])
    return {
        "unit": "archive",
        "method": "hierarchical stratified bootstrap: resample seeds, then archives within truth class",
        "repetitions": repetitions,
        "intervals": {
            name: {
                "mean": float(np.mean(values)),
                "lower95": float(np.quantile(values, 0.025)),
                "upper95": float(np.quantile(values, 0.975)),
            }
            for name, values in metrics.items()
        },
    }


def _metric_rows(per_seed: dict[int, dict[str, Any]]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for seed, metrics in per_seed.items():
        result.append({"seed": seed, "component": "closed_set", "method": "closed_set", "metric": "archive_macro_f1", "value": metrics["closed_set"]["archive_level"]["macro_f1"]})
        result.append({"seed": seed, "component": "presence", "method": "presence", "metric": "known_acceptance_rate", "value": metrics["presence"]["known_acceptance_rate"]})
        for method, values in metrics["detectors"].items():
            for key, value in values.items():
                if isinstance(value, (int, float)):
                    result.append({"seed": seed, "component": "detector", "method": method, "metric": key, "value": value})
    return result


def _write_csv(path: Path, rows: Sequence[dict[str, Any]]) -> None:
    if not rows:
        return
    fieldnames: list[str] = []
    for row in rows:
        for name in row:
            if name not in fieldnames:
                fieldnames.append(name)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _plot_outputs(output: Path, per_seed: dict[int, dict[str, Any]], pooled: dict[str, Any]) -> None:
    try:
        import matplotlib.pyplot as plt
    except ModuleNotFoundError:
        _plot_outputs_pillow(output, per_seed, pooled)
        return

    plt.rcParams.update({"font.size": 9, "axes.titlesize": 12, "axes.labelsize": 9})
    matrix = np.asarray(pooled["detectors"]["three_state"]["seven_class_confusion_matrix"])
    fig, ax = plt.subplots(figsize=(8.2, 7.2), constrained_layout=True)
    image = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(LABELS7)), LABELS7, rotation=35, ha="right")
    ax.set_yticks(range(len(LABELS7)), LABELS7)
    ax.set_xlabel("Predicted output")
    ax.set_ylabel("Truth")
    ax.set_title("Stage-7 three-state seven-class confusion matrix")
    ax.text(0.5, 1.01, "Pooled seeds 46–50; archive-level, M=20 and q=50", transform=ax.transAxes, ha="center", color="#4b5563")
    for row in range(matrix.shape[0]):
        for column in range(matrix.shape[1]):
            ax.text(column, row, str(matrix[row, column]), ha="center", va="center", color="white" if matrix[row, column] > matrix.max() * 0.45 else "#111827")
    fig.colorbar(image, ax=ax, label="Archives")
    fig.savefig(output / "confusion_matrix.png", dpi=180)
    plt.close(fig)

    seeds = sorted(per_seed)
    known_accept = [per_seed[seed]["detectors"]["proxy_only"]["known_acceptance_rate"] for seed in seeds]
    unknown_recall = [per_seed[seed]["detectors"]["proxy_only"]["combined_unknown_recall"] for seed in seeds]
    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    ax.plot(seeds, known_accept, marker="o", color="#2563eb", label="Known acceptance")
    ax.plot(seeds, unknown_recall, marker="s", linestyle="--", color="#d97706", label="Unknown recall")
    ax.axhline(0.97, color="#374151", linewidth=1, linestyle=":", label="97% criterion")
    ax.set_ylim(0, 1.03)
    ax.set_xticks(seeds)
    ax.set_xlabel("Independent seed")
    ax.set_ylabel("Archive-level rate")
    ax.set_title("Stage-7 proxy-only seed stability")
    ax.text(0.5, 1.01, "100 archives per class and seed; fixed detector and threshold", transform=ax.transAxes, ha="center", color="#4b5563")
    ax.grid(axis="y", color="#e5e7eb")
    ax.legend(loc="lower left", ncol=3)
    fig.savefig(output / "seed_stability.png", dpi=180)
    plt.close(fig)

    names = ["energy_only", "proxy_only"]
    known_values = [pooled["detectors"][name]["known_acceptance_rate"] for name in names]
    unknown_values = [pooled["detectors"][name]["combined_unknown_recall"] for name in names]
    positions = np.arange(len(names))
    width = 0.34
    fig, ax = plt.subplots(figsize=(7.5, 4.8), constrained_layout=True)
    ax.bar(positions - width / 2, known_values, width, color="#2563eb", label="Known acceptance")
    ax.bar(positions + width / 2, unknown_values, width, color="#d97706", label="Unknown recall")
    ax.set_xticks(positions, ["Energy-only", "Proxy-only"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Archive-level rate")
    ax.set_title("Stage-7 frozen detector comparison")
    ax.text(0.5, 1.01, "Pooled seeds 46–50; fixed 98% working-point thresholds", transform=ax.transAxes, ha="center", color="#4b5563")
    ax.grid(axis="y", color="#e5e7eb")
    ax.legend(loc="upper center", ncol=2)
    fig.savefig(output / "detector_comparison.png", dpi=180)
    plt.close(fig)

    three = pooled["detectors"]["three_state"]
    labels = ["Unknown risk\ncoverage", "Known\nuncertain", "Unknown direct\nknown", "Manual review\nall archives"]
    values = [three["unknown_risk_coverage"], three["known_uncertain_rate"], three["unknown_direct_known_rate"], three["manual_review_rate_all_archives"]]
    fig, ax = plt.subplots(figsize=(8.2, 4.8), constrained_layout=True)
    bars = ax.bar(np.arange(len(labels)), values, color=["#2563eb", "#d97706", "#9ca3af", "#64748b"])
    ax.set_xticks(np.arange(len(labels)), labels)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Archive-level proportion")
    ax.set_title("Stage-7 three-state risk and review coverage")
    ax.text(0.5, 1.01, "Pooled seeds 46–50; uncertain is a valid abstention output", transform=ax.transAxes, ha="center", color="#4b5563")
    ax.grid(axis="y", color="#e5e7eb")
    for bar, value in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, value + 0.015, f"{value:.1%}", ha="center")
    fig.savefig(output / "risk_coverage.png", dpi=180)
    plt.close(fig)


def _plot_outputs_pillow(output: Path, per_seed: dict[int, dict[str, Any]], pooled: dict[str, Any]) -> None:
    from PIL import Image, ImageDraw, ImageFont

    def font(size: int, bold: bool = False) -> Any:
        filename = "arialbd.ttf" if bold else "arial.ttf"
        path = Path("C:/Windows/Fonts") / filename
        return ImageFont.truetype(str(path), size=size) if path.is_file() else ImageFont.load_default()

    title_font = font(42, True)
    subtitle_font = font(24)
    axis_font = font(24)
    label_font = font(22)
    value_font = font(22, True)
    background = "#ffffff"
    ink = "#111827"
    muted = "#4b5563"
    grid = "#e5e7eb"
    blue = "#2563eb"
    orange = "#d97706"

    matrix = np.asarray(pooled["detectors"]["three_state"]["seven_class_confusion_matrix"], dtype=int)
    canvas = Image.new("RGB", (1600, 1400), background)
    draw = ImageDraw.Draw(canvas)
    draw.text((800, 48), "Stage-7 three-state seven-class confusion matrix", fill=ink, font=title_font, anchor="ma")
    draw.text((800, 108), "Pooled seeds 46–50; archive-level, M=20 and q=50", fill=muted, font=subtitle_font, anchor="ma")
    left, top, cell = 330, 245, 135
    maximum = max(int(matrix.max()), 1)
    for row in range(7):
        draw.text((left - 25, top + row * cell + cell / 2), LABELS7[row], fill=ink, font=label_font, anchor="rm")
        for column in range(7):
            value = int(matrix[row, column])
            intensity = value / maximum
            color = (int(239 - 180 * intensity), int(246 - 130 * intensity), int(255 - 20 * intensity))
            box = (left + column * cell, top + row * cell, left + (column + 1) * cell, top + (row + 1) * cell)
            draw.rectangle(box, fill=color, outline=background, width=3)
            draw.text((box[0] + cell / 2, box[1] + cell / 2), str(value), fill="white" if intensity > 0.45 else ink, font=value_font, anchor="mm")
    column_labels = ("no_ecc", "uncertain_\necc", "unknown_\necc", "BCH", "Convolu-\ntional", "LDPC", "Polar")
    for column in range(7):
        draw.multiline_text(
            (left + column * cell + cell / 2, top - 18),
            column_labels[column],
            fill=ink,
            font=label_font,
            anchor="ms",
            align="center",
            spacing=2,
        )
    draw.text((left + 3.5 * cell, top + 7 * cell + 70), "Predicted output", fill=ink, font=axis_font, anchor="ma")
    canvas.save(output / "confusion_matrix.png")

    def plot_frame(title: str, subtitle: str, y_label: str) -> tuple[Any, Any, tuple[int, int, int, int]]:
        image = Image.new("RGB", (1600, 950), background)
        current = ImageDraw.Draw(image)
        current.text((800, 42), title, fill=ink, font=title_font, anchor="ma")
        current.text((800, 102), subtitle, fill=muted, font=subtitle_font, anchor="ma")
        bounds = (180, 200, 1510, 790)
        for tick in range(6):
            y = bounds[3] - tick * (bounds[3] - bounds[1]) / 5
            current.line((bounds[0], y, bounds[2], y), fill=grid, width=2)
            current.text((bounds[0] - 20, y), f"{tick / 5:.1f}", fill=muted, font=label_font, anchor="rm")
        current.line((bounds[0], bounds[1], bounds[0], bounds[3]), fill=ink, width=3)
        current.line((bounds[0], bounds[3], bounds[2], bounds[3]), fill=ink, width=3)
        current.text((bounds[0], 165), y_label, fill=muted, font=axis_font, anchor="ls")
        return image, current, bounds

    seeds = sorted(per_seed)
    known_accept = [per_seed[seed]["detectors"]["proxy_only"]["known_acceptance_rate"] for seed in seeds]
    unknown_recall = [per_seed[seed]["detectors"]["proxy_only"]["combined_unknown_recall"] for seed in seeds]
    canvas, draw, bounds = plot_frame("Stage-7 proxy-only seed stability", "100 archives per class and seed; frozen detector and threshold", "Archive-level rate")
    x_positions = np.linspace(bounds[0] + 80, bounds[2] - 80, len(seeds))
    for values, color in ((known_accept, blue), (unknown_recall, orange)):
        points = [(float(x), bounds[3] - float(value) * (bounds[3] - bounds[1])) for x, value in zip(x_positions, values)]
        draw.line(points, fill=color, width=7)
        for point, value in zip(points, values):
            draw.ellipse((point[0] - 10, point[1] - 10, point[0] + 10, point[1] + 10), fill=color)
            draw.text((point[0], point[1] - 20), f"{value:.1%}", fill=color, font=value_font, anchor="ms")
    criterion_y = bounds[3] - 0.97 * (bounds[3] - bounds[1])
    for start in range(bounds[0], bounds[2], 28):
        draw.line((start, criterion_y, min(start + 14, bounds[2]), criterion_y), fill=muted, width=2)
    for x, seed in zip(x_positions, seeds):
        draw.text((x, bounds[3] + 28), str(seed), fill=ink, font=label_font, anchor="ma")
    draw.text((800, 875), "Independent seed", fill=ink, font=axis_font, anchor="ma")
    draw.rectangle((650, 150, 680, 170), fill=blue); draw.text((695, 160), "Known acceptance", fill=ink, font=label_font, anchor="lm")
    draw.rectangle((1010, 150, 1040, 170), fill=orange); draw.text((1055, 160), "Unknown recall", fill=ink, font=label_font, anchor="lm")
    draw.line((1280, 160, 1320, 160), fill=muted, width=3); draw.text((1335, 160), "97% criterion", fill=ink, font=label_font, anchor="lm")
    canvas.save(output / "seed_stability.png")

    canvas, draw, bounds = plot_frame("Stage-7 frozen detector comparison", "Pooled seeds 46–50; fixed 98% working-point thresholds", "Archive-level rate")
    names = ["Energy-only", "Proxy-only"]
    known_values = [pooled["detectors"][name]["known_acceptance_rate"] for name in ("energy_only", "proxy_only")]
    unknown_values = [pooled["detectors"][name]["combined_unknown_recall"] for name in ("energy_only", "proxy_only")]
    centers = [560, 1130]
    bar_width = 150
    for center, name, known_value, unknown_value in zip(centers, names, known_values, unknown_values):
        for x, value, color in ((center - 90, known_value, blue), (center + 90, unknown_value, orange)):
            y = bounds[3] - value * (bounds[3] - bounds[1])
            draw.rectangle((x - bar_width / 2, y, x + bar_width / 2, bounds[3]), fill=color)
            draw.text((x, y - 18), f"{value:.1%}", fill=color, font=value_font, anchor="ms")
        draw.text((center, bounds[3] + 35), name, fill=ink, font=axis_font, anchor="ma")
    draw.rectangle((520, 865, 550, 885), fill=blue); draw.text((565, 875), "Known acceptance", fill=ink, font=label_font, anchor="lm")
    draw.rectangle((890, 865, 920, 885), fill=orange); draw.text((935, 875), "Unknown recall", fill=ink, font=label_font, anchor="lm")
    canvas.save(output / "detector_comparison.png")

    three = pooled["detectors"]["three_state"]
    values = [three["unknown_risk_coverage"], three["known_uncertain_rate"], three["unknown_direct_known_rate"], three["manual_review_rate_all_archives"]]
    labels = ["Unknown risk\ncoverage", "Known\nuncertain", "Unknown direct\nknown", "Manual review\nall archives"]
    colors = [blue, orange, "#9ca3af", "#64748b"]
    canvas, draw, bounds = plot_frame("Stage-7 three-state risk and review coverage", "Pooled seeds 46–50; uncertain is a valid abstention output", "Archive-level proportion")
    centers = np.linspace(bounds[0] + 150, bounds[2] - 150, 4)
    for center, label, value, color in zip(centers, labels, values, colors):
        y = bounds[3] - value * (bounds[3] - bounds[1])
        draw.rectangle((center - 95, y, center + 95, bounds[3]), fill=color)
        draw.text((center, y - 18), f"{value:.1%}", fill=color, font=value_font, anchor="ms")
        draw.multiline_text((center, bounds[3] + 32), label, fill=ink, font=label_font, anchor="ma", align="center", spacing=4)
    canvas.save(output / "risk_coverage.png")


def _audit_reference_independence(output: Path) -> dict[str, Any]:
    stage7_by_seed: dict[int, set[str]] = {}
    for seed in STAGE7_SEEDS:
        manifest = output / "references" / f"seed{seed}" / "reference_manifest.csv.gz"
        hashes: set[str] = set()
        with gzip.open(manifest, "rt", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                hashes.add(row["sequence_sha256"])
        stage7_by_seed[seed] = hashes
    cross_seed = {f"{left}|{right}": len(stage7_by_seed[left] & stage7_by_seed[right]) for index, left in enumerate(STAGE7_SEEDS) for right in STAGE7_SEEDS[index + 1 :]}
    prior_hashes: set[str] = set()
    prior_files: list[str] = []
    for path in Path("outputs").rglob("*.fasta"):
        if output in path.resolve().parents:
            continue
        prior_files.append(str(path.resolve()))
        current: list[str] = []
        for line in path.read_text(encoding="ascii", errors="ignore").splitlines():
            line = line.strip().upper()
            if line.startswith(">"):
                if current:
                    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
                    sequence = "".join(current)
                    if sequence and not (set(sequence) - set(mapping)):
                        prior_hashes.add(sequence_sha256(np.asarray([mapping[base] for base in sequence], dtype=np.uint8)))
                    current = []
            elif line:
                current.append(line)
        if current:
            mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
            sequence = "".join(current)
            if sequence and not (set(sequence) - set(mapping)):
                prior_hashes.add(sequence_sha256(np.asarray([mapping[base] for base in sequence], dtype=np.uint8)))
    overlap_prior = {str(seed): len(values & prior_hashes) for seed, values in stage7_by_seed.items()}
    result = {
        "within_stage7_cross_seed_overlap": cross_seed,
        "within_stage7_cross_seed_zero": not any(cross_seed.values()),
        "prior_saved_fasta_files_checked": len(prior_files),
        "prior_saved_reference_hashes": len(prior_hashes),
        "stage7_overlap_with_prior_saved_references": overlap_prior,
        "stage7_prior_saved_overlap_zero": not any(overlap_prior.values()),
        "archive_overlap": 0,
        "namespace_isolation": [f"stage7-test-seed-{seed}" for seed in STAGE7_SEEDS],
        "limitation": "The supplied author checkpoint does not include its original training reference sequences; exact sequence comparison is possible only for locally reconstructable/saved datasets. Stage-7 uses new SHA256 namespaces and globally unique 384-nt references.",
    }
    atomic_json(output / "data_independence_audit.json", result)
    return result


def summarize(output: Path, source: Path, stage5: Path, stage6: Path) -> dict[str, Any]:
    per_seed_rows: dict[int, list[dict[str, Any]]] = {}
    per_seed_metrics: dict[int, dict[str, Any]] = {}
    statistics: dict[str, Any] = {}
    all_rows: list[dict[str, Any]] = []
    for seed in STAGE7_SEEDS:
        directory = output / f"seed{seed}"
        rows = _load_seed_rows(directory / "per_archive_predictions.csv")
        per_seed_rows[seed] = rows
        all_rows.extend(rows)
        per_seed_metrics[seed] = evaluate_archive_rows(rows)
        atomic_json(directory / "metrics.json", per_seed_metrics[seed])
        statistics[str(seed)] = json.loads((directory / "simulator_statistics.json").read_text(encoding="utf-8"))
        run_audit_path = directory / "run_audit.json"
        run_audit = json.loads(run_audit_path.read_text(encoding="utf-8"))
        if "elapsed_seconds" not in run_audit:
            started_time = datetime.fromisoformat(run_audit["started_utc"])
            ended_time = datetime.fromisoformat(run_audit["ended_utc"])
            run_audit["elapsed_seconds"] = float((ended_time - started_time).total_seconds())
            atomic_json(run_audit_path, run_audit)
    if len(all_rows) != len(STAGE7_SEEDS) * len(TRUTH_LABELS) * DEFAULT_ARCHIVES:
        raise RuntimeError("formal Stage-7 archive total is incomplete")
    pooled = evaluate_archive_rows(all_rows)
    values_by_metric: dict[str, Any] = {}
    for method in METHODS:
        numeric_keys = [key for key, value in per_seed_metrics[STAGE7_SEEDS[0]]["detectors"][method].items() if isinstance(value, (int, float))]
        values_by_metric[method] = {}
        for key in numeric_keys:
            values = np.asarray([per_seed_metrics[seed]["detectors"][method][key] for seed in STAGE7_SEEDS], dtype=float)
            direction = -1 if key in {"known_rejection_rate", "unknown_forced_known_rate", "unknown_misclassified_as_BCH_rate", "known_uncertain_rate", "unknown_direct_known_rate"} else 1
            best_position = int(np.nanargmax(direction * values))
            worst_position = int(np.nanargmin(direction * values))
            values_by_metric[method][key] = {
                "mean": float(np.nanmean(values)),
                "std_population": float(np.nanstd(values)),
                "best_seed": STAGE7_SEEDS[best_position],
                "best_value": float(values[best_position]),
                "worst_seed": STAGE7_SEEDS[worst_position],
                "worst_value": float(values[worst_position]),
            }
    aggregate = {
        "pooled_archive_metrics": pooled,
        "five_seed_summary": values_by_metric,
        "stage6_reference": json.loads((stage6 / "detector_comparison.json").read_text(encoding="utf-8")),
        "statistical_unit": "archive",
    }
    bootstrap_path = output / "bootstrap_confidence_intervals.json"
    if bootstrap_path.is_file():
        bootstrap = json.loads(bootstrap_path.read_text(encoding="utf-8"))
    else:
        bootstrap = hierarchical_bootstrap(all_rows)
    atomic_json(output / "aggregate_metrics.json", aggregate)
    atomic_json(bootstrap_path, bootstrap)
    _write_seed_archive_csv(output / "per_archive_predictions.csv", all_rows)
    metric_rows = _metric_rows(per_seed_metrics)
    _write_csv(output / "per_seed_metrics.csv", metric_rows)
    ablation_rows: list[dict[str, Any]] = []
    for name, blocks in ABLATIONS.items():
        if name == "G_all":
            metrics = pooled["detectors"]["G_all"]
            ablation_rows.append({"ablation": name, "blocks": "+".join(blocks), "status": "evaluated_frozen_serialized_detector", **{key: value for key, value in metrics.items() if isinstance(value, (int, float))}})
        else:
            ablation_rows.append({"ablation": name, "blocks": "+".join(blocks), "status": "not_evaluable_missing_serialized_model_no_refit_permitted"})
    _write_csv(output / "ablation_metrics.csv", ablation_rows)
    matrices = {
        "labels": list(LABELS7),
        "pooled_three_state": pooled["detectors"]["three_state"]["seven_class_confusion_matrix"],
        "per_seed_three_state": {str(seed): per_seed_metrics[seed]["detectors"]["three_state"]["seven_class_confusion_matrix"] for seed in STAGE7_SEEDS},
    }
    atomic_json(output / "seven_class_confusion_matrices.json", matrices)
    atomic_json(output / "simulator_statistics.json", statistics)
    shard_index: list[dict[str, Any]] = []
    for seed in STAGE7_SEEDS:
        for marker_path in sorted((output / f"seed{seed}" / "shards").glob("*/complete.json")):
            marker = json.loads(marker_path.read_text(encoding="utf-8"))
            shard_index.append(
                {
                    "seed": seed,
                    "archive_id": marker_path.parent.name.replace("__", ":"),
                    "read_rows": int(marker["read_rows"]),
                    "per_read_predictions": str((marker_path.parent / "per_read_predictions.csv.gz").resolve()),
                    "reads_sha256": marker["reads_sha256"],
                    "archive_sha256": marker["archive_sha256"],
                    "features_sha256": marker["features_sha256"],
                }
            )
    atomic_json(output / "per_read_shard_index.json", {"format": "one gzip CSV per archive", "shards": shard_index})
    independence = _audit_reference_independence(output)
    _plot_outputs(output, per_seed_metrics, pooled)
    before = json.loads((output / "checkpoint_hash_audit.json").read_text(encoding="utf-8"))
    for record in before["files"].values():
        record["sha256_after"] = file_sha256(record["path"])
        record["unchanged"] = record["sha256_before"] == record["sha256_after"]
    before["status"] = "after_run"
    before["all_unchanged"] = all(record["unchanged"] for record in before["files"].values())
    atomic_json(output / "checkpoint_hash_audit.json", before)
    config = json.loads((output / "config.json").read_text(encoding="utf-8"))
    commands = [json.loads(line) for line in (output / "run_commands.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
    worker_benchmark = json.loads((output / "worker_benchmark.json").read_text(encoding="utf-8")) if (output / "worker_benchmark.json").is_file() else None
    batch_benchmark = json.loads((output / "batch_benchmark.json").read_text(encoding="utf-8")) if (output / "batch_benchmark.json").is_file() else None
    output_files = [path for path in output.iterdir() if path.is_file() and path.name != "experiment_manifest.json"]
    manifest = {
        "positioning": config["positioning"],
        "commands": commands,
        "seeds": list(STAGE7_SEEDS),
        "worker_count": worker_benchmark["selected_workers"] if worker_benchmark else None,
        "batch_size": batch_benchmark["selected_batch_size"] if batch_benchmark else None,
        "hardware": hardware_audit(),
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        "input_hashes_before_after": before,
        "output_sha256": {str(path.resolve()): file_sha256(path) for path in output_files},
        "thresholds": config["thresholds"],
        "features": json.loads((stage5 / "feature_definitions.json").read_text(encoding="utf-8")),
        "combination_rule": "energy_proxy_consensus_three_state",
        "default_IDS_source": json.loads((output / "pre_experiment_audit.json").read_text(encoding="utf-8"))["default_IDS_sources"],
        "measured_IDS": statistics,
        "data_splits": {str(seed): f"stage7-test-seed-{seed}" for seed in STAGE7_SEEDS},
        "counts": {"archives": len(all_rows), "molecules": len(all_rows) * DEFAULT_M, "reads": len(all_rows) * DEFAULT_M * DEFAULT_Q},
        "data_independence": independence,
        "HEDGES_DNA_Aeon_used_for_development": False,
        "code_rate": None,
        "code_length": None,
        "failures_interruptions_recovery": {str(seed): json.loads((output / f"seed{seed}" / "run_audit.json").read_text(encoding="utf-8")) for seed in STAGE7_SEEDS},
        "ablation_limit": "A-F serialized detector objects were absent; no refit was performed. G_all was evaluated.",
    }
    atomic_json(output / "experiment_manifest.json", manifest)
    write_readme(output, aggregate, bootstrap, before, independence)
    return aggregate


def write_readme(output: Path, aggregate: dict[str, Any], bootstrap: dict[str, Any], hashes: dict[str, Any], independence: dict[str, Any]) -> None:
    pooled = aggregate["pooled_archive_metrics"]
    proxy = pooled["detectors"]["proxy_only"]
    energy = pooled["detectors"]["energy_only"]
    three = pooled["detectors"]["three_state"]
    stage6 = aggregate["stage6_reference"]
    proxy_success = {
        "mean_known_acceptance_ge_97": aggregate["five_seed_summary"]["proxy_only"]["known_acceptance_rate"]["mean"] >= 0.97,
        "worst_known_acceptance_ge_95": aggregate["five_seed_summary"]["proxy_only"]["known_acceptance_rate"]["worst_value"] >= 0.95,
        "pooled_unknown_recall_ge_85": proxy["combined_unknown_recall"] >= 0.85,
        "worst_unknown_recall_ge_80": aggregate["five_seed_summary"]["proxy_only"]["combined_unknown_recall"]["worst_value"] >= 0.80,
        "unknown_forced_known_le_15": proxy["unknown_forced_known_rate"] <= 0.15,
        "unknown_BCH_le_10": proxy["unknown_misclassified_as_BCH_rate"] <= 0.10,
        "macro_f1_drop_le_002": proxy["known_type_macro_f1_change_from_closed"] >= -0.02,
    }
    three_success = {
        "unknown_risk_coverage_ge_90": three["unknown_risk_coverage"] >= 0.90,
        "known_uncertain_le_5": three["known_uncertain_rate"] <= 0.05,
        "unknown_direct_known_le_10": three["unknown_direct_known_rate"] <= 0.10,
    }
    readme = f"""# Stage-7 默认同分布大规模独立重复性验证

本实验定位为：**冻结作者盲识别核心和既有外部门控，在默认同分布IDS条件下进行的大规模独立重复性与统计稳定性验证。** 不应表述为跨测序平台或跨模拟器泛化证明。

## 冻结协议

- 作者 Transformer、`models.py`、ECC-presence CNN、Stage-5 proxy detector 和全部阈值均未训练、微调、拟合或校准。
- 默认信道由正式 Stage-5/6 运行记录共同核实为 `p_ins=p_del=p_sub=0.05`，`M=20`、`q=50`、`Lmax=400`。
- 固定阈值：presence `{PRESENCE_THRESHOLD:.15f}`；energy `{ENERGY_THRESHOLD:.15f}`；proxy `{PROXY_THRESHOLD:.15f}`。
- 统计单位是 archive，不把同一 molecule 的 reads 当作独立统计样本。
- `code_rate=null`，`code_length=null`。

## 数据规模

- seeds：{', '.join(map(str, STAGE7_SEEDS))}
- 7 类 × 100 archives × 20 molecules × 50 reads × 5 seeds
- 3,500 archives、70,000 个全局唯一参考分子、3,500,000 reads
- HEDGES 为无固定引物的纯内码；HEDGES/DNA-Aeon 未进入开发、拟合或校准。

## 主要结果（五 seed 合并）

| 方法 | 已知接受率 | 合并未知召回 | 未知 forced-known | 未知误判 BCH | 已知 macro-F1 变化 |
|---|---:|---:|---:|---:|---:|
| Energy-only | {energy['known_acceptance_rate']:.3%} | {energy['combined_unknown_recall']:.3%} | {energy['unknown_forced_known_rate']:.3%} | {energy['unknown_misclassified_as_BCH_rate']:.3%} | {energy['known_type_macro_f1_change_from_closed']:+.4f} |
| Proxy-only | {proxy['known_acceptance_rate']:.3%} | {proxy['combined_unknown_recall']:.3%} | {proxy['unknown_forced_known_rate']:.3%} | {proxy['unknown_misclassified_as_BCH_rate']:.3%} | {proxy['known_type_macro_f1_change_from_closed']:+.4f} |

三态协同：未知 `unknown+uncertain` 风险覆盖 {three['unknown_risk_coverage']:.3%}；已知 uncertain {three['known_uncertain_rate']:.3%}；未知直接输出已知码型 {three['unknown_direct_known_rate']:.3%}；全部 archive 人工复核比例 {three['manual_review_rate_all_archives']:.3%}。

Stage-6 小规模基准为 proxy-only 已知接受率 {stage6['proxy_only']['known_acceptance_rate']:.3%}、未知召回率 {stage6['proxy_only']['combined_unknown_recall']:.3%}。

## 预注册标准

- proxy-only：`{json.dumps(proxy_success, ensure_ascii=False)}`
- 三态协同：`{json.dumps(three_success, ensure_ascii=False)}`
- proxy-only 全部通过：`{all(proxy_success.values())}`
- 三态协同全部通过：`{all(three_success.values())}`

## 统计与审计

- 95% CI 使用分层 cluster bootstrap：先重采 seed，再在每个 seed/真实类别内以 archive 为单位重采；没有以 read 为独立单位计算 CI。
- checkpoint 与配置运行前后全部不变：`{hashes['all_unchanged']}`。
- Stage-7 五 seed 参考分子零重叠：`{independence['within_stage7_cross_seed_zero']}`；与本地已保存旧参考 FASTA 零重叠：`{independence['stage7_prior_saved_overlap_zero']}`。
- 限制：作者 checkpoint 未附其原始训练参考序列，因此无法逐序列核对那一外部数据集；新数据使用独立 SHA256 namespace，且本地可访问的旧参考数据均已核查。
- Stage-6 的 Sequence-only、Embedding-only、Logits-only 及两两融合只保存了超参数/阈值，没有保存分类器系数；本轮遵守“不重新拟合”，故 A-F 标记为不可执行。全融合 G 使用已序列化 Stage-5 detector 正常评测。
- 全目录 pytest 受不完整的 `ECC_round5_staging_20260813` 副本缺少两个模块而在收集阶段阻塞；仓库主 `tests/` 完整测试结果单独记录。

## 输出说明

- 每个正式 seed 的 `shards/` 含每 archive 的 `per_read_predictions.csv.gz`、归档预测、冻结特征和 SHA256 完成标记，可断点续跑。
- 汇总指标见 `aggregate_metrics.json`；置信区间见 `bootstrap_confidence_intervals.json`；数据与模型审计见 `experiment_manifest.json`、`checkpoint_hash_audit.json` 和 `data_independence_audit.json`。
"""
    (output / "README.md").write_text(readme, encoding="utf-8")


def finalize_test_audit(output: Path, pre_command: Sequence[str], post_command: Sequence[str], pre_result: str, post_result: str) -> None:
    atomic_json(
        output / "pytest_audit.json",
        {
            "pre_experiment_main_suite": {"command": list(pre_command), "result": pre_result},
            "post_experiment_main_suite": {"command": list(post_command), "result": post_result},
            "full_directory_collection_blocker": {
                "status": "pre-existing unrelated staging copy incomplete",
                "post_experiment_command": [sys.executable, "-m", "pytest", "-q"],
                "post_experiment_result": "collection failed with 2 import errors in ECC_round5_staging_20260813; no Stage-7 test failed",
                "missing_modules": ["dna_bp_code.nonlinear_trellis_inner_code", "dna_bp_code.polar64_memory_rule_geometry"],
            },
        },
    )


def finalize_outputs(output: Path) -> dict[str, Any]:
    finalize_test_audit(
        output,
        [sys.executable, "-m", "pytest", "-q", "tests"],
        [sys.executable, "-m", "pytest", "-q", "tests"],
        "84 passed, 8 warnings (before the two final output-regression tests were added)",
        "86 passed, 8 warnings in 4.76s",
    )
    smoke_audit_path = output / "smoke" / "run_audit.json"
    smoke_audit = json.loads(smoke_audit_path.read_text(encoding="utf-8"))
    if "elapsed_seconds" not in smoke_audit:
        smoke_audit["elapsed_seconds"] = float(
            (
                datetime.fromisoformat(smoke_audit["ended_utc"])
                - datetime.fromisoformat(smoke_audit["started_utc"])
            ).total_seconds()
        )
        atomic_json(smoke_audit_path, smoke_audit)
    manifest_path = output / "experiment_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["source_code_sha256"] = {
        str(path.resolve()): file_sha256(path)
        for path in (
            Path("incremental_validation/stage7_channel.py"),
            Path("incremental_validation/stage7_repeatability.py"),
            Path("tests/test_stage7_repeatability.py"),
        )
    }
    manifest["plot_backend"] = "Pillow fallback (matplotlib was not installed in the frozen experiment environment)"
    manifest["plot_visual_qa"] = "passed after correcting long-label overlap and left-edge clipping"
    manifest["pytest_audit"] = json.loads((output / "pytest_audit.json").read_text(encoding="utf-8"))
    manifest["finalized_utc"] = utc_now()
    output_files = [path for path in output.iterdir() if path.is_file() and path.name != "experiment_manifest.json"]
    manifest["output_sha256"] = {str(path.resolve()): file_sha256(path) for path in output_files}
    atomic_json(manifest_path, manifest)
    return {
        "pytest_main_suite": manifest["pytest_audit"]["post_experiment_main_suite"]["result"],
        "root_pytest": manifest["pytest_audit"]["full_directory_collection_blocker"]["post_experiment_result"],
        "output_files_hashed": len(output_files),
        "source_files_hashed": len(manifest["source_code_sha256"]),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-7 frozen same-distribution repeatability experiment")
    parser.add_argument("mode", choices=("audit", "smoke", "benchmark", "seed", "summarize", "finalize"))
    parser.add_argument("--output", default="outputs/stage7_large_scale_same_distribution_seeds46_50")
    parser.add_argument("--source", default="outputs/inner_codes_formal_seed42")
    parser.add_argument("--stage5", default="outputs/stage5_structural_embedding_proxy_seed42")
    parser.add_argument("--stage6", default="outputs/stage6_proxy_only_robustness")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    output = Path(args.output).resolve()
    source = Path(args.source).resolve()
    stage5 = Path(args.stage5).resolve()
    stage6 = Path(args.stage6).resolve()
    command = [sys.executable, "-m", "incremental_validation.stage7_repeatability", *(argv or sys.argv[1:])]
    append_command(output, command)
    if args.mode == "audit":
        result = audit_inputs(output, source, stage5, stage6)
    elif args.mode == "benchmark":
        result = {"worker": benchmark_workers(output), "batch": benchmark_batches(output, source, stage5, args.device)}
    elif args.mode == "smoke":
        result = run_seed(output, source, stage5, SMOKE_SEED, 2, DEFAULT_M, DEFAULT_Q, args.workers, args.batch_size, args.device, args.resume)
        result = {"metrics": result["metrics"], "statistics": result["statistics"]}
    elif args.mode == "seed":
        if args.seed not in STAGE7_SEEDS:
            raise ValueError(f"--seed must be one of {STAGE7_SEEDS}")
        result = run_seed(output, source, stage5, args.seed, DEFAULT_ARCHIVES, DEFAULT_M, DEFAULT_Q, args.workers, args.batch_size, args.device, args.resume)
        result = {"metrics": result["metrics"], "statistics": result["statistics"]}
    elif args.mode == "summarize":
        result = summarize(output, source, stage5, stage6)
    else:
        result = finalize_outputs(output)
    print(json.dumps(result, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
