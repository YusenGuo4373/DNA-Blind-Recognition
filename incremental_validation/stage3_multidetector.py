from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from hashlib import sha256
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
from hierarchical_ecc.coding import bits_to_bases, stable_seed
from hierarchical_ecc.config import ExperimentConfig, KNOWN_CODE_TYPES, NO_ECC_TYPES
from incremental_validation.collector import TorchPresenceDetector
from incremental_validation.comparison import KNOWN_TYPES, IncrementalThresholds
from incremental_validation.embedding_rejection import (
    _closed_indices,
    _fingerprints,
    _known_reference_sets,
    archive_mean_std,
    extract_archive_embeddings,
    fit_pca,
    transform_embeddings,
)
from incremental_validation.inner_codes import archives_from_references, generate_inner_code_references
from incremental_validation.simulation import ExternalPresenceCNN, SimulationRunConfig, _make_archives, audit_molecular_references
from incremental_validation.stage2_feature_rejection import (
    KnownFeatureModel,
    _average_precision,
    _auroc,
    _fpr_at_95_tpr,
    _macro_f1,
    _sha256,
    acceptance_threshold,
    conformal_p_values,
    extract_logit_features,
)


PROXY_FAMILIES = ("ReedSolomon", "Turbo", "LT-XOR")
DETECTORS = (
    "global_energy",
    "pca16_archive_diagonal_minimum",
    "raw128_archive_conformal_maximum_pvalue",
    "logits_diagonal_minimum",
)
TARGETS = (0.98, 0.95, 0.93)
SEVEN_LABELS = ("no_ecc", "uncertain_ecc", "unknown_ecc", *KNOWN_TYPES)


def _gf_mul(left: int, right: int) -> int:
    result = 0
    a, b = int(left), int(right)
    for _ in range(8):
        if b & 1:
            result ^= a
        high = a & 0x80
        a = (a << 1) & 0xFF
        if high:
            a ^= 0x1D
        b >>= 1
    return result


def _rs_reference(rng: np.random.Generator) -> np.ndarray:
    coefficients = rng.integers(0, 256, size=64, dtype=np.uint8)
    codeword = np.empty(96, dtype=np.uint8)
    for position, x_value in enumerate(range(1, 97)):
        value = 0
        for coefficient in coefficients[::-1]:
            value = _gf_mul(value, x_value) ^ int(coefficient)
        codeword[position] = value
    bits = np.unpackbits(codeword, bitorder="big")
    return bits_to_bases(bits)


def _rsc_parity(bits: np.ndarray) -> np.ndarray:
    state = np.zeros(3, dtype=np.uint8)
    parity = np.empty(bits.size, dtype=np.uint8)
    for index, bit in enumerate(bits):
        feedback = int(bit) ^ int(state[1]) ^ int(state[2])
        parity[index] = feedback ^ int(state[0]) ^ int(state[2])
        state[1:] = state[:-1]
        state[0] = feedback
    return parity


def _turbo_reference(rng: np.random.Generator) -> np.ndarray:
    payload = rng.integers(0, 2, size=256, dtype=np.uint8)
    interleaver = rng.permutation(payload.size)
    encoded = np.concatenate((payload, _rsc_parity(payload), _rsc_parity(payload[interleaver])))
    return bits_to_bases(encoded[:768])


def _lt_reference(source: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    degree = int(rng.integers(1, source.shape[0] + 1))
    indices = rng.choice(source.shape[0], size=degree, replace=False)
    droplet = np.bitwise_xor.reduce(source[indices], axis=0)
    return bits_to_bases(droplet)


def generate_proxy_references(
    family: str,
    split: str,
    archives: int,
    molecules: int,
    seed: int,
    output_fasta: str | Path,
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    if family not in PROXY_FAMILIES:
        raise ValueError(f"unsupported proxy family: {family}")
    if family in {"HEDGES", "DNA-Aeon"}:
        raise ValueError("target inner codes are forbidden proxy anomalies")
    output_fasta = Path(output_fasta)
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    references: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    seen: set[bytes] = set()
    for archive_id in range(archives):
        source_seed = stable_seed("proxy-source", seed, split, family, archive_id)
        source_rng = np.random.default_rng(source_seed)
        source = source_rng.integers(0, 2, size=(8, 768), dtype=np.uint8)
        for molecule_id in range(molecules):
            payload_seed = stable_seed("proxy-payload", seed, split, family, archive_id, molecule_id)
            encoder_seed = stable_seed("proxy-encoder", seed, split, family, archive_id, molecule_id)
            for attempt in range(100):
                rng = np.random.default_rng(stable_seed(encoder_seed, attempt))
                if family == "ReedSolomon":
                    reference = _rs_reference(rng)
                elif family == "Turbo":
                    reference = _turbo_reference(rng)
                else:
                    reference = _lt_reference(source, rng)
                encoded = reference.astype(np.uint8).tobytes()
                if encoded not in seen:
                    break
            else:
                raise RuntimeError(f"could not generate unique {family} molecule")
            if reference.shape != (384,) or np.any(reference > 3):
                raise RuntimeError("proxy reference violates 384-nt DNA contract")
            seen.add(encoded)
            references.append(reference)
            metadata.append({
                "family": family,
                "split": split,
                "archive_id": archive_id,
                "molecule_id": molecule_id,
                "payload_seed": int(payload_seed),
                "encoder_seed": int(encoder_seed),
                "source_seed": int(source_seed),
                "sha256": sha256(encoded).hexdigest(),
            })
    array = np.stack(references).astype(np.uint8)
    names = "ACGT"
    with output_fasta.open("w", encoding="ascii", newline="\n") as handle:
        for index, reference in enumerate(array):
            handle.write(f">{family}|{split}|{index}\n")
            handle.write("".join(names[int(value)] for value in reference) + "\n")
    return array, metadata


def apply_multidetector_rule(rejected: np.ndarray, logic: str) -> np.ndarray:
    rejected = np.asarray(rejected, dtype=np.bool_)
    if rejected.ndim != 2 or rejected.shape[1] == 0:
        raise ValueError("rejected decisions must have shape [N,D]")
    count = rejected.sum(axis=1)
    detectors = rejected.shape[1]
    result = np.full(rejected.shape[0], "uncertain_ecc", dtype=object)
    if logic == "consensus":
        result[count == 0] = "known_ecc"
        result[count == detectors] = "unknown_ecc"
    elif logic == "and_reject":
        result[:] = "known_ecc"
        result[count == detectors] = "unknown_ecc"
    elif logic == "or_reject":
        result[:] = "known_ecc"
        result[count > 0] = "unknown_ecc"
    elif logic == "majority":
        result[count < detectors / 2] = "known_ecc"
        result[count > detectors / 2] = "unknown_ecc"
    elif logic == "energy_only":
        if detectors != 1:
            raise ValueError("energy_only requires exactly one detector")
        result[:] = "known_ecc"
        result[rejected[:, 0]] = "unknown_ecc"
    else:
        raise ValueError(f"unsupported combination logic: {logic}")
    return result.astype(str)


def risk_coverage_metrics(states: Sequence[str], truth_unknown: np.ndarray) -> dict[str, float]:
    states = np.asarray(states, dtype=str)
    truth_unknown = np.asarray(truth_unknown, dtype=np.bool_)
    if states.shape != truth_unknown.shape:
        raise ValueError("states/truth must align")
    decisive = states != "uncertain_ecc"
    correct_binary = np.where(truth_unknown, states == "unknown_ecc", states == "known_ecc")
    return {
        "uncertain_rate": float(np.mean(~decisive)),
        "decisive_coverage": float(np.mean(decisive)),
        "decisive_binary_accuracy": float(np.mean(correct_binary[decisive])) if np.any(decisive) else 0.0,
    }


@dataclass
class DetectorSystem:
    pca_mean: np.ndarray
    pca_components: np.ndarray
    pca_diagonal_model: KnownFeatureModel
    raw_conformal_model: KnownFeatureModel
    raw_conformal_calibration: list[np.ndarray]
    logit_model: KnownFeatureModel
    calibration_scores: dict[str, np.ndarray]

    @classmethod
    def from_embedding_npz(cls, path: str | Path) -> "DetectorSystem":
        payload = np.load(Path(path), allow_pickle=False)
        fit_categories = payload["fit_categories"].astype(str)
        calibration_categories = payload["calibration_categories"].astype(str)
        if set(fit_categories) != set(KNOWN_TYPES) or set(calibration_categories) != set(KNOWN_TYPES):
            raise RuntimeError("detector development data must contain only four known ECC classes")
        fit_embeddings = payload["fit_embeddings"]
        calibration_embeddings = payload["calibration_embeddings"]
        fit_logits = payload["fit_logits"]
        calibration_logits = payload["calibration_logits"]
        pca_mean, pca_components, _ = fit_pca(fit_embeddings, 16)
        fit_pca_embeddings = transform_embeddings(fit_embeddings, pca_mean, pca_components)
        cal_pca = transform_embeddings(calibration_embeddings, pca_mean, pca_components)
        pca_model = KnownFeatureModel.fit(archive_mean_std(fit_pca_embeddings), fit_categories)
        pca_cal_score = pca_model.diagonal_distances(archive_mean_std(cal_pca)).min(axis=1)
        raw_model = KnownFeatureModel.fit(archive_mean_std(fit_embeddings), fit_categories)
        raw_cal_distances = raw_model.shrinkage_distances(archive_mean_std(calibration_embeddings))
        raw_calibration = [
            raw_cal_distances[calibration_categories == category, index]
            for index, category in enumerate(KNOWN_TYPES)
        ]
        raw_cal_p = conformal_p_values(raw_cal_distances, raw_calibration)
        raw_cal_score = -raw_cal_p.max(axis=1)
        fit_logit_features = extract_logit_features(fit_logits).features
        cal_bundle = extract_logit_features(calibration_logits)
        logit_model = KnownFeatureModel.fit(fit_logit_features, fit_categories)
        logit_cal_score = logit_model.diagonal_distances(cal_bundle.features).min(axis=1)
        return cls(
            pca_mean, pca_components, pca_model, raw_model, raw_calibration, logit_model,
            {
                "global_energy": cal_bundle.energy,
                "pca16_archive_diagonal_minimum": pca_cal_score,
                "raw128_archive_conformal_maximum_pvalue": raw_cal_score,
                "logits_diagonal_minimum": logit_cal_score,
            },
        )

    def score(self, logits: np.ndarray, embeddings: np.ndarray) -> dict[str, float]:
        logits_batch = np.asarray(logits)[None]
        embeddings_batch = np.asarray(embeddings)[None]
        bundle = extract_logit_features(logits_batch)
        pca = transform_embeddings(embeddings_batch, self.pca_mean, self.pca_components)
        pca_score = self.pca_diagonal_model.diagonal_distances(archive_mean_std(pca)).min(axis=1)[0]
        raw_distances = self.raw_conformal_model.shrinkage_distances(
            archive_mean_std(embeddings_batch)
        )
        raw_p = conformal_p_values(raw_distances, self.raw_conformal_calibration)
        logit_score = self.logit_model.diagonal_distances(bundle.features).min(axis=1)[0]
        return {
            "global_energy": float(bundle.energy[0]),
            "pca16_archive_diagonal_minimum": float(pca_score),
            "raw128_archive_conformal_maximum_pvalue": float(-raw_p.max(axis=1)[0]),
            "logits_diagonal_minimum": float(logit_score),
        }


def build_candidate_registry(thresholds: dict[str, dict[str, float]]) -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for target in TARGETS:
        tag = str(int(round(target * 100)))
        registry[f"t{tag}_energy_only"] = {
            "target": target, "detectors": ["global_energy"], "logic": "energy_only"
        }
        for detector in DETECTORS[1:]:
            registry[f"t{tag}_{detector}_only"] = {
                "target": target, "detectors": [detector], "logic": "energy_only"
            }
        for subset_name, detectors in (
            ("energy_raw", ["global_energy", "raw128_archive_conformal_maximum_pvalue"]),
            ("all4", list(DETECTORS)),
        ):
            for logic in ("consensus", "and_reject", "or_reject", "majority"):
                registry[f"t{tag}_{subset_name}_{logic}"] = {
                    "target": target, "detectors": detectors, "logic": logic
                }
    for value in registry.values():
        value["thresholds"] = {
            detector: thresholds[str(value["target"])][detector]
            for detector in value["detectors"]
        }
    return registry


def candidate_states(
    scores: dict[str, np.ndarray], candidate: dict[str, Any]
) -> tuple[np.ndarray, np.ndarray]:
    rejected = np.column_stack([
        np.asarray(scores[detector]) > candidate["thresholds"][detector]
        for detector in candidate["detectors"]
    ])
    return apply_multidetector_rule(rejected, candidate["logic"]), rejected


def _proxy_family_metrics(states: np.ndarray) -> dict[str, float]:
    states = np.asarray(states, dtype=str)
    return {
        "unknown_recall": float(np.mean(states == "unknown_ecc")),
        "uncertain_rate": float(np.mean(states == "uncertain_ecc")),
        "unknown_or_uncertain_coverage": float(np.mean(states != "known_ecc")),
        "direct_known_error": float(np.mean(states == "known_ecc")),
    }


def select_candidate(
    candidates: dict[str, dict[str, Any]],
    candidate_names: Sequence[str],
    known_states: dict[str, np.ndarray],
    proxy_scores_by_family: dict[str, dict[str, np.ndarray]],
    development_families: Sequence[str],
    mode: str,
) -> str:
    ranked: list[tuple[tuple[float, ...], str]] = []
    for name in candidate_names:
        known = known_states[name]
        known_acceptance = float(np.mean(known == "known_ecc"))
        known_uncertain = float(np.mean(known == "uncertain_ecc"))
        family_metrics = [
            _proxy_family_metrics(candidate_states(proxy_scores_by_family[family], candidates[name])[0])
            for family in development_families
        ]
        min_unknown = min(value["unknown_recall"] for value in family_metrics)
        min_coverage = min(value["unknown_or_uncertain_coverage"] for value in family_metrics)
        max_direct_known = max(value["direct_known_error"] for value in family_metrics)
        if mode == "balanced":
            eligible = known_acceptance >= 0.93 and known_uncertain <= 0.07
            key = (float(eligible), min_unknown, min_coverage, -max_direct_known, known_acceptance)
        elif mode == "safety":
            eligible = float(np.mean(known != "unknown_ecc")) >= 0.93
            key = (float(eligible), min_coverage, -max_direct_known, min_unknown, -known_uncertain)
        else:
            raise ValueError(mode)
        ranked.append((key, name))
    return max(ranked, key=lambda item: (item[0], item[1]))[1]


def _score_archives(
    archives: Sequence[Any],
    categories: Sequence[str],
    identifiers: Sequence[str],
    detector_system: DetectorSystem,
    author_task: Any,
    presence: TorchPresenceDetector,
    label: str,
) -> dict[str, Any]:
    detector_scores = {detector: [] for detector in DETECTORS}
    presence_scores: list[float] = []
    closed_indices: list[int] = []
    for index, archive in enumerate(archives):
        logits, embeddings = extract_archive_embeddings(author_task, archive)
        scores = detector_system.score(logits, embeddings)
        for detector in DETECTORS:
            detector_scores[detector].append(scores[detector])
        presence_scores.append(float(presence.predict_probabilities(archive).mean(axis=1).mean(axis=0)))
        closed_indices.append(int(_closed_indices(logits[None])[0]))
        if (index + 1) % 10 == 0 or index + 1 == len(archives):
            print(f"stage3 scoring {label}: {index + 1}/{len(archives)}", flush=True)
    return {
        "categories": np.asarray(categories, dtype=str),
        "archive_ids": np.asarray(identifiers, dtype=str),
        "scores": {key: np.asarray(value, dtype=np.float64) for key, value in detector_scores.items()},
        "ecc_scores": np.asarray(presence_scores, dtype=np.float64),
        "closed_index": np.asarray(closed_indices, dtype=np.int64),
    }


def _seven_confusion(categories: np.ndarray, outputs: np.ndarray) -> list[list[int]]:
    truth = np.asarray([
        category if category in KNOWN_TYPES else ("no_ecc" if category in NO_ECC_TYPES else "unknown_ecc")
        for category in categories
    ])
    index = {label: position for position, label in enumerate(SEVEN_LABELS)}
    matrix = np.zeros((7, 7), dtype=np.int64)
    for expected, observed in zip(truth, outputs):
        matrix[index[str(expected)], index[str(observed)]] += 1
    return matrix.tolist()


def evaluate_confirmation(
    data: dict[str, Any],
    states: np.ndarray,
    rejected: np.ndarray,
    tau1: float,
    closed_macro_f1: float,
    risk_score: np.ndarray | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    categories = data["categories"]
    no_ecc = np.isin(categories, NO_ECC_TYPES)
    known = np.isin(categories, KNOWN_TYPES)
    unknown = np.isin(categories, ("HEDGES", "DNA-Aeon"))
    final_states = states.astype(object)
    final_states[data["ecc_scores"] < tau1] = "no_ecc"
    closed_labels = np.asarray(KNOWN_TYPES, dtype=object)[data["closed_index"]]
    outputs = final_states.copy()
    outputs[final_states == "known_ecc"] = closed_labels[final_states == "known_ecc"]
    truth_unknown = unknown[np.logical_or(known, unknown)]
    stage2_states = final_states[np.logical_or(known, unknown)]
    risk = risk_coverage_metrics(stage2_states, truth_unknown)
    macro_f1 = _macro_f1(categories[known], outputs[known])
    vote_score = (
        rejected.mean(axis=1)
        if risk_score is None
        else np.asarray(risk_score, dtype=np.float64)
    )
    binary = np.logical_or(known, unknown)
    metrics = {
        "known_acceptance_rate": float(np.mean(final_states[known] == "known_ecc")),
        "known_rejection_rate": float(np.mean(final_states[known] == "unknown_ecc")),
        "HEDGES_unknown_recall": float(np.mean(final_states[categories == "HEDGES"] == "unknown_ecc")),
        "DNA_Aeon_unknown_recall": float(np.mean(final_states[categories == "DNA-Aeon"] == "unknown_ecc")),
        "combined_unknown_recall": float(np.mean(final_states[unknown] == "unknown_ecc")),
        "unknown_output_as_known_rate": float(np.mean(final_states[unknown] == "known_ecc")),
        "unknown_misclassified_as_BCH_rate": float(np.mean(outputs[unknown] == "BCH")),
        "uncertain_ecc_rate": float(np.mean(final_states == "uncertain_ecc")),
        "known_uncertain_rate": float(np.mean(final_states[known] == "uncertain_ecc")),
        "unknown_uncertain_rate": float(np.mean(final_states[unknown] == "uncertain_ecc")),
        "unknown_or_uncertain_coverage": float(np.mean(final_states[unknown] != "known_ecc")),
        **risk,
        "known_type_macro_f1": macro_f1,
        "known_type_macro_f1_change_from_closed": macro_f1 - closed_macro_f1,
        "AUROC": _auroc(vote_score[unknown], vote_score[known]),
        "AUPR": _average_precision(unknown[binary], vote_score[binary]),
        "FPR_at_95_TPR": _fpr_at_95_tpr(vote_score[unknown], vote_score[known]),
        "uncertain_as_independent_output": True,
        "uncertain_counted_incorrect_accuracy": float(np.mean([
            (output == category) if category in KNOWN_TYPES else
            (output == "no_ecc" if category in NO_ECC_TYPES else output == "unknown_ecc")
            for category, output in zip(categories, outputs)
        ])),
        "labels": list(SEVEN_LABELS),
        "seven_class_confusion_matrix": _seven_confusion(categories, outputs),
        "no_ecc_specificity": float(np.mean(final_states[no_ecc] == "no_ecc")),
    }
    return metrics, outputs.astype(str)


def _write_simple_svg(path: Path, title: str, rows: Sequence[tuple[str, float, float]]) -> None:
    width, height = 900, 90 + 42 * len(rows)
    svg = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}">', '<rect width="100%" height="100%" fill="white"/>', f'<text x="450" y="25" text-anchor="middle" font-family="sans-serif" font-size="17">{title}</text>']
    for index, (name, first, second) in enumerate(rows):
        y = 55 + index * 42
        svg.append(f'<text x="10" y="{y+12}" font-family="sans-serif" font-size="11">{name}</text>')
        svg.append(f'<rect x="300" y="{y}" width="{500*first:.1f}" height="12" fill="#1f77b4"/>')
        svg.append(f'<rect x="300" y="{y+15}" width="{500*second:.1f}" height="12" fill="#d62728"/>')
    svg.append('</svg>')
    path.write_text("\n".join(svg), encoding="utf-8")


def run_stage3(
    source: str | Path,
    stage2_root: str | Path,
    output: str | Path,
    seed: int = 42,
    device: str | None = None,
    command: Sequence[str] | None = None,
) -> dict[str, Any]:
    source = Path(source).resolve()
    stage2_root = Path(stage2_root).resolve()
    output = Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True)
    reference_root = output / "proxy_anomaly_references"
    experiment = ExperimentConfig()
    run = SimulationRunConfig(seed=seed, molecules=20, reads_per_molecule=50, test_error_rate=0.05)
    device_value = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models_path = Path("vendor/zhouph0313_DNA/models.py").resolve()
    weight_path = Path(DEFAULT_WEIGHT_ROOT).resolve() / "type" / "transformer_model_f10.6033.pt"
    models_hash_before, weight_hash_before = _sha256(models_path), _sha256(weight_path)
    detector_system = DetectorSystem.from_embedding_npz(
        stage2_root / "embedding_detector" / "embedding_features.npz"
    )
    thresholds = {
        str(target): {
            detector: acceptance_threshold(detector_system.calibration_scores[detector], target)
            for detector in DETECTORS
        }
        for target in TARGETS
    }
    registry = build_candidate_registry(thresholds)
    (output / "detector_registry.json").write_text(json.dumps({
        "detectors": {
            "global_energy": "original per-read energy mean_q then mean_M",
            "pca16_archive_diagonal_minimum": "PCA16 archive mean/std standardized diagonal minimum-class distance",
            "raw128_archive_conformal_maximum_pvalue": "raw128 archive mean/std class-conditional conformal -max(p)",
            "logits_diagonal_minimum": "43D logit-statistics diagonal minimum-class distance",
        },
        "threshold_targets": list(TARGETS), "candidates": registry,
        "target_inner_codes_used": False,
    }, indent=2, ensure_ascii=False), encoding="utf-8")

    recognizer = build_primary_type_recognizer(device=device_value, batch_size=64)
    for parameter in recognizer.code_type.model.parameters():
        parameter.requires_grad_(False)
    presence_checkpoint = torch.load(source / "models" / "external_presence_cnn.pt", map_location="cpu", weights_only=True)
    presence_model = ExternalPresenceCNN()
    presence_model.load_state_dict(presence_checkpoint["state_dict"], strict=True)
    presence = TorchPresenceDetector(presence_model, device=device_value, batch_size=64)

    # Development phase: no target inner-code reference exists before frozen_rules.json is written.
    proxy_metadata: list[dict[str, Any]] = []
    proxy_reference_sets: dict[str, dict[str, np.ndarray]] = {}
    proxy_data: dict[str, dict[str, Any]] = {}
    for split in ("fit", "calibration", "validation", "final-test"):
        proxy_reference_sets[split] = {}
        for family in PROXY_FAMILIES:
            references, metadata = generate_proxy_references(
                family, f"proxy-{split}", 10, 20, seed,
                reference_root / split / f"{family.lower()}.fasta",
            )
            proxy_reference_sets[split][family] = references
            proxy_metadata.extend(metadata)
            archives = archives_from_references(
                experiment, family, f"proxy-{split}-seed-{seed}", references, 10, 20, 50, 0.05
            )
            proxy_data[f"{split}:{family}"] = _score_archives(
                archives, [family] * 10,
                [f"proxy-{split}:{family}:{i}" for i in range(10)],
                detector_system, recognizer.code_type, presence, f"proxy-{split}-{family}",
            )
    known_validation_archives, known_validation_categories, known_validation_ids = _make_archives(
        experiment, "stage3-known-validation", KNOWN_CODE_TYPES, 20, run
    )
    known_validation_data = _score_archives(
        known_validation_archives, known_validation_categories, known_validation_ids,
        detector_system, recognizer.code_type, presence, "known-validation",
    )
    existing_fit = _known_reference_sets(experiment, "embedding-fit", seed, 20, 20, KNOWN_CODE_TYPES)
    existing_cal = _known_reference_sets(experiment, "embedding-calibration", seed, 20, 20, KNOWN_CODE_TYPES)
    known_validation_refs = _known_reference_sets(experiment, "stage3-known-validation", seed, 20, 20, KNOWN_CODE_TYPES)
    split_fingerprints: dict[str, set[bytes]] = {
        "stage2-fit": _fingerprints(existing_fit),
        "stage2-calibration": _fingerprints(existing_cal),
        "stage3-known-validation": _fingerprints(known_validation_refs),
        **{f"proxy-{split}": _fingerprints(values) for split, values in proxy_reference_sets.items()},
    }
    split_names = list(split_fingerprints)
    overlap_matrix: dict[str, int] = {}
    for left_index, left in enumerate(split_names):
        for right in split_names[left_index + 1:]:
            overlap = len(split_fingerprints[left] & split_fingerprints[right])
            overlap_matrix[f"{left}|{right}"] = overlap
            if overlap:
                raise RuntimeError(f"development molecular overlap: {left}/{right}")
    proxy_audit = {
        "families": list(PROXY_FAMILIES), "target_inner_codes_absent": True,
        "splits": {
            split: audit_molecular_references(values) for split, values in proxy_reference_sets.items()
        },
        "molecule_metadata": proxy_metadata, "cross_split_overlaps": overlap_matrix,
        "reference_length": 384, "alphabet": "ACGT", "stable_seed_only": True,
    }
    (output / "proxy_anomaly_dataset_audit.json").write_text(
        json.dumps(proxy_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    known_states: dict[str, np.ndarray] = {}
    for name, candidate in registry.items():
        known_states[name] = candidate_states(known_validation_data["scores"], candidate)[0]
    proxy_development_scores = {
        family: {
            detector: np.concatenate([
                proxy_data[f"{split}:{family}"]["scores"][detector]
                for split in ("fit", "calibration")
            ])
            for detector in DETECTORS
        }
        for family in PROXY_FAMILIES
    }
    balanced_candidates = [
        name for name in registry
        if name == "t95_energy_only"
        or name.startswith("t95_energy_raw_")
        or name.startswith("t95_all4_")
    ]
    safety_candidates = [
        name for name, value in registry.items()
        if name.startswith("t93_") and value["logic"] == "consensus"
    ]
    lofo: dict[str, Any] = {}
    balanced_selected: list[str] = []
    safety_selected: list[str] = []
    for held in PROXY_FAMILIES:
        development = [family for family in PROXY_FAMILIES if family != held]
        selected_balanced = select_candidate(
            registry, balanced_candidates, known_states, proxy_development_scores, development, "balanced"
        )
        selected_safety = select_candidate(
            registry, safety_candidates, known_states, proxy_development_scores, development, "safety"
        )
        balanced_selected.append(selected_balanced)
        safety_selected.append(selected_safety)
        held_scores = proxy_data[f"validation:{held}"]["scores"]
        lofo[held] = {
            "development_families": development,
            "balanced_selected_rule": selected_balanced,
            "balanced_held_family_metrics": _proxy_family_metrics(candidate_states(held_scores, registry[selected_balanced])[0]),
            "safety_selected_rule": selected_safety,
            "safety_held_family_metrics": _proxy_family_metrics(candidate_states(held_scores, registry[selected_safety])[0]),
        }
    def majority_choice(values: Sequence[str]) -> str:
        counts = Counter(values)
        return sorted(counts, key=lambda value: (-counts[value], value))[0]
    frozen_modes = {
        "conservative": "t98_energy_only",
        "balanced": majority_choice(balanced_selected),
        "safety": majority_choice(safety_selected),
        "three_state_rule1": "t95_energy_raw_consensus",
    }
    (output / "leave_one_proxy_family_out_results.json").write_text(
        json.dumps({"folds": lofo, "target_inner_codes_used": False}, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    frozen_config = {
        "frozen_before_confirmation": True,
        "modes": frozen_modes,
        "candidate_registry": registry,
        "selection_source": "known ECC calibration/validation plus leave-one-proxy-family-out only",
        "HEDGES_DNA_Aeon_used": False,
        "conservative_raw128_risk_flag_only": True,
    }
    (output / "multidetector_calibration.json").write_text(
        json.dumps({"thresholds": thresholds, "known_calibration_counts": 80, **frozen_config}, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    # Confirmation phase starts only after all rules and thresholds are frozen above.
    confirmation_seed = stable_seed("stage3-independent-confirmation", seed)
    confirmation_run = SimulationRunConfig(
        seed=confirmation_seed, molecules=20, reads_per_molecule=50,
        test_archives_per_category=50, test_error_rate=0.05,
    )
    base_categories = KNOWN_CODE_TYPES + NO_ECC_TYPES
    confirmation_archives, confirmation_categories, confirmation_ids = _make_archives(
        experiment, "stage3-confirmation", base_categories, 50, confirmation_run
    )
    confirmation_reference_sets = _known_reference_sets(
        experiment, "stage3-confirmation", confirmation_seed, 50, 20, base_categories
    )
    confirmation_inner_validation: list[dict[str, Any]] = []
    for category in ("HEDGES", "DNA-Aeon"):
        references, validation = generate_inner_code_references(
            category, 1000, confirmation_seed,
            output / "confirmation_references" / f"{category.lower().replace('-', '_')}.fasta",
            namespace="stage3-independent-confirmation",
        )
        confirmation_reference_sets[category] = references
        confirmation_inner_validation.append(validation.__dict__)
        confirmation_archives.extend(archives_from_references(
            experiment, category, f"stage3-confirmation-seed-{confirmation_seed}",
            references, 50, 20, 50, 0.05,
        ))
        confirmation_categories.extend([category] * 50)
        confirmation_ids.extend([f"stage3-confirmation:{category}:{i}" for i in range(50)])
    confirmation_fingerprints = _fingerprints(confirmation_reference_sets)
    confirmation_overlaps = {
        split: len(confirmation_fingerprints & values) for split, values in split_fingerprints.items()
    }
    if any(confirmation_overlaps.values()):
        raise RuntimeError("confirmation molecules overlap development data")
    confirmation_data = _score_archives(
        confirmation_archives, confirmation_categories, confirmation_ids,
        detector_system, recognizer.code_type, presence, "independent-confirmation",
    )
    closed_labels = np.asarray(KNOWN_TYPES, dtype=object)[confirmation_data["closed_index"]]
    known_mask = np.isin(confirmation_data["categories"], KNOWN_TYPES)
    closed_macro_f1 = _macro_f1(confirmation_data["categories"][known_mask], closed_labels[known_mask])
    tau1 = IncrementalThresholds.load(source / "thresholds.json").ecc_presence

    all_rules_metrics: dict[str, Any] = {}
    all_rules_outputs: dict[str, np.ndarray] = {}
    all_rules_states: dict[str, np.ndarray] = {}
    all_rules_rejected: dict[str, np.ndarray] = {}
    for name, candidate in registry.items():
        states, rejected = candidate_states(confirmation_data["scores"], candidate)
        metrics, outputs = evaluate_confirmation(
            confirmation_data,
            states,
            rejected,
            tau1,
            closed_macro_f1,
            risk_score=(
                confirmation_data["scores"][candidate["detectors"][0]]
                if len(candidate["detectors"]) == 1
                else None
            ),
        )
        all_rules_metrics[name] = metrics
        all_rules_outputs[name] = outputs
        all_rules_states[name] = states
        all_rules_rejected[name] = rejected
    mode_metrics = {
        mode: all_rules_metrics[rule] for mode, rule in frozen_modes.items()
    }
    results = {
        "experiment_positioning": "冻结作者盲识别核心、通过外部代理异常暴露和多检测器协同实现风险分层的增量功能验证",
        "frozen_modes": frozen_modes,
        "closed_set_known_type_macro_f1": closed_macro_f1,
        "modes": mode_metrics,
        "all_preregistered_rules": all_rules_metrics,
        "confirmation_is_not_first_family_blind_test": True,
    }
    (output / "multidetector_metrics.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    matrices = {
        name: {"labels": list(SEVEN_LABELS), "matrix": value["seven_class_confusion_matrix"]}
        for name, value in all_rules_metrics.items()
    }
    (output / "seven_class_confusion_matrices.json").write_text(
        json.dumps(matrices, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output / "multidetector_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("archive_id", "category", "mode", "rule", "state", "cascade_output", "ecc_score", "risk_flag", "code_rate", "code_length", *DETECTORS)
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode, rule in frozen_modes.items():
            states = all_rules_states[rule]
            outputs = all_rules_outputs[rule]
            raw_risk = confirmation_data["scores"]["raw128_archive_conformal_maximum_pvalue"] > thresholds["0.98"]["raw128_archive_conformal_maximum_pvalue"]
            for index, archive_id in enumerate(confirmation_data["archive_ids"]):
                reported_state = (
                    "no_ecc" if confirmation_data["ecc_scores"][index] < tau1 else states[index]
                )
                writer.writerow({
                    "archive_id": archive_id, "category": confirmation_data["categories"][index],
                    "mode": mode, "rule": rule, "state": reported_state, "cascade_output": outputs[index],
                    "ecc_score": confirmation_data["ecc_scores"][index],
                    "risk_flag": bool(raw_risk[index]) if mode == "conservative" else "",
                    "code_rate": "null", "code_length": "null",
                    **{detector: confirmation_data["scores"][detector][index] for detector in DETECTORS},
                })
    with (output / "risk_mode_comparison.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("mode", "rule", "known_acceptance_rate", "known_rejection_rate", "HEDGES_unknown_recall", "DNA_Aeon_unknown_recall", "combined_unknown_recall", "unknown_output_as_known_rate", "unknown_misclassified_as_BCH_rate", "known_uncertain_rate", "unknown_uncertain_rate", "unknown_or_uncertain_coverage", "decisive_coverage", "decisive_binary_accuracy", "known_type_macro_f1", "known_type_macro_f1_change_from_closed")
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for mode, rule in frozen_modes.items():
            writer.writerow({"mode": mode, "rule": rule, **{field: mode_metrics[mode][field] for field in fields if field not in {"mode", "rule"}}})

    decisions = np.column_stack([
        confirmation_data["scores"][detector] > thresholds["0.95"][detector]
        for detector in DETECTORS
    ])
    with (output / "score_agreement_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("detector", *DETECTORS))
        for left_index, left in enumerate(DETECTORS):
            writer.writerow((left, *[
                float(np.mean(decisions[:, left_index] == decisions[:, right_index]))
                for right_index in range(len(DETECTORS))
            ]))
    disagreement = {
        "target": 0.95,
        "all_detector_unanimous_rate": float(np.mean(np.logical_or(decisions.all(1), ~decisions.any(1)))),
        "by_category": {
            category: {
                "count": int(np.sum(confirmation_data["categories"] == category)),
                "disagreement_rate": float(np.mean(
                    np.logical_and(decisions[confirmation_data["categories"] == category].any(1),
                                   ~decisions[confirmation_data["categories"] == category].all(1))
                )),
            }
            for category in sorted(set(confirmation_data["categories"]))
        },
    }
    (output / "detector_disagreement_analysis.json").write_text(
        json.dumps(disagreement, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    _write_simple_svg(output / "multidetector_score_distributions.svg", "Detector rejection rates: known vs target inner codes", [
        (detector,
         float(np.mean(decisions[known_mask, index])),
         float(np.mean(decisions[np.isin(confirmation_data["categories"], ("HEDGES", "DNA-Aeon")), index])))
        for index, detector in enumerate(DETECTORS)
    ])
    _write_simple_svg(output / "risk_coverage_curves.svg", "Risk modes: known acceptance vs unknown risk coverage", [
        (mode, value["known_acceptance_rate"], value["unknown_or_uncertain_coverage"])
        for mode, value in mode_metrics.items()
    ])

    models_hash_after, weight_hash_after = _sha256(models_path), _sha256(weight_path)
    final_audit = {
        "confirmation_seed": int(confirmation_seed),
        "statement": "在既往测试已暴露编码家族信息后，使用独立分子和独立随机种子进行冻结规则的确认性评估。",
        "rules_frozen_before_confirmation_generation": True,
        "confirmation_development_overlap": confirmation_overlaps,
        "reference_audit": audit_molecular_references(confirmation_reference_sets),
        "inner_code_validation": confirmation_inner_validation,
        "HEDGES_DNA_Aeon_used_for_fit_feature_threshold_hyperparameter_or_rule_selection": False,
    }
    (output / "final_confirmation_audit.json").write_text(
        json.dumps(final_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    input_files = [
        source / "calibration_shared_logits.npz", source / "test_shared_logits.npz",
        source / "thresholds.json", stage2_root / "stage2_metrics.json",
        stage2_root / "embedding_detector" / "embedding_features.npz",
        stage2_root / "embedding_detector" / "embedding_metrics.json",
        stage2_root / "embedding_detector" / "embedding_extraction_audit.json",
    ]
    manifest = {
        "experiment_positioning": results["experiment_positioning"],
        "command": list(command or []), "seed": seed, "confirmation_seed": int(confirmation_seed),
        "environment": {
            "python": sys.version, "platform": platform.platform(), "numpy": np.__version__,
            "torch": torch.__version__, "device": str(device_value),
            "gpu": torch.cuda.get_device_name(0) if device_value.type == "cuda" else None,
        },
        "input_sha256": {str(path): _sha256(path) for path in input_files},
        "models_py_sha256_before": models_hash_before, "models_py_sha256_after": models_hash_after,
        "models_py_unchanged": models_hash_before == models_hash_after,
        "author_weight_sha256_before": weight_hash_before, "author_weight_sha256_after": weight_hash_after,
        "author_weight_expected_sha256": EXPECTED_SHA256["type/transformer_model_f10.6033.pt"],
        "author_weight_unchanged": weight_hash_before == weight_hash_after,
        "author_transformer_frozen": all(not parameter.requires_grad for parameter in recognizer.code_type.model.parameters()),
        "all_data_splits": list(split_fingerprints) + ["stage3-independent-confirmation"],
        "thresholds": thresholds, "combination_rules": registry, "frozen_modes": frozen_modes,
        "proxy_anomaly_families": list(PROXY_FAMILIES),
        "HEDGES_DNA_Aeon_absent_from_development": True,
        "stable_seed_only": True, "python_hash_used": False,
        "code_rate": None, "code_length": None,
    }
    (output / "experiment_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return results


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-3 proxy exposure and multidetector risk modes")
    parser.add_argument("--source", default="outputs/inner_codes_formal_seed42")
    parser.add_argument("--stage2-root", default="outputs/stage2_feature_rejection_seed42")
    parser.add_argument("--output", default="outputs/stage3_multidetector_proxy_exposure_seed42")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default=None)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    command = [sys.executable, "-m", "incremental_validation.stage3_multidetector", *(argv or sys.argv[1:])]
    result = run_stage3(args.source, args.stage2_root, args.output, args.seed, args.device, command)
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
