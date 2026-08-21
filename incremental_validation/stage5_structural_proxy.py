from __future__ import annotations

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
from hierarchical_ecc.coding import stable_seed
from hierarchical_ecc.config import ExperimentConfig, KNOWN_CODE_TYPES, NO_ECC_TYPES
from incremental_validation.collector import TorchPresenceDetector
from incremental_validation.comparison import IncrementalThresholds, KNOWN_TYPES
from incremental_validation.embedding_rejection import extract_archive_embeddings
from incremental_validation.inner_codes import archives_from_references, generate_inner_code_references
from incremental_validation.simulation import ExternalPresenceCNN, SimulationRunConfig, _make_archives
from incremental_validation.stage2_feature_rejection import (
    _average_precision, _auroc, _fpr_at_95_tpr, _macro_f1, _sha256,
    acceptance_threshold, extract_logit_features,
)
from incremental_validation.stage3_multidetector import (
    PROXY_FAMILIES, SEVEN_LABELS, generate_proxy_references,
)


TARGETS = (0.98, 0.95, 0.93)
PCA_DIMENSIONS = (16, 32, 48)
RIDGE_VALUES = (0.01, 0.1, 1.0, 10.0)
LAGS = (1, 2, 3, 4, 6, 8, 12)
CONFIRMATION_CATEGORIES = (*KNOWN_CODE_TYPES, *NO_ECC_TYPES, "HEDGES", "DNA-Aeon")


def _entropy(probabilities: np.ndarray) -> np.ndarray:
    values = np.asarray(probabilities, dtype=np.float64)
    return -(np.where(values > 0, values * np.log2(np.maximum(values, 1e-12)), 0.0)).sum(axis=-1)


def sequence_structure_features(archive: Any) -> np.ndarray:
    """Archive statistics from observed reads only; padding never contributes."""

    one_hot = np.asarray(archive.one_hot, dtype=np.float64)
    mask = np.asarray(archive.mask, dtype=np.bool_)
    if one_hot.ndim != 4 or one_hot.shape[2] != 4 or mask.shape != (one_hot.shape[0], one_hot.shape[1], one_hot.shape[3]):
        raise ValueError("archive must obey [M,q,4,L] plus [M,q,L] mask")
    reads = one_hot.reshape(-1, 4, one_hot.shape[-1])
    valid = mask.reshape(-1, mask.shape[-1])
    lengths = valid.sum(axis=1).astype(np.float64)
    if np.any(lengths <= 0):
        raise ValueError("empty reads are not supported")
    base_counts = (reads * valid[:, None, :]).sum(axis=-1)
    base_freq = base_counts / lengths[:, None]
    pair_valid = valid[:, :-1] & valid[:, 1:]
    pair_denominator = np.maximum(pair_valid.sum(axis=1), 1).astype(np.float64)
    pair_counts = np.einsum("ril,rjl,rl->rij", reads[:, :, :-1], reads[:, :, 1:], pair_valid)
    pair_freq = pair_counts.reshape(reads.shape[0], 16) / pair_denominator[:, None]
    bases = reads.argmax(axis=1).astype(np.int16)
    maximum_run = np.empty(reads.shape[0], dtype=np.float64)
    long_run_fraction = np.empty(reads.shape[0], dtype=np.float64)
    for index, length_value in enumerate(lengths.astype(int)):
        values = bases[index, :length_value]
        boundaries = np.flatnonzero(np.r_[True, values[1:] != values[:-1], True])
        runs = np.diff(boundaries)
        maximum_run[index] = runs.max() / length_value
        long_run_fraction[index] = runs[runs >= 3].sum() / length_value
    lag_features = []
    for lag in LAGS:
        comparison_valid = valid[:, :-lag] & valid[:, lag:]
        denominator = np.maximum(comparison_valid.sum(axis=1), 1)
        lag_features.append((((bases[:, :-lag] == bases[:, lag:]) & comparison_valid).sum(axis=1) / denominator)[:, None])
    read_features = np.column_stack((
        lengths / one_hot.shape[-1], base_freq, (base_freq[:, 1] + base_freq[:, 2])[:, None],
        _entropy(base_freq)[:, None], pair_freq, _entropy(pair_freq)[:, None],
        maximum_run[:, None], long_run_fraction[:, None], *lag_features,
    ))
    statistics = (
        read_features.mean(axis=0), read_features.std(axis=0),
        np.quantile(read_features, 0.10, axis=0), np.median(read_features, axis=0),
        np.quantile(read_features, 0.90, axis=0),
    )
    return np.concatenate(statistics).astype(np.float64)


def archive_feature_blocks(archive: Any, logits: np.ndarray, embeddings: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "sequence": sequence_structure_features(archive),
        "embedding": np.concatenate((embeddings.mean(axis=(0, 1)), embeddings.std(axis=(0, 1)))).astype(np.float64),
        "logits": extract_logit_features(np.asarray(logits)[None]).features[0].astype(np.float64),
    }


def combined_archive_features(archive: Any, logits: np.ndarray, embeddings: np.ndarray) -> np.ndarray:
    blocks = archive_feature_blocks(archive, logits, embeddings)
    return np.concatenate((blocks["sequence"], blocks["embedding"], blocks["logits"])).astype(np.float64)


@dataclass
class ProxyClassifier:
    mean: np.ndarray
    scale: np.ndarray
    pca_mean: np.ndarray
    components: np.ndarray
    coefficients: np.ndarray
    ridge: float

    @classmethod
    def fit(cls, features: np.ndarray, labels: np.ndarray, dimensions: int, ridge: float) -> "ProxyClassifier":
        x = np.asarray(features, dtype=np.float64)
        y = np.asarray(labels, dtype=np.float64)
        mean = x.mean(axis=0)
        scale = x.std(axis=0)
        scale[scale < 1e-8] = 1.0
        standardized = (x - mean) / scale
        pca_mean = standardized.mean(axis=0)
        _, _, vt = np.linalg.svd(standardized - pca_mean, full_matrices=False)
        dimensions = min(int(dimensions), vt.shape[0])
        components = vt[:dimensions]
        reduced = (standardized - pca_mean) @ components.T
        design = np.column_stack((reduced, np.ones(reduced.shape[0])))
        weights = np.where(y > 0.5, 0.5 / max(np.sum(y > 0.5), 1), 0.5 / max(np.sum(y <= 0.5), 1))
        coefficients = np.zeros(design.shape[1], dtype=np.float64)
        penalty = np.eye(design.shape[1]) * float(ridge)
        penalty[-1, -1] = 0.0
        for _ in range(80):
            linear = np.clip(design @ coefficients, -30.0, 30.0)
            probability = 1.0 / (1.0 + np.exp(-linear))
            gradient = design.T @ (weights * (probability - y)) + penalty @ coefficients
            curvature = weights * probability * (1.0 - probability)
            hessian = design.T @ (curvature[:, None] * design) + penalty + np.eye(design.shape[1]) * 1e-9
            update = np.linalg.solve(hessian, gradient)
            coefficients -= update
            if np.linalg.norm(update) < 1e-8:
                break
        return cls(mean, scale, pca_mean, components, coefficients, float(ridge))

    def score(self, features: np.ndarray) -> np.ndarray:
        standardized = (np.asarray(features, dtype=np.float64) - self.mean) / self.scale
        reduced = (standardized - self.pca_mean) @ self.components.T
        linear = np.column_stack((reduced, np.ones(reduced.shape[0]))) @ self.coefficients
        return 1.0 / (1.0 + np.exp(-np.clip(linear, -30.0, 30.0)))

    def save(self, path: Path) -> None:
        np.savez_compressed(path, mean=self.mean, scale=self.scale, pca_mean=self.pca_mean,
                            components=self.components, coefficients=self.coefficients,
                            ridge=np.asarray(self.ridge))

    @classmethod
    def load(cls, path: str | Path) -> "ProxyClassifier":
        with np.load(Path(path), allow_pickle=False) as payload:
            return cls(payload["mean"], payload["scale"], payload["pca_mean"],
                       payload["components"], payload["coefficients"], float(payload["ridge"]))


def three_state_consensus(energy_rejected: np.ndarray, proxy_rejected: np.ndarray) -> np.ndarray:
    energy = np.asarray(energy_rejected, dtype=np.bool_)
    proxy = np.asarray(proxy_rejected, dtype=np.bool_)
    if energy.shape != proxy.shape:
        raise ValueError("detector decisions must align")
    states = np.full(energy.shape, "uncertain_ecc", dtype=object)
    states[~energy & ~proxy] = "known_ecc"
    states[energy & proxy] = "unknown_ecc"
    return states.astype(str)


def _extract_many(task: Any, archives: Sequence[Any], categories: Sequence[str], identifiers: Sequence[str], label: str) -> dict[str, np.ndarray]:
    features, energies, closed = [], [], []
    for index, archive in enumerate(archives):
        logits, embeddings = extract_archive_embeddings(task, archive)
        features.append(combined_archive_features(archive, logits, embeddings))
        bundle = extract_logit_features(logits[None])
        energies.append(float(bundle.energy[0]))
        probabilities = np.exp(logits - logits.max(axis=-1, keepdims=True))
        probabilities /= probabilities.sum(axis=-1, keepdims=True)
        closed.append(int(probabilities.mean(axis=1).mean(axis=0).argmax()))
        if (index + 1) % 10 == 0 or index + 1 == len(archives):
            print(f"stage5 extract {label}: {index + 1}/{len(archives)}", flush=True)
    return {"features": np.stack(features), "energy": np.asarray(energies),
            "closed_index": np.asarray(closed), "categories": np.asarray(categories, dtype=str),
            "archive_ids": np.asarray(identifiers, dtype=str)}


def _presence_many(detector: TorchPresenceDetector, archives: Sequence[Any]) -> np.ndarray:
    return np.asarray([detector.predict_probabilities(archive).mean(axis=1).mean(axis=0) for archive in archives])


def _save_payload(path: Path, payload: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **payload)


def _load_or_extract(path: Path, task: Any, archives: Sequence[Any], categories: Sequence[str], ids: Sequence[str], label: str, resume: bool) -> dict[str, np.ndarray]:
    if resume and path.is_file():
        with np.load(path, allow_pickle=False) as payload:
            return {key: payload[key] for key in payload.files}
    result = _extract_many(task, archives, categories, ids, label)
    _save_payload(path, result)
    return result


def _proxy_metrics(states: np.ndarray) -> dict[str, float]:
    return {"unknown": float(np.mean(states == "unknown_ecc")),
            "uncertain": float(np.mean(states == "uncertain_ecc")),
            "risk_coverage": float(np.mean(states != "known_ecc")),
            "direct_known": float(np.mean(states == "known_ecc"))}


def evaluate(categories: np.ndarray, closed_index: np.ndarray, ecc: np.ndarray, energy: np.ndarray,
             proxy_score: np.ndarray, tau1: float, energy_tau: float, proxy_tau: float) -> tuple[dict[str, Any], np.ndarray]:
    states = three_state_consensus(energy > energy_tau, proxy_score > proxy_tau).astype(object)
    states[ecc < tau1] = "no_ecc"
    closed = np.asarray(KNOWN_TYPES, dtype=object)[closed_index]
    outputs = states.copy()
    outputs[states == "known_ecc"] = closed[states == "known_ecc"]
    known = np.isin(categories, KNOWN_TYPES); unknown = np.isin(categories, ("HEDGES", "DNA-Aeon")); no_ecc = np.isin(categories, NO_ECC_TYPES)
    closed_f1 = _macro_f1(categories[known], closed[known]); final_f1 = _macro_f1(categories[known], outputs[known])
    binary_score = np.maximum((energy - energy_tau), proxy_score - proxy_tau)
    truth_binary = np.concatenate((np.zeros(np.sum(known), dtype=bool), np.ones(np.sum(unknown), dtype=bool)))
    score_binary = np.concatenate((binary_score[known], binary_score[unknown]))
    labels = {name: index for index, name in enumerate(SEVEN_LABELS)}
    matrix = np.zeros((7, 7), dtype=np.int64)
    truth = [category if category in KNOWN_TYPES else ("no_ecc" if category in NO_ECC_TYPES else "unknown_ecc") for category in categories]
    for expected, observed in zip(truth, outputs): matrix[labels[str(expected)], labels[str(observed)]] += 1
    metrics = {
        "known_acceptance_rate": float(np.mean(states[known] == "known_ecc")),
        "known_uncertain_rate": float(np.mean(states[known] == "uncertain_ecc")),
        "known_unknown_rate": float(np.mean(states[known] == "unknown_ecc")),
        "no_ecc_specificity": float(np.mean(states[no_ecc] == "no_ecc")),
        "HEDGES_unknown_recall": float(np.mean(states[categories == "HEDGES"] == "unknown_ecc")),
        "DNA_Aeon_unknown_recall": float(np.mean(states[categories == "DNA-Aeon"] == "unknown_ecc")),
        "combined_unknown_recall": float(np.mean(states[unknown] == "unknown_ecc")),
        "unknown_uncertain_rate": float(np.mean(states[unknown] == "uncertain_ecc")),
        "unknown_risk_coverage": float(np.mean(states[unknown] != "known_ecc")),
        "unknown_direct_known_rate": float(np.mean(states[unknown] == "known_ecc")),
        "unknown_misclassified_as_BCH_rate": float(np.mean(outputs[unknown] == "BCH")),
        "decisive_coverage": float(np.mean(states[np.logical_or(known, unknown)] != "uncertain_ecc")),
        "closed_known_type_macro_f1": closed_f1, "known_type_macro_f1": final_f1,
        "known_type_macro_f1_change_from_closed": final_f1 - closed_f1,
        "AUROC": _auroc(score_binary[truth_binary], score_binary[~truth_binary]),
        "AUPR": _average_precision(truth_binary, score_binary),
        "FPR_at_95_TPR": _fpr_at_95_tpr(score_binary[truth_binary], score_binary[~truth_binary]),
        "labels": list(SEVEN_LABELS), "seven_class_confusion_matrix": matrix.tolist(),
    }
    return metrics, outputs.astype(str)


def run_stage5(source: str | Path, stage3: str | Path, output: str | Path, seed: int = 42,
               device: str | None = None, resume: bool = False, command: Sequence[str] | None = None) -> dict[str, Any]:
    source, stage3, output = Path(source).resolve(), Path(stage3).resolve(), Path(output).resolve()
    if output.exists() and any(output.iterdir()) and not resume:
        raise FileExistsError(f"refusing to overwrite non-empty output: {output}")
    output.mkdir(parents=True, exist_ok=True); cache = output / "feature_cache"; cache.mkdir(exist_ok=True)
    experiment = ExperimentConfig(); device_value = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    models_path = Path("vendor/zhouph0313_DNA/models.py").resolve(); weight_path = Path(DEFAULT_WEIGHT_ROOT).resolve() / "type" / "transformer_model_f10.6033.pt"
    models_before, weight_before = _sha256(models_path), _sha256(weight_path)
    recognizer = build_primary_type_recognizer(device=device_value, batch_size=64)
    recognizer.code_type.model.eval()
    for parameter in recognizer.code_type.model.parameters(): parameter.requires_grad_(False)
    presence_checkpoint = torch.load(source / "models" / "external_presence_cnn.pt", map_location="cpu", weights_only=True)
    presence_model = ExternalPresenceCNN(); presence_model.load_state_dict(presence_checkpoint["state_dict"], strict=True)
    presence = TorchPresenceDetector(presence_model, device=device_value, batch_size=64)
    run = SimulationRunConfig(seed=seed, molecules=20, reads_per_molecule=50, test_archives_per_category=50, test_error_rate=0.05)

    known_data: dict[str, dict[str, np.ndarray]] = {}
    for split in ("fit", "calibration", "validation"):
        archives, categories, ids = _make_archives(experiment, f"stage5-known-{split}", KNOWN_CODE_TYPES, 15, run)
        known_data[split] = _load_or_extract(cache / f"known_{split}.npz", recognizer.code_type, archives, categories, ids, f"known-{split}", resume)

    proxy_data: dict[str, dict[str, dict[str, np.ndarray]]] = {"fit": {}, "validation": {}}
    proxy_audit: dict[str, Any] = {"families": list(PROXY_FAMILIES), "target_inner_codes_absent": True, "splits": {}}
    for split in proxy_data:
        proxy_audit["splits"][split] = {}
        for family in PROXY_FAMILIES:
            refs, metadata = generate_proxy_references(family, f"stage5-proxy-{split}", 15, 20, seed,
                output / "proxy_references" / split / f"{family.lower()}.fasta")
            archives = archives_from_references(experiment, family, f"stage5-proxy-{split}-seed-{seed}", refs, 15, 20, 50, 0.05)
            ids = [f"stage5-proxy-{split}:{family}:{index}" for index in range(15)]
            proxy_data[split][family] = _load_or_extract(cache / f"proxy_{split}_{family}.npz", recognizer.code_type,
                archives, [family] * 15, ids, f"proxy-{split}-{family}", resume)
            proxy_audit["splits"][split][family] = {"molecules": int(refs.shape[0]), "unique": len({row.tobytes() for row in refs}),
                "length": int(refs.shape[1]), "metadata_sha256": sha256(json.dumps(metadata, sort_keys=True).encode()).hexdigest()}
    (output / "proxy_anomaly_audit.json").write_text(json.dumps(proxy_audit, indent=2, ensure_ascii=False), encoding="utf-8")

    candidates, lofo = [], {}
    for dimensions in PCA_DIMENSIONS:
        for ridge in RIDGE_VALUES:
            fold_values = []
            for held in PROXY_FAMILIES:
                development = [family for family in PROXY_FAMILIES if family != held]
                train_x = np.concatenate((known_data["fit"]["features"], *[proxy_data["fit"][family]["features"] for family in development]))
                train_y = np.concatenate((np.zeros(len(known_data["fit"]["features"])), *[np.ones(len(proxy_data["fit"][family]["features"])) for family in development]))
                classifier = ProxyClassifier.fit(train_x, train_y, dimensions, ridge)
                threshold = acceptance_threshold(classifier.score(known_data["calibration"]["features"]), 0.95)
                known_accept = float(np.mean(classifier.score(known_data["validation"]["features"]) <= threshold))
                held_recall = float(np.mean(classifier.score(proxy_data["validation"][held]["features"]) > threshold))
                fold_values.append({"held_family": held, "known_acceptance": known_accept, "held_proxy_recall": held_recall})
            summary = {"dimensions": dimensions, "ridge": ridge, "folds": fold_values,
                "minimum_known_acceptance": min(item["known_acceptance"] for item in fold_values),
                "minimum_held_proxy_recall": min(item["held_proxy_recall"] for item in fold_values),
                "mean_held_proxy_recall": float(np.mean([item["held_proxy_recall"] for item in fold_values]))}
            candidates.append(summary)
    eligible = [item for item in candidates if item["minimum_known_acceptance"] >= 0.93] or candidates
    selected = max(eligible, key=lambda item: (item["minimum_held_proxy_recall"], item["mean_held_proxy_recall"], item["minimum_known_acceptance"], -item["dimensions"], -item["ridge"]))
    lofo = {"candidates": candidates, "selection_rule": "max worst-family recall subject to every-fold known acceptance >=0.93",
            "selected": {key: selected[key] for key in ("dimensions", "ridge")}, "target_inner_codes_used": False}
    (output / "leave_one_proxy_family_out.json").write_text(json.dumps(lofo, indent=2, ensure_ascii=False), encoding="utf-8")

    final_train_x = np.concatenate((known_data["fit"]["features"], *[proxy_data["fit"][family]["features"] for family in PROXY_FAMILIES]))
    final_train_y = np.concatenate((np.zeros(len(known_data["fit"]["features"])), *[np.ones(len(proxy_data["fit"][family]["features"])) for family in PROXY_FAMILIES]))
    classifier = ProxyClassifier.fit(final_train_x, final_train_y, selected["dimensions"], selected["ridge"])
    classifier.save(output / "structural_embedding_proxy_detector.npz")
    known_cal_scores = classifier.score(known_data["calibration"]["features"])
    stage3_cal = json.loads((stage3 / "multidetector_calibration.json").read_text(encoding="utf-8"))
    calibration = {str(target): {"energy": float(stage3_cal["thresholds"][str(target)]["global_energy"]),
        "proxy": float(acceptance_threshold(known_cal_scores, target))} for target in TARGETS}
    frozen = {"frozen_before_confirmation": True, "feature_blocks": ["observed_read_sequence_structure", "embedding128_mean_std", "rich_logit_statistics"],
        "classifier": "standardized_PCA_L2_logistic", "selected": lofo["selected"], "thresholds": calibration,
        "combination": "energy_proxy_consensus_three_state", "HEDGES_DNA_Aeon_used": False}
    (output / "frozen_detector_config.json").write_text(json.dumps(frozen, indent=2, ensure_ascii=False), encoding="utf-8")

    confirmation_seed = stable_seed("stage5-independent-confirmation", seed)
    confirmation_run = SimulationRunConfig(seed=confirmation_seed, molecules=20, reads_per_molecule=50, test_archives_per_category=50, test_error_rate=0.05)
    archives, categories, ids = _make_archives(experiment, "stage5-confirmation", KNOWN_CODE_TYPES + NO_ECC_TYPES, 50, confirmation_run)
    for category in ("HEDGES", "DNA-Aeon"):
        refs, _ = generate_inner_code_references(category, 1000, confirmation_seed,
            output / "confirmation_references" / f"{category.lower().replace('-', '_')}.fasta", namespace="stage5-independent-confirmation")
        archives.extend(archives_from_references(experiment, category, f"stage5-confirmation-seed-{confirmation_seed}", refs, 50, 20, 50, 0.05))
        categories.extend([category] * 50); ids.extend([f"stage5-confirmation:{category}:{index}" for index in range(50)])
    confirmation = _load_or_extract(cache / "confirmation.npz", recognizer.code_type, archives, categories, ids, "confirmation", resume)
    if resume and (cache / "confirmation_presence.npy").is_file(): ecc = np.load(cache / "confirmation_presence.npy")
    else:
        ecc = _presence_many(presence, archives); np.save(cache / "confirmation_presence.npy", ecc)
    proxy_scores = classifier.score(confirmation["features"]); tau1 = IncrementalThresholds.load(source / "thresholds.json").ecc_presence
    metrics, detector_comparison, outputs_by_target = {}, {}, {}
    categories_array = confirmation["categories"].astype(str)
    for target in TARGETS:
        key = str(target); values, outputs = evaluate(categories_array, confirmation["closed_index"], ecc, confirmation["energy"], proxy_scores,
            tau1, calibration[key]["energy"], calibration[key]["proxy"])
        metrics[key] = values; outputs_by_target[key] = outputs
        energy_decision = (confirmation["energy"] > calibration[key]["energy"]).astype(np.float64)
        energy_values, _ = evaluate(
            categories_array, confirmation["closed_index"], ecc,
            confirmation["energy"], energy_decision, tau1,
            calibration[key]["energy"], 0.5,
        )
        proxy_values, _ = evaluate(
            categories_array, confirmation["closed_index"], ecc,
            proxy_scores, proxy_scores, tau1,
            calibration[key]["proxy"], calibration[key]["proxy"],
        )
        detector_comparison[key] = {
            "energy_only": energy_values,
            "structural_embedding_logits_proxy_only": proxy_values,
            "energy_proxy_three_state": values,
        }
    (output / "confirmation_metrics.json").write_text(json.dumps({"modes": metrics, "confirmation_not_first_family_blind_test": True}, indent=2, ensure_ascii=False), encoding="utf-8")
    (output / "detector_comparison.json").write_text(
        json.dumps(detector_comparison, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    with (output / "confirmation_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ("archive_id", "category", "target", "ecc_score", "energy", "proxy_score", "cascade_output", "code_rate", "code_length")
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        for target in TARGETS:
            key = str(target)
            for index, archive_id in enumerate(confirmation["archive_ids"].astype(str)):
                writer.writerow({"archive_id": archive_id, "category": categories_array[index], "target": target, "ecc_score": ecc[index],
                    "energy": confirmation["energy"][index], "proxy_score": proxy_scores[index], "cascade_output": outputs_by_target[key][index],
                    "code_rate": "null", "code_length": "null"})
    feature_definition = {"sequence_statistics": {"read_features": ["length", "base_frequencies_4", "GC", "base_entropy", "dinucleotide_frequencies_16", "dinucleotide_entropy", "normalized_max_homopolymer", "long_run_fraction", "same_base_lags_1_2_3_4_6_8_12"],
        "archive_aggregation": ["mean", "std", "q10", "median", "q90"]}, "embedding": "128D archive mean+std", "logits": "Stage-2 rich logit feature vector",
        "feature_dimension": int(confirmation["features"].shape[1])}
    (output / "feature_definitions.json").write_text(json.dumps(feature_definition, indent=2, ensure_ascii=False), encoding="utf-8")
    models_after, weight_after = _sha256(models_path), _sha256(weight_path)
    manifest = {"experiment_positioning": "冻结作者盲识别核心、通过序列结构特征与embedding/logits统计的外部代理异常检测器，并与能量检测器构成三态协同的增量功能验证。",
        "command": list(command or []), "seed": seed, "confirmation_seed": int(confirmation_seed), "proxy_families": list(PROXY_FAMILIES),
        "HEDGES_DNA_Aeon_absent_from_development": True, "rules_frozen_before_confirmation": True, "stable_seed_only": True,
        "environment": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__, "torch": torch.__version__, "device": str(device_value), "gpu": torch.cuda.get_device_name(0) if device_value.type == "cuda" else None},
        "models_py_sha256_before": models_before, "models_py_sha256_after": models_after, "models_py_unchanged": models_before == models_after,
        "author_weight_sha256_before": weight_before, "author_weight_sha256_after": weight_after, "author_weight_expected": EXPECTED_SHA256["type/transformer_model_f10.6033.pt"], "author_weight_unchanged": weight_before == weight_after,
        "author_transformer_frozen": all(not parameter.requires_grad for parameter in recognizer.code_type.model.parameters()), "thresholds": calibration,
        "code_rate": None, "code_length": None}
    (output / "experiment_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return {"selected": lofo["selected"], "metrics": metrics, "output": str(output)}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Stage-5 structural/representation proxy anomaly detector")
    parser.add_argument("--source", default="outputs/inner_codes_formal_seed42")
    parser.add_argument("--stage3", default="outputs/stage3_multidetector_proxy_exposure_seed42")
    parser.add_argument("--output", default="outputs/stage5_structural_embedding_proxy_seed42")
    parser.add_argument("--seed", type=int, default=42); parser.add_argument("--device", default=None); parser.add_argument("--resume", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv); command = [sys.executable, "-m", "incremental_validation.stage5_structural_proxy", *(argv or sys.argv[1:])]
    print(json.dumps(run_stage5(args.source, args.stage3, args.output, args.seed, args.device, args.resume, command), indent=2, ensure_ascii=False))


if __name__ == "__main__": main()
