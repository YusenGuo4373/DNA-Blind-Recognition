from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Sequence
import argparse
import csv
import hashlib
import json
import random
import shutil

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from author_baseline.recognizer import OneHotArchive
from author_baseline.weights import DEFAULT_WEIGHT_ROOT, build_primary_type_recognizer
from hierarchical_ecc.coding import stable_seed
from hierarchical_ecc.config import (
    ExperimentConfig,
    KNOWN_CODE_TYPES,
    NO_ECC_TYPES,
)
from hierarchical_ecc.data import ReferenceFactory, generate_archive_reads, sample_noisy_read

from .collector import SharedLogitDataset, TorchPresenceDetector, collect_shared_logits
from .inner_codes import (
    UNKNOWN_INNER_CODES,
    archives_from_references,
    generate_inner_code_references,
)
from .ecc_score_reporting import write_ecc_score_distributions
from .comparison import (
    IncrementalThresholds,
    calibrate_thresholds,
    compare_shared_logits,
    save_comparisons,
)


@dataclass(frozen=True)
class SimulationRunConfig:
    seed: int = 42
    formal: bool = False
    train_known_reads_per_category: int = 500
    train_inner_code_reads_per_category: int = 500
    train_no_ecc_reads_per_category: int = 500
    validation_known_reads_per_category: int = 120
    validation_inner_code_reads_per_category: int = 120
    validation_no_ecc_reads_per_category: int = 120
    presence_epochs: int = 5
    batch_size: int = 64
    validation_archives_per_category: int = 5
    test_archives_per_category: int = 5
    molecules: int = 5
    reads_per_molecule: int = 5
    test_error_rate: float = 0.05
    known_acceptance: float = 0.95


def bases_to_one_hot(bases: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    """Convert base ids to the author's [M,q,4,L] representation."""

    bases = np.asarray(bases)
    valid_mask = np.asarray(valid_mask, dtype=np.bool_)
    if bases.shape != valid_mask.shape:
        raise ValueError("bases and valid_mask must have the same [M,q,L] shape")
    safe = np.where(valid_mask, bases, 0).astype(np.int64)
    if np.any((safe < 0) | (safe > 3)):
        raise ValueError("valid bases must be integers in [0,3]")
    encoded = np.eye(4, dtype=np.float32)[safe]
    encoded *= valid_mask[..., None]
    return np.moveaxis(encoded, -1, -2)


def simulate_archive(
    config: ExperimentConfig,
    category: str,
    split: str,
    archive_id: int,
    error_rate: float,
    molecules: int,
    reads_per_molecule: int,
) -> OneHotArchive:
    generated = generate_archive_reads(
        config,
        category=category,
        split=split,
        archive_id=archive_id,
        error_rate=error_rate,
        molecules=molecules,
        reads_per_molecule=reads_per_molecule,
    )
    return OneHotArchive(
        one_hot=bases_to_one_hot(generated.bases, generated.valid_mask),
        mask=generated.valid_mask,
    )


class SimulatedPresenceReadDataset(Dataset):
    """Deterministic per-read ECC (including inner codes) versus No-ECC dataset."""

    def __init__(
        self,
        config: ExperimentConfig,
        split: str,
        seed: int,
        known_reads_per_category: int,
        inner_code_reads_per_category: int,
        no_ecc_reads_per_category: int,
        inner_code_references: dict[str, np.ndarray],
    ):
        if split not in {"train", "validation"}:
            raise ValueError("presence training data support train/validation only")
        if min(
            known_reads_per_category,
            inner_code_reads_per_category,
            no_ecc_reads_per_category,
        ) <= 0:
            raise ValueError("per-category read counts must be positive")
        self.config = config
        self.split = split
        self.seed = int(seed)
        self.categories = KNOWN_CODE_TYPES + UNKNOWN_INNER_CODES + NO_ECC_TYPES
        self.inner_code_references = {
            category: np.asarray(inner_code_references[category], dtype=np.uint8)
            for category in UNKNOWN_INNER_CODES
        }
        for category, references in self.inner_code_references.items():
            expected = (inner_code_reads_per_category, config.channel.reference_length)
            if references.shape != expected:
                raise ValueError(f"{category} references must have shape {expected}")
        self.segments: list[tuple[int, int, str]] = []
        cursor = 0
        for category in self.categories:
            if category in KNOWN_CODE_TYPES:
                count = known_reads_per_category
            elif category in UNKNOWN_INNER_CODES:
                count = inner_code_reads_per_category
            else:
                count = no_ecc_reads_per_category
            self.segments.append((cursor, cursor + int(count), category))
            cursor += int(count)
        self.total = cursor
        self.factory = ReferenceFactory(config)

    def __len__(self) -> int:
        return self.total

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        category = ""
        local_index = 0
        for start, stop, value in self.segments:
            if start <= index < stop:
                category = value
                local_index = int(index) - start
                break
        if not category:
            raise IndexError(index)
        namespace = f"presence-{self.split}-seed-{self.seed}"
        if category in UNKNOWN_INNER_CODES:
            reference = self.inner_code_references[category][local_index]
        else:
            reference = self.factory.make_reference(category, namespace, local_index, 0)
        rate_rng = np.random.default_rng(
            stable_seed("presence-rate", namespace, category, local_index)
        )
        rates = self.config.channel.train_error_rates
        error_rate = float(rates[int(rate_rng.integers(0, len(rates)))])
        read_rng = np.random.default_rng(
            stable_seed("presence-read", namespace, category, local_index)
        )
        read = sample_noisy_read(
            reference,
            error_rate,
            self.config.channel.min_read_length,
            self.config.channel.max_read_length,
            self.config.channel.padded_length,
            read_rng,
        )
        length = self.config.channel.padded_length
        mask = np.zeros(length, dtype=np.bool_)
        mask[: min(read.size, length)] = True
        bases = np.zeros(length, dtype=np.uint8)
        bases[mask] = read[:length]
        one_hot = np.eye(4, dtype=np.float32)[bases].T
        one_hot[:, ~mask] = 0.0
        target = float(category not in NO_ECC_TYPES)
        return (
            torch.from_numpy(one_hot),
            torch.from_numpy(mask),
            torch.tensor(target, dtype=torch.float32),
        )


class ExternalPresenceCNN(nn.Module):
    """Small external detector; it is never inserted into the author recognizer."""

    def __init__(self) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv1d(4, 32, kernel_size=7, padding=3),
            nn.GELU(),
            nn.Conv1d(32, 64, kernel_size=5, padding=2),
            nn.GELU(),
        )
        self.classifier = nn.Sequential(nn.Linear(128, 64), nn.GELU(), nn.Linear(64, 1))

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 4 or mask.shape != (x.shape[0], x.shape[2]):
            raise ValueError("presence input must be [B,4,L] with mask [B,L]")
        valid = mask.to(dtype=torch.bool)
        features = self.features(x)
        weights = valid.unsqueeze(1).to(dtype=features.dtype)
        mean = (features * weights).sum(dim=-1) / weights.sum(dim=-1).clamp_min(1.0)
        masked = features.masked_fill(~valid.unsqueeze(1), torch.finfo(features.dtype).min)
        maximum = masked.max(dim=-1).values
        return self.classifier(torch.cat((mean, maximum), dim=1))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _reference_fingerprints(references: np.ndarray) -> set[str]:
    return {
        hashlib.sha256(np.asarray(reference, dtype=np.uint8).tobytes()).hexdigest()
        for reference in np.asarray(references)
    }


@torch.inference_mode()
def _presence_validation_loss(
    model: nn.Module, loader: DataLoader, device: torch.device, criterion: nn.Module
) -> float:
    model.eval()
    total = 0.0
    count = 0
    for x, mask, target in loader:
        logits = model(x.to(device), mask.to(device)).squeeze(1)
        loss = criterion(logits, target.to(device))
        total += float(loss.item()) * int(target.shape[0])
        count += int(target.shape[0])
    return total / max(count, 1)


def train_presence_detector(
    experiment: ExperimentConfig,
    run: SimulationRunConfig,
    output: Path,
    device: torch.device,
) -> tuple[
    ExternalPresenceCNN,
    list[dict[str, float]],
    dict[str, Any],
    set[str],
]:
    _set_seed(run.seed)
    reference_root = output.parent / "stage1_inner_code_references"
    split_references: dict[str, dict[str, np.ndarray]] = {}
    split_validations: dict[str, list[dict[str, Any]]] = {}
    all_fingerprints: set[str] = set()
    for split, count in (
        ("train", run.train_inner_code_reads_per_category),
        ("validation", run.validation_inner_code_reads_per_category),
    ):
        split_references[split] = {}
        split_validations[split] = []
        for category in UNKNOWN_INNER_CODES:
            references, validation = generate_inner_code_references(
                category,
                count,
                run.seed,
                reference_root / split / f"{category.lower().replace('-', '_')}.fasta",
                namespace=f"stage1-{split}",
            )
            fingerprints = _reference_fingerprints(references)
            overlap = all_fingerprints & fingerprints
            if overlap:
                raise RuntimeError("inner-code molecular references overlap across train/validation")
            all_fingerprints.update(fingerprints)
            split_references[split][category] = references
            split_validations[split].append(asdict(validation))
    stage1_audit = {
        "train": audit_molecular_references(split_references["train"]),
        "validation": audit_molecular_references(split_references["validation"]),
        "cross_split_overlap": 0,
        "generator_validation": split_validations,
    }
    reference_root.mkdir(parents=True, exist_ok=True)
    (reference_root / "audit.json").write_text(
        json.dumps(stage1_audit, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    train = SimulatedPresenceReadDataset(
        experiment,
        "train",
        run.seed,
        run.train_known_reads_per_category,
        run.train_inner_code_reads_per_category,
        run.train_no_ecc_reads_per_category,
        split_references["train"],
    )
    validation = SimulatedPresenceReadDataset(
        experiment,
        "validation",
        run.seed,
        run.validation_known_reads_per_category,
        run.validation_inner_code_reads_per_category,
        run.validation_no_ecc_reads_per_category,
        split_references["validation"],
    )
    generator = torch.Generator().manual_seed(run.seed)
    train_loader = DataLoader(
        train, batch_size=run.batch_size, shuffle=True, generator=generator, num_workers=0
    )
    validation_loader = DataLoader(
        validation, batch_size=run.batch_size, shuffle=False, num_workers=0
    )
    model = ExternalPresenceCNN().to(device)
    positive_count = sum(
        stop - start
        for start, stop, category in train.segments
        if category not in NO_ECC_TYPES
    )
    negative_count = len(train) - positive_count
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([negative_count / positive_count], device=device)
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    best_loss = float("inf")
    best_state: dict[str, torch.Tensor] | None = None
    epochs_without_improvement = 0
    for epoch in range(1, run.presence_epochs + 1):
        model.train()
        total = 0.0
        count = 0
        for x, mask, target in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(x.to(device), mask.to(device)).squeeze(1)
            loss = criterion(logits, target.to(device))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total += float(loss.item()) * int(target.shape[0])
            count += int(target.shape[0])
        validation_loss = _presence_validation_loss(model, validation_loader, device, criterion)
        record = {
            "epoch": float(epoch),
            "train_loss": total / max(count, 1),
            "validation_loss": validation_loss,
        }
        history.append(record)
        print(
            f"presence epoch={epoch}/{run.presence_epochs} "
            f"train_loss={record['train_loss']:.6f} val_loss={validation_loss:.6f}",
            flush=True,
        )
        if validation_loss < best_loss:
            best_loss = validation_loss
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= 3:
                break
    if best_state is None:
        raise RuntimeError("presence training did not produce a checkpoint")
    model.load_state_dict(best_state, strict=True)
    output.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "state_dict": best_state,
            "seed": run.seed,
            "input_channels": 4,
            "role": "external_no_ecc_detector",
        },
        output / "external_presence_cnn.pt",
    )
    with (output / "presence_history.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=("epoch", "train_loss", "validation_loss"))
        writer.writeheader()
        writer.writerows(history)
    return model.eval(), history, stage1_audit, all_fingerprints


def _make_archives(
    experiment: ExperimentConfig,
    split: str,
    categories: Sequence[str],
    count_per_category: int,
    run: SimulationRunConfig,
) -> tuple[list[OneHotArchive], list[str], list[str]]:
    archives: list[OneHotArchive] = []
    labels: list[str] = []
    identifiers: list[str] = []
    for category in categories:
        for archive_id in range(count_per_category):
            archives.append(
                simulate_archive(
                    experiment,
                    category,
                    split=f"{split}-seed-{run.seed}",
                    archive_id=archive_id,
                    error_rate=run.test_error_rate,
                    molecules=run.molecules,
                    reads_per_molecule=run.reads_per_molecule,
                )
            )
            labels.append(category)
            identifiers.append(f"{split}:{category}:{archive_id}")
    return archives, labels, identifiers


def _max_base_run(reference: np.ndarray) -> int:
    values = np.asarray(reference, dtype=np.uint8).reshape(-1)
    best = run = 0
    previous = -1
    for value in values:
        current = int(value)
        run = run + 1 if current == previous else 1
        previous = current
        best = max(best, run)
    return best


def audit_molecular_references(reference_sets: dict[str, np.ndarray]) -> dict[str, Any]:
    """Fail closed on length/alphabet/constraint/uniqueness violations."""

    limits = {"NoECC-Constrained": 3, "HEDGES": 4, "DNA-Aeon": 3}
    category_audit: dict[str, dict[str, Any]] = {}
    global_sequences: list[bytes] = []
    for category, references_value in reference_sets.items():
        references = np.asarray(references_value, dtype=np.uint8)
        if references.ndim != 2 or references.shape[1] != 384:
            raise RuntimeError(f"{category} references are not all 384 nt")
        legal = bool(np.all((references >= 0) & (references <= 3)))
        if not legal:
            raise RuntimeError(f"{category} contains an illegal base")
        encoded = [reference.tobytes() for reference in references]
        unique_count = len(set(encoded))
        if unique_count != len(encoded):
            raise RuntimeError(f"{category} contains duplicate molecular references")
        runs = [_max_base_run(reference) for reference in references]
        limit = limits.get(category)
        violations = 0 if limit is None else sum(value > limit for value in runs)
        if violations:
            raise RuntimeError(f"{category} violates its homopolymer limit")
        global_sequences.extend(encoded)
        category_audit[category] = {
            "count": int(references.shape[0]),
            "length": 384,
            "all_bases_legal": legal,
            "unique_count": unique_count,
            "duplicate_count": len(encoded) - unique_count,
            "max_homopolymer": max(runs),
            "homopolymer_limit": limit,
            "homopolymer_violations": violations,
        }
    global_unique = len(set(global_sequences))
    if global_unique != len(global_sequences):
        raise RuntimeError("duplicate molecular references occur across categories")
    return {
        "all_categories_passed": True,
        "global_count": len(global_sequences),
        "global_unique_count": global_unique,
        "global_duplicate_count": len(global_sequences) - global_unique,
        "categories": category_audit,
    }


def run_simulated_validation(
    output: str | Path,
    run: SimulationRunConfig,
    weight_root: str | Path = DEFAULT_WEIGHT_ROOT,
    device: str | torch.device | None = None,
    reuse_calibration_from: str | Path | None = None,
) -> dict[str, Any]:
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    experiment = ExperimentConfig()

    reused_calibration_from: str | None = None
    stage1_reference_fingerprints: set[str] = set()
    stage1_data_audit: dict[str, Any] = {}
    if reuse_calibration_from is None:
        presence_model, history, stage1_data_audit, stage1_reference_fingerprints = train_presence_detector(
            experiment, run, output / "models", device
        )
        presence_best_validation_loss = min(row["validation_loss"] for row in history)
    else:
        source = Path(reuse_calibration_from).resolve()
        source_manifest = json.loads((source / "run_manifest.json").read_text(encoding="utf-8"))
        source_config = source_manifest["run_config"]
        calibration_keys = (
            "seed",
            "train_known_reads_per_category",
            "train_inner_code_reads_per_category",
            "train_no_ecc_reads_per_category",
            "validation_known_reads_per_category",
            "validation_inner_code_reads_per_category",
            "validation_no_ecc_reads_per_category",
            "validation_archives_per_category",
            "molecules",
            "reads_per_molecule",
            "known_acceptance",
        )
        current_config = asdict(run)
        mismatches = [
            key for key in calibration_keys if source_config.get(key) != current_config.get(key)
        ]
        if mismatches:
            raise RuntimeError(f"reused calibration config mismatch: {mismatches}")
        calibration_payload = np.load(source / "calibration_shared_logits.npz", allow_pickle=False)
        calibration_categories = calibration_payload["categories"].astype(str)
        for category in UNKNOWN_INNER_CODES:
            if not np.any(calibration_categories == category):
                raise RuntimeError(f"reused calibration is missing supervised {category} archives")
        checkpoint = torch.load(
            source / "models" / "external_presence_cnn.pt",
            map_location="cpu",
            weights_only=True,
        )
        if checkpoint.get("role") != "external_no_ecc_detector" or int(checkpoint["seed"]) != run.seed:
            raise RuntimeError("reused presence checkpoint metadata do not match this run")
        presence_model = ExternalPresenceCNN()
        presence_model.load_state_dict(checkpoint["state_dict"], strict=True)
        presence_model.to(device).eval()
        (output / "models").mkdir(parents=True, exist_ok=True)
        for relative in (
            Path("models/external_presence_cnn.pt"),
            Path("models/presence_history.csv"),
            Path("calibration_shared_logits.npz"),
            Path("thresholds.json"),
        ):
            shutil.copy2(source / relative, output / relative)
        presence_best_validation_loss = float(source_manifest["presence_best_validation_loss"])
        stage1_data_audit = dict(source_manifest["stage1_data_audit"])
        reused_calibration_from = str(source)
    presence = TorchPresenceDetector(presence_model, device=device, batch_size=run.batch_size)
    recognizer = build_primary_type_recognizer(
        weight_root=weight_root, device=device, batch_size=run.batch_size
    )

    if reuse_calibration_from is None:
        validation_categories = KNOWN_CODE_TYPES + NO_ECC_TYPES
        val_archives, val_labels, val_ids = _make_archives(
            experiment,
            "calibration",
            validation_categories,
            run.validation_archives_per_category,
            run,
        )
        calibration_inner_validations: list[dict[str, Any]] = []
        calibration_fingerprints: set[str] = set()
        for category in UNKNOWN_INNER_CODES:
            count = run.validation_archives_per_category * run.molecules
            references, validation = generate_inner_code_references(
                category,
                count,
                run.seed,
                output / "stage1_inner_code_references" / "calibration" /
                f"{category.lower().replace('-', '_')}.fasta",
                namespace="stage1-calibration",
            )
            fingerprints = _reference_fingerprints(references)
            if fingerprints & (stage1_reference_fingerprints | calibration_fingerprints):
                raise RuntimeError("inner-code references leak into stage1 calibration")
            calibration_fingerprints.update(fingerprints)
            calibration_inner_validations.append(asdict(validation))
            val_archives.extend(
                archives_from_references(
                    experiment,
                    category,
                    f"stage1-calibration-seed-{run.seed}",
                    references,
                    run.validation_archives_per_category,
                    run.molecules,
                    run.reads_per_molecule,
                    run.test_error_rate,
                )
            )
            val_labels.extend([category] * run.validation_archives_per_category)
            val_ids.extend(
                [
                    f"calibration:{category}:{archive_id}"
                    for archive_id in range(run.validation_archives_per_category)
                ]
            )
        stage1_reference_fingerprints.update(calibration_fingerprints)
        stage1_data_audit["calibration"] = {
            "generator_validation": calibration_inner_validations,
            "cross_split_overlap": 0,
        }
        validation_logits = collect_shared_logits(
            val_archives, val_labels, val_ids, presence, recognizer.code_type
        )
        validation_logits.save(output / "calibration_shared_logits.npz")
        thresholds, calibration = calibrate_thresholds(
            validation_logits.categories,
            validation_logits.presence_probabilities,
            validation_logits.type_logits,
            known_acceptance=run.known_acceptance,
        )
        thresholds.save(output / "thresholds.json")
    else:
        thresholds = IncrementalThresholds.load(output / "thresholds.json")
        calibration = dict(source_manifest["calibration"])
        calibration["reused_after_unknown_test_only_correction"] = True

    test_categories = KNOWN_CODE_TYPES + NO_ECC_TYPES
    test_archives, test_labels, test_ids = _make_archives(
        experiment, "test", test_categories, run.test_archives_per_category, run
    )
    factory = ReferenceFactory(experiment)
    test_namespace = f"test-seed-{run.seed}"
    reference_sets: dict[str, np.ndarray] = {
        category: np.stack(
            [
                factory.make_reference(category, test_namespace, archive_id, molecule_id)
                for archive_id in range(run.test_archives_per_category)
                for molecule_id in range(run.molecules)
            ]
        )
        for category in test_categories
    }
    inner_code_validation: list[dict[str, Any]] = []
    for unknown_category in UNKNOWN_INNER_CODES:
        reference_count = run.test_archives_per_category * run.molecules
        references, validation = generate_inner_code_references(
            unknown_category,
            reference_count,
            run.seed,
            output / "simulated_references" / f"{unknown_category.lower().replace('-', '_')}.fasta",
            namespace="final-test",
        )
        fingerprints = _reference_fingerprints(references)
        if fingerprints & stage1_reference_fingerprints:
            raise RuntimeError("final-test inner-code references leak into stage1 data")
        stage1_reference_fingerprints.update(fingerprints)
        reference_sets[unknown_category] = references
        unknown_archives = archives_from_references(
            experiment,
            unknown_category,
            f"test-seed-{run.seed}",
            references,
            run.test_archives_per_category,
            run.molecules,
            run.reads_per_molecule,
            run.test_error_rate,
        )
        test_archives.extend(unknown_archives)
        test_labels.extend([unknown_category] * run.test_archives_per_category)
        test_ids.extend(
            [
                f"test:{unknown_category}:{archive_id}"
                for archive_id in range(run.test_archives_per_category)
            ]
        )
        inner_code_validation.append(asdict(validation))
    reference_audit = audit_molecular_references(reference_sets)
    test_logits = collect_shared_logits(
        test_archives, test_labels, test_ids, presence, recognizer.code_type
    )
    test_logits.save(output / "test_shared_logits.npz")
    ecc_score_diagnostic = write_ecc_score_distributions(
        output / "test_shared_logits.npz", thresholds, output / "ecc_score_distribution"
    )
    comparisons = [
        compare_shared_logits(
            category,
            test_logits.presence_probabilities[index],
            test_logits.type_logits[index],
            thresholds,
            archive_id=test_ids[index],
        )
        for index, category in enumerate(test_labels)
    ]
    save_comparisons(
        comparisons,
        output / "shared_logits_predictions.csv",
        output / "shared_logits_comparison.json",
    )
    summary = json.loads((output / "shared_logits_comparison.json").read_text(encoding="utf-8"))
    manifest = {
        "experiment_name": (
            "基于作者盲识别核心的增量功能验证（正式模拟测试）"
            if run.formal
            else "基于作者盲识别核心的增量功能验证（模拟先导测试）"
        ),
        "result_scope": "formal_simulation" if run.formal else "engineering_pilot_not_paper_result",
        "simulation_warning": (
            "Known-ECC sequences are independently simulated and may not match the unavailable "
            "training distribution of the supplied author checkpoint."
        ),
        "author_models_py_modified": False,
        "primary_type_checkpoint": str(Path(weight_root).resolve() / "type" / "transformer_model_f10.6033.pt"),
        "rate_length_called": False,
        "unknown_inner_codes": list(UNKNOWN_INNER_CODES),
        "inner_codes_used_for_stage1_training": True,
        "inner_codes_used_for_stage1_validation": True,
        "inner_codes_used_for_tau1_calibration": True,
        "inner_codes_used_for_author_type_training": False,
        "inner_codes_used_for_tau2_calibration": False,
        "reused_calibration_from": reused_calibration_from,
        "inner_code_validation": inner_code_validation,
        "stage1_data_audit": stage1_data_audit,
        "reference_audit": reference_audit,
        "device": str(device),
        "run_config": asdict(run),
        "data_contract": {"one_hot": "[M,q,4,400]", "mask": "[M,q,400]"},
        "calibration": calibration,
        "thresholds": asdict(thresholds),
        "presence_best_validation_loss": presence_best_validation_loss,
        "comparison": summary,
        "ecc_score_diagnostic": ecc_score_diagnostic,
    }
    (output / "run_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return manifest


def refresh_simulated_comparison(output: str | Path) -> dict[str, Any]:
    """Recompute reporting from persisted scores without retraining or inference."""

    output = Path(output).resolve()
    payload = np.load(output / "test_shared_logits.npz", allow_pickle=False)
    categories = payload["categories"].astype(str)
    presence = payload["presence_probabilities"]
    logits = payload["type_logits"]
    archive_ids = payload["archive_ids"].astype(str)
    thresholds = IncrementalThresholds.load(output / "thresholds.json")
    comparisons = [
        compare_shared_logits(
            str(category), presence[index], logits[index], thresholds, str(archive_ids[index])
        )
        for index, category in enumerate(categories)
    ]
    save_comparisons(
        comparisons,
        output / "shared_logits_predictions.csv",
        output / "shared_logits_comparison.json",
    )
    summary = json.loads((output / "shared_logits_comparison.json").read_text(encoding="utf-8"))
    ecc_score_diagnostic = write_ecc_score_distributions(
        output / "test_shared_logits.npz",
        thresholds,
        output / "ecc_score_distribution",
    )
    manifest_path = output / "run_manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["comparison"] = summary
        manifest["ecc_score_diagnostic"] = ecc_score_diagnostic
        manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a simulated shared-logit pilot validation")
    parser.add_argument("--output", default="outputs/incremental_simulation_pilot")
    parser.add_argument("--weights-root", default=str(DEFAULT_WEIGHT_ROOT))
    parser.add_argument("--device", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-known-reads-per-category", type=int, default=500)
    parser.add_argument("--train-inner-code-reads-per-category", type=int, default=500)
    parser.add_argument("--train-no-ecc-reads-per-category", type=int, default=500)
    parser.add_argument("--validation-known-reads-per-category", type=int, default=120)
    parser.add_argument("--validation-inner-code-reads-per-category", type=int, default=120)
    parser.add_argument("--validation-no-ecc-reads-per-category", type=int, default=120)
    parser.add_argument("--presence-epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--validation-archives-per-category", type=int, default=5)
    parser.add_argument("--test-archives-per-category", type=int, default=5)
    parser.add_argument("--molecules", type=int, default=5)
    parser.add_argument("--reads-per-molecule", type=int, default=5)
    parser.add_argument("--test-error-rate", type=float, default=0.05)
    parser.add_argument("--known-acceptance", type=float, default=0.95)
    parser.add_argument(
        "--reuse-calibration-from",
        default=None,
        help="reuse matching stage1 weights and known/NoECC/inner-code tau1 calibration artifacts",
    )
    parser.add_argument(
        "--formal",
        action="store_true",
        help="use the preregistered M=20, q=50 and full read/archive counts",
    )
    parser.add_argument(
        "--refresh-existing",
        action="store_true",
        help="recompute reports from existing NPZ/threshold files only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.refresh_existing:
        print(json.dumps(refresh_simulated_comparison(args.output), indent=2, ensure_ascii=False))
        return
    values = {
        "train_known_reads_per_category": args.train_known_reads_per_category,
        "train_inner_code_reads_per_category": args.train_inner_code_reads_per_category,
        "train_no_ecc_reads_per_category": args.train_no_ecc_reads_per_category,
        "validation_known_reads_per_category": args.validation_known_reads_per_category,
        "validation_inner_code_reads_per_category": args.validation_inner_code_reads_per_category,
        "validation_no_ecc_reads_per_category": args.validation_no_ecc_reads_per_category,
        "presence_epochs": args.presence_epochs,
        "validation_archives_per_category": args.validation_archives_per_category,
        "test_archives_per_category": args.test_archives_per_category,
        "molecules": args.molecules,
        "reads_per_molecule": args.reads_per_molecule,
    }
    if args.formal:
        values.update(
            train_known_reads_per_category=20_000,
            train_inner_code_reads_per_category=20_000,
            train_no_ecc_reads_per_category=40_000,
            validation_known_reads_per_category=2_000,
            validation_inner_code_reads_per_category=2_000,
            validation_no_ecc_reads_per_category=4_000,
            presence_epochs=12,
            validation_archives_per_category=20,
            test_archives_per_category=50,
            molecules=20,
            reads_per_molecule=50,
        )
    run = SimulationRunConfig(
        seed=args.seed,
        formal=args.formal,
        batch_size=args.batch_size,
        test_error_rate=args.test_error_rate,
        known_acceptance=args.known_acceptance,
        **values,
    )
    result = run_simulated_validation(
        output=args.output,
        run=run,
        weight_root=args.weights_root,
        device=args.device,
        reuse_calibration_from=args.reuse_calibration_from,
    )
    print(json.dumps(result, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
