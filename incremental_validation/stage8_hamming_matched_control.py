from __future__ import annotations

"""Independent observation-model-matched OOD control using extended Hamming codes.

The control code is not used to fit or calibrate any component.  Its 384-nt
references use the same two-bit nucleotide mapping, archive organization, and
IDS channel as the candidate-set data.  The script evaluates the already fixed
coded/uncoded discriminator and post-hoc OOD detector.
"""

import argparse
import csv
import json
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from author_baseline.weights import DEFAULT_WEIGHT_ROOT, build_primary_type_recognizer
from hierarchical_ecc.coding import bits_to_bases, stable_seed
from hierarchical_ecc.config import ExperimentConfig
from hierarchical_ecc.data import ReferenceFactory
from incremental_validation.collector import TorchPresenceDetector
from incremental_validation.comparison import IncrementalThresholds
from incremental_validation.inner_codes import archives_from_references
from incremental_validation.simulation import ExternalPresenceCNN
from incremental_validation.stage5_structural_proxy import ProxyClassifier
from incremental_validation.stage6_proxy_robustness import score_archives


KNOWN_TYPES = ("BCH", "Convolutional", "LDPC", "Polar")


def extended_hamming_8_4(payload: np.ndarray) -> np.ndarray:
    """Encode four-bit messages as systematic extended-Hamming (8,4) blocks."""

    d = np.asarray(payload, dtype=np.uint8).reshape(-1, 4)
    d1, d2, d3, d4 = (d[:, index] for index in range(4))
    p1 = d1 ^ d2 ^ d4
    p2 = d1 ^ d3 ^ d4
    p3 = d2 ^ d3 ^ d4
    first_seven = np.column_stack((p1, p2, d1, p3, d2, d3, d4))
    p0 = np.bitwise_xor.reduce(first_seven, axis=1)
    return np.column_stack((first_seven, p0)).reshape(-1)


def make_hamming_references(seed: int, count: int) -> np.ndarray:
    references: list[np.ndarray] = []
    seen: set[bytes] = set()
    for index in range(count):
        rng = np.random.default_rng(stable_seed("stage8-hamming", seed, index))
        payload = rng.integers(0, 2, size=96 * 4, dtype=np.uint8)
        reference = bits_to_bases(extended_hamming_8_4(payload)).astype(np.uint8)
        if reference.shape != (384,):
            raise RuntimeError("Hamming control must produce a 384-nt reference")
        key = reference.tobytes()
        if key in seen:
            raise RuntimeError("duplicate Hamming control reference")
        seen.add(key)
        references.append(reference)
    return np.stack(references)


def maximum_run(reference: np.ndarray) -> int:
    boundaries = np.flatnonzero(np.r_[True, reference[1:] != reference[:-1], True])
    return int(np.diff(boundaries).max())


def composition_summary(references: np.ndarray) -> dict[str, float]:
    refs = np.asarray(references, dtype=np.uint8)
    gc = np.mean(np.logical_or(refs == 1, refs == 2), axis=1)
    runs = np.asarray([maximum_run(reference) for reference in refs], dtype=np.float64)
    pair_counts = np.zeros((refs.shape[0], 16), dtype=np.float64)
    for left in range(4):
        for right in range(4):
            pair_counts[:, 4 * left + right] = np.sum(
                (refs[:, :-1] == left) & (refs[:, 1:] == right), axis=1
            )
    pair_frequency = pair_counts / (refs.shape[1] - 1)
    uniform_pair_l1 = np.abs(pair_frequency - 1.0 / 16.0).sum(axis=1)
    return {
        "reference_length": float(refs.shape[1]),
        "gc_mean": float(gc.mean()),
        "gc_std": float(gc.std()),
        "maximum_run_mean": float(runs.mean()),
        "maximum_run_q95": float(np.quantile(runs, 0.95)),
        "dinucleotide_uniform_l1_mean": float(uniform_pair_l1.mean()),
    }


def bootstrap_rate(values: np.ndarray, seed: int, replicates: int = 2000) -> dict[str, float]:
    x = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    sampled = x[rng.integers(0, x.size, size=(replicates, x.size))].mean(axis=1)
    return {
        "estimate": float(x.mean()),
        "lower95": float(np.quantile(sampled, 0.025)),
        "upper95": float(np.quantile(sampled, 0.975)),
    }


def run(
    source: str | Path,
    stage5: str | Path,
    output: str | Path,
    seeds: Sequence[int],
    archives_per_seed: int,
    device: str,
    resume: bool,
) -> dict[str, object]:
    source = Path(source).resolve()
    stage5 = Path(stage5).resolve()
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    cache = output / "cache"
    cache.mkdir(exist_ok=True)

    config = ExperimentConfig()
    selected_device = torch.device(device if torch.cuda.is_available() else "cpu")
    recognizer = build_primary_type_recognizer(device=selected_device, batch_size=64)
    for parameter in recognizer.code_type.model.parameters():
        parameter.requires_grad_(False)

    checkpoint = torch.load(
        source / "models" / "external_presence_cnn.pt",
        map_location="cpu",
        weights_only=True,
    )
    presence_model = ExternalPresenceCNN()
    presence_model.load_state_dict(checkpoint["state_dict"])
    presence = TorchPresenceDetector(presence_model, device=selected_device, batch_size=64)
    detector = ProxyClassifier.load(stage5 / "structural_embedding_proxy_detector.npz")
    presence_threshold = IncrementalThresholds.load(source / "thresholds.json").ecc_presence
    ood_threshold = 0.8268975481068935

    rows: list[dict[str, object]] = []
    composition: dict[str, object] = {}
    factory = ReferenceFactory(config)
    for seed in seeds:
        reference_count = archives_per_seed * 20
        hamming = make_hamming_references(seed, reference_count)
        composition[str(seed)] = {"Hamming": composition_summary(hamming)}
        for known_type in KNOWN_TYPES:
            known = np.stack(
                [
                    factory.make_reference(
                        known_type,
                        f"stage8-composition-seed-{seed}",
                        index,
                        0,
                    )
                    for index in range(reference_count)
                ]
            )
            composition[str(seed)][known_type] = composition_summary(known)

        cache_path = cache / f"hamming_seed{seed}.npz"
        if resume and cache_path.is_file():
            with np.load(cache_path, allow_pickle=False) as payload:
                scored = {key: payload[key] for key in payload.files}
        else:
            archives = archives_from_references(
                config,
                "Hamming",
                f"stage8-hamming-test-seed-{seed}",
                hamming,
                archives_per_seed,
                20,
                50,
                0.05,
            )
            scored = score_archives(
                recognizer.code_type,
                presence,
                detector,
                archives,
                np.asarray(["Hamming"] * archives_per_seed),
                np.asarray(
                    [f"stage8:hamming:{seed}:{index}" for index in range(archives_per_seed)]
                ),
            )
            np.savez_compressed(cache_path, **scored)

        for index in range(archives_per_seed):
            coded = bool(scored["ecc"][index] >= presence_threshold)
            rejected = bool(scored["proxy"][index] > ood_threshold)
            if not coded:
                cascade = "uncoded"
            elif rejected:
                cascade = "unknown_code"
            else:
                cascade = KNOWN_TYPES[int(scored["closed"][index])]
            rows.append(
                {
                    "seed": seed,
                    "archive_id": str(scored["archive_ids"][index]),
                    "presence_score": float(scored["ecc"][index]),
                    "ood_score": float(scored["proxy"][index]),
                    "coded": int(coded),
                    "ood_rejected": int(rejected),
                    "cascade_output": cascade,
                    "closed_output": KNOWN_TYPES[int(scored["closed"][index])],
                }
            )

    with (output / "per_archive_predictions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "composition_audit.json").write_text(
        json.dumps(composition, indent=2), encoding="utf-8"
    )

    coded = np.asarray([row["coded"] for row in rows], dtype=bool)
    rejected = np.asarray([row["ood_rejected"] for row in rows], dtype=bool)
    unknown = np.asarray([row["cascade_output"] == "unknown_code" for row in rows])
    forced = np.asarray([row["cascade_output"] in KNOWN_TYPES for row in rows])
    result: dict[str, object] = {
        "control_code": "extended Hamming (8,4)",
        "development_use": "none",
        "seeds": list(seeds),
        "archives_per_seed": archives_per_seed,
        "archives": len(rows),
        "M": 20,
        "q": 50,
        "IDS": {"insertion": 0.05, "deletion": 0.05, "substitution": 0.05},
        "presence_threshold": presence_threshold,
        "ood_threshold": ood_threshold,
        "coded_detection": bootstrap_rate(coded, 8101),
        "ood_rejection_before_presence_gate": bootstrap_rate(rejected, 8102),
        "cascade_unknown_code_output": bootstrap_rate(unknown, 8103),
        "cascade_forced_known_output": bootstrap_rate(forced, 8104),
        "cascade_uncoded_output": bootstrap_rate(~coded, 8105),
    }
    (output / "aggregate_metrics.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="outputs/inner_codes_formal_seed42")
    parser.add_argument("--stage5", default="outputs/stage5_structural_embedding_proxy_seed42")
    parser.add_argument("--output", default="outputs/stage8_hamming_matched_control")
    parser.add_argument("--seeds", nargs="+", type=int, default=[51, 52, 53, 54, 55])
    parser.add_argument("--archives-per-seed", type=int, default=50)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args(argv)
    print(
        json.dumps(
            run(
                args.source,
                args.stage5,
                args.output,
                args.seeds,
                args.archives_per_seed,
                args.device,
                args.resume,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
