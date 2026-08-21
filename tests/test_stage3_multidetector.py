from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from incremental_validation.stage3_multidetector import (
    PROXY_FAMILIES,
    apply_multidetector_rule,
    build_candidate_registry,
    generate_proxy_references,
    risk_coverage_metrics,
    select_candidate,
)


def test_and_or_consensus_majority_and_uncertain_logic() -> None:
    rejected = np.asarray([[False, False], [True, True], [True, False]])
    assert apply_multidetector_rule(rejected, "consensus").tolist() == ["known_ecc", "unknown_ecc", "uncertain_ecc"]
    assert apply_multidetector_rule(rejected, "and_reject").tolist() == ["known_ecc", "unknown_ecc", "known_ecc"]
    assert apply_multidetector_rule(rejected, "or_reject").tolist() == ["known_ecc", "unknown_ecc", "unknown_ecc"]
    assert apply_multidetector_rule(rejected, "majority").tolist() == ["known_ecc", "unknown_ecc", "uncertain_ecc"]


def test_risk_coverage_metrics() -> None:
    states = np.asarray(["known_ecc", "uncertain_ecc", "unknown_ecc", "unknown_ecc"])
    truth = np.asarray([False, False, True, False])
    metrics = risk_coverage_metrics(states, truth)
    assert metrics["uncertain_rate"] == 0.25
    assert metrics["decisive_coverage"] == 0.75
    assert metrics["decisive_binary_accuracy"] == pytest.approx(2 / 3)


def test_proxy_families_are_unique_384nt_and_split_isolated(tmp_path: Path) -> None:
    all_splits: list[set[bytes]] = []
    for split in ("fit", "calibration", "validation", "final-test"):
        split_values: set[bytes] = set()
        for family in PROXY_FAMILIES:
            references, metadata = generate_proxy_references(
                family, split, 2, 2, 42, tmp_path / split / f"{family}.fasta"
            )
            assert references.shape == (4, 384)
            assert np.all(references <= 3)
            assert len({row.tobytes() for row in references}) == 4
            assert all("payload_seed" in row and "encoder_seed" in row for row in metadata)
            split_values.update(row.tobytes() for row in references)
        all_splits.append(split_values)
    for left in range(len(all_splits)):
        for right in range(left + 1, len(all_splits)):
            assert all_splits[left].isdisjoint(all_splits[right])


def test_target_inner_codes_forbidden_as_proxy(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unsupported|forbidden"):
        generate_proxy_references("HEDGES", "fit", 1, 1, 42, tmp_path / "x.fasta")


def test_registry_has_all_targets_and_rule_types() -> None:
    thresholds = {
        str(target): {name: 0.0 for name in (
            "global_energy", "pca16_archive_diagonal_minimum",
            "raw128_archive_conformal_maximum_pvalue", "logits_diagonal_minimum",
        )} for target in (0.98, 0.95, 0.93)
    }
    registry = build_candidate_registry(thresholds)
    assert "t98_energy_only" in registry
    assert "t98_pca16_archive_diagonal_minimum_only" in registry
    assert "t95_raw128_archive_conformal_maximum_pvalue_only" in registry
    assert "t93_logits_diagonal_minimum_only" in registry
    assert "t95_energy_raw_consensus" in registry
    assert "t93_all4_majority" in registry


def test_lofo_selection_never_receives_held_family() -> None:
    thresholds = {
        str(target): {name: 0.5 for name in (
            "global_energy", "pca16_archive_diagonal_minimum",
            "raw128_archive_conformal_maximum_pvalue", "logits_diagonal_minimum",
        )} for target in (0.98, 0.95, 0.93)
    }
    registry = build_candidate_registry(thresholds)
    candidates = [name for name in registry if name.startswith("t95_")]
    known_states = {name: np.asarray(["known_ecc"] * 20) for name in candidates}
    proxy = {
        family: {detector: np.ones(5) for detector in thresholds["0.95"]}
        for family in ("ReedSolomon", "Turbo")
    }
    selected = select_candidate(registry, candidates, known_states, proxy, list(proxy), "balanced")
    assert selected in candidates
