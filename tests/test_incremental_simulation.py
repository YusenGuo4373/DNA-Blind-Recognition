import numpy as np
import pytest
import torch
from pathlib import Path

from hierarchical_ecc.config import ExperimentConfig
from incremental_validation.simulation import (
    ExternalPresenceCNN,
    audit_molecular_references,
    bases_to_one_hot,
    simulate_archive,
)
from incremental_validation.inner_codes import (
    DNA_AEON_EXECUTABLE,
    HEDGES_EXECUTABLE,
    generate_inner_code_references,
)


def test_base_one_hot_and_padding_contract() -> None:
    bases = np.array([[[0, 1, 2, 3, 255]]], dtype=np.uint8)
    mask = np.array([[[True, True, True, True, False]]])
    encoded = bases_to_one_hot(bases, mask)
    assert encoded.shape == (1, 1, 4, 5)
    np.testing.assert_array_equal(encoded[0, 0, :, :4], np.eye(4, dtype=np.float32))
    np.testing.assert_array_equal(encoded[0, 0, :, 4], np.zeros(4, dtype=np.float32))


def test_simulated_archive_uses_author_contract() -> None:
    archive = simulate_archive(
        ExperimentConfig(), "BCH", "test-contract", 0, 0.01, molecules=2, reads_per_molecule=3
    )
    assert archive.validate() == (2, 3, 400)
    x = np.asarray(archive.one_hot)
    mask = np.asarray(archive.mask)
    assert np.allclose(x.sum(axis=2)[mask], 1.0)
    assert np.allclose(x.sum(axis=2)[~mask], 0.0)


def test_fountain_remains_forbidden_in_calibration() -> None:
    with pytest.raises(ValueError, match="test-only"):
        simulate_archive(
            ExperimentConfig(), "Fountain", "calibration-seed-42", 0, 0.0, 1, 1
        )


def test_external_presence_model_has_four_channel_interface() -> None:
    model = ExternalPresenceCNN()
    logits = model(torch.zeros(2, 4, 20), torch.ones(2, 20, dtype=torch.bool))
    assert logits.shape == (2, 1)


@pytest.mark.skipif(
    not HEDGES_EXECUTABLE.is_file() or not DNA_AEON_EXECUTABLE.is_file(),
    reason="official inner-code adapters are not built",
)
@pytest.mark.parametrize("category", ["HEDGES", "DNA-Aeon"])
def test_official_unknown_inner_code_smoke(category: str, tmp_path: Path) -> None:
    references, validation = generate_inner_code_references(
        category, count=2, seed=42, output_fasta=tmp_path / f"{category}.fasta"
    )
    assert references.shape == (2, 384)
    assert np.all((references >= 0) & (references <= 3))
    assert validation.length == 384
    assert validation.duplicate_count == 0
    assert validation.homopolymer_violations == 0
    if category == "HEDGES":
        assert validation.fixed_primer_prefix_matches == 0
        assert validation.fixed_primer_suffix_matches == 0


@pytest.mark.skipif(
    not HEDGES_EXECUTABLE.is_file(), reason="official HEDGES adapter is not built"
)
def test_inner_code_namespaces_produce_disjoint_molecules(tmp_path: Path) -> None:
    train, _ = generate_inner_code_references(
        "HEDGES", 4, 42, tmp_path / "train.fasta", namespace="stage1-train"
    )
    test, _ = generate_inner_code_references(
        "HEDGES", 4, 42, tmp_path / "test.fasta", namespace="final-test"
    )
    assert {row.tobytes() for row in train}.isdisjoint({row.tobytes() for row in test})


def test_reference_audit_rejects_duplicates() -> None:
    duplicate = np.zeros((2, 384), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="duplicate"):
        audit_molecular_references({"BCH": duplicate})
