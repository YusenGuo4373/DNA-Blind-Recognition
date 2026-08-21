from __future__ import annotations

from pathlib import Path

import numpy as np

from hierarchical_ecc.coding import stable_seed
from hierarchical_ecc.config import ExperimentConfig
from hierarchical_ecc.data import ReferenceFactory, sample_noisy_read
from incremental_validation.stage5_structural_proxy import three_state_consensus
from incremental_validation.stage7_channel import (
    generate_archive_payload,
    sample_noisy_read_with_stats,
)
from incremental_validation.stage7_repeatability import (
    ENERGY_THRESHOLD,
    LABELS7,
    PRESENCE_THRESHOLD,
    PROXY_THRESHOLD,
    TRUTH_LABELS,
    _classification_report_from_matrix,
    _load_models,
    _safe_classification_report,
    _write_csv,
    archive_to_one_hot,
    frozen_decisions,
    hierarchical_bootstrap,
    risk_coverage_metrics,
    seven_class_confusion,
    streaming_two_level_soft_vote,
    two_level_soft_vote,
)


ROOT = Path(__file__).resolve().parents[1]


def test_stable_seed_is_repeatable_and_cross_seed_references_do_not_overlap() -> None:
    assert stable_seed("stage7", 46, "BCH", 0) == stable_seed("stage7", 46, "BCH", 0)
    factory = ReferenceFactory(ExperimentConfig())
    first = {
        factory.make_reference("BCH", "stage7-test-seed-46", archive, molecule).tobytes()
        for archive in range(2)
        for molecule in range(20)
    }
    second = {
        factory.make_reference("BCH", "stage7-test-seed-47", archive, molecule).tobytes()
        for archive in range(2)
        for molecule in range(20)
    }
    assert len(first) == len(second) == 40
    assert first.isdisjoint(second)


def test_stage7_channel_matches_repository_scalar_channel() -> None:
    reference = np.arange(384, dtype=np.uint16).astype(np.uint8) % 4
    seed = stable_seed("stage7-channel-equivalence")
    expected = sample_noisy_read(reference, 0.05, 130, 384, 400, np.random.default_rng(seed))
    observed, stats = sample_noisy_read_with_stats(reference, 0.05, 130, 384, 400, np.random.default_rng(seed))
    assert np.array_equal(expected, observed)
    assert stats[0] >= 130
    assert all(value >= 0 for value in stats)


def test_archive_generation_has_exact_m_q_one_hot_and_mask_semantics() -> None:
    references = np.stack([np.arange(384, dtype=np.uint16).astype(np.uint8) % 4 for _ in range(20)])
    spec = {
        "references": references,
        "molecules": 20,
        "reads_per_molecule": 50,
        "reference_length": 384,
        "min_read_length": 130,
        "max_read_length": 384,
        "padded_length": 400,
        "noise_category": "BCH",
        "split": "stage7-test-shape",
        "archive_index": 0,
        "error_rate": 0.05,
    }
    payload = generate_archive_payload(spec)
    archive = archive_to_one_hot(payload)
    assert archive.one_hot.shape == (20, 50, 4, 400)
    assert archive.mask.shape == (20, 50, 400)
    assert np.all(archive.one_hot.sum(axis=2)[archive.mask] == 1)
    assert np.all(archive.one_hot.sum(axis=2)[~archive.mask] == 0)
    first_valid = int(np.flatnonzero(archive.mask[0, 0])[0])
    base = int(payload["bases"][0, 0, first_valid])
    assert archive.one_hot[0, 0, base, first_valid] == 1


def test_two_level_streaming_vote_matches_materialized_vote() -> None:
    rng = np.random.default_rng(3)
    probabilities = rng.random((20, 50, 4))
    probabilities /= probabilities.sum(axis=-1, keepdims=True)
    expected = two_level_soft_vote(probabilities)
    observed = streaming_two_level_soft_vote(probabilities.sum(axis=1), np.full(20, 50))
    assert np.allclose(expected, observed)


def test_matrix_classification_report_matches_explicit_labels() -> None:
    matrix = np.asarray([[3, 1], [2, 4]], dtype=np.int64)
    truth = np.asarray(["A"] * 4 + ["B"] * 6)
    prediction = np.asarray(["A"] * 3 + ["B"] + ["A"] * 2 + ["B"] * 4)
    explicit = _safe_classification_report(truth, prediction, ("A", "B"))
    direct = _classification_report_from_matrix(matrix, ("A", "B"))
    assert direct == explicit


def test_csv_writer_uses_union_of_fields(tmp_path: Path) -> None:
    output = tmp_path / "mixed.csv"
    _write_csv(output, [{"status": "missing"}, {"status": "available", "score": 0.5}])
    text = output.read_text(encoding="utf-8")
    assert text.splitlines()[0] == "status,score"
    assert "available,0.5" in text


def test_three_state_and_cascade_rules_are_frozen() -> None:
    states = three_state_consensus(
        np.asarray([False, True, False, True]),
        np.asarray([False, True, True, False]),
    )
    assert states.tolist() == ["known_ecc", "unknown_ecc", "uncertain_ecc", "uncertain_ecc"]
    assert frozen_decisions("BCH", PRESENCE_THRESHOLD - 1e-6, ENERGY_THRESHOLD, PROXY_THRESHOLD) == {
        "energy_only": "no_ecc",
        "proxy_only": "no_ecc",
        "three_state": "no_ecc",
        "G_all": "no_ecc",
    }
    disagreement = frozen_decisions("BCH", 1.0, ENERGY_THRESHOLD + 1.0, PROXY_THRESHOLD - 0.1)
    assert disagreement["three_state"] == "uncertain_ecc"


def test_seven_class_matrix_and_risk_metrics() -> None:
    truth = list(TRUTH_LABELS)
    observed = ["no_ecc", "BCH", "Convolutional", "LDPC", "Polar", "unknown_ecc", "uncertain_ecc"]
    matrix = np.asarray(seven_class_confusion(truth, observed))
    assert matrix.shape == (7, 7)
    assert matrix.sum() == 7
    assert tuple(LABELS7) == ("no_ecc", "uncertain_ecc", "unknown_ecc", "BCH", "Convolutional", "LDPC", "Polar")
    metrics = risk_coverage_metrics(truth, observed)
    assert metrics["unknown_risk_coverage"] == 1.0
    assert metrics["manual_review_rate_all_archives"] == 1 / 7


def _synthetic_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in (46, 47, 48, 49, 50):
        for category in TRUTH_LABELS:
            for archive in range(2):
                closed = category if category in ("BCH", "Convolutional", "LDPC", "Polar") else "BCH"
                proxy_output = "unknown_ecc" if category in ("HEDGES", "DNA-Aeon") else ("no_ecc" if category == "no_ecc" else closed)
                rows.append(
                    {
                        "seed": seed,
                        "truth_category": category,
                        "closed_output": closed,
                        "ecc_score": 0.0 if category == "no_ecc" else 1.0,
                        "energy_score": 0.0,
                        "proxy_score": 1.0 if category in ("HEDGES", "DNA-Aeon") else 0.0,
                        "energy_only_output": "no_ecc" if category == "no_ecc" else closed,
                        "proxy_only_output": proxy_output,
                        "three_state_output": proxy_output,
                        "G_all_output": proxy_output,
                        "read_confusion_matrix": np.eye(4, dtype=int).tolist() if category in ("BCH", "Convolutional", "LDPC", "Polar") else np.zeros((4, 4), dtype=int).tolist(),
                    }
                )
    return rows


def test_hierarchical_bootstrap_uses_archive_rows_and_is_deterministic() -> None:
    rows = _synthetic_rows()
    first = hierarchical_bootstrap(rows, repetitions=20, seed=11)
    second = hierarchical_bootstrap(rows, repetitions=20, seed=11)
    assert first == second
    assert first["unit"] == "archive"
    assert first["repetitions"] == 20


def test_frozen_models_and_threshold_files_are_unchanged() -> None:
    source = ROOT / "outputs" / "inner_codes_formal_seed42"
    stage5 = ROOT / "outputs" / "stage5_structural_embedding_proxy_seed42"
    author, presence, _proxy = _load_models(source, stage5, __import__("torch").device("cpu"), 16)
    assert all(not parameter.requires_grad for parameter in author.code_type.model.parameters())
    assert all(not parameter.requires_grad for parameter in presence.model.parameters())
    thresholds = __import__("json").loads((source / "thresholds.json").read_text(encoding="utf-8"))
    frozen = __import__("json").loads((stage5 / "frozen_detector_config.json").read_text(encoding="utf-8"))
    assert thresholds["ecc_presence"] == PRESENCE_THRESHOLD
    assert frozen["thresholds"]["0.98"]["energy"] == ENERGY_THRESHOLD
    assert frozen["thresholds"]["0.98"]["proxy"] == PROXY_THRESHOLD
    assert frozen["HEDGES_DNA_Aeon_used"] is False
