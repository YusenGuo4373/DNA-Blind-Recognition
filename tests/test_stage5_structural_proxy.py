from __future__ import annotations

import numpy as np

from author_baseline.recognizer import OneHotArchive
from incremental_validation.stage5_structural_proxy import (
    ProxyClassifier, combined_archive_features, sequence_structure_features,
    three_state_consensus,
)


def _archive(molecules: int = 2, reads: int = 3, length: int = 12) -> OneHotArchive:
    bases = np.arange(molecules * reads * length, dtype=np.uint8).reshape(molecules, reads, length) % 4
    one_hot = np.eye(4, dtype=np.float32)[bases].transpose(0, 1, 3, 2)
    mask = np.ones((molecules, reads, length), dtype=bool)
    return OneHotArchive(one_hot, mask)


def test_sequence_features_ignore_padding_values() -> None:
    archive = _archive(); first = sequence_structure_features(archive)
    one_hot = np.asarray(archive.one_hot).copy(); mask = np.asarray(archive.mask).copy()
    mask[..., -2:] = False; one_hot[..., -2:] = 0
    padded_a = sequence_structure_features(OneHotArchive(one_hot, mask))
    one_hot[..., -2:] = 99
    padded_b = sequence_structure_features(OneHotArchive(one_hot, mask))
    np.testing.assert_allclose(padded_a, padded_b)
    assert first.ndim == 1 and np.isfinite(first).all()


def test_combined_features_include_sequence_embedding_and_logits() -> None:
    archive = _archive(1, 2, 12)
    logits = np.zeros((1, 2, 4), dtype=np.float32)
    embeddings = np.zeros((1, 2, 128), dtype=np.float32)
    features = combined_archive_features(archive, logits, embeddings)
    assert features.size > 256
    assert np.isfinite(features).all()


def test_proxy_classifier_is_deterministic_and_orders_separable_data() -> None:
    rng = np.random.default_rng(7)
    known = rng.normal(-1, 0.1, size=(30, 8)); proxy = rng.normal(1, 0.1, size=(30, 8))
    x = np.vstack((known, proxy)); y = np.r_[np.zeros(30), np.ones(30)]
    first = ProxyClassifier.fit(x, y, 4, 0.1); second = ProxyClassifier.fit(x, y, 4, 0.1)
    np.testing.assert_allclose(first.score(x), second.score(x))
    assert first.score(proxy).mean() > first.score(known).mean()


def test_energy_proxy_consensus_is_three_state() -> None:
    states = three_state_consensus(np.array([False, True, False, True]), np.array([False, True, True, False]))
    assert states.tolist() == ["known_ecc", "unknown_ecc", "uncertain_ecc", "uncertain_ecc"]


def test_consensus_never_calls_a_disagreement_unknown() -> None:
    energy = np.array([True, False, True, False])
    proxy = np.array([False, True, True, False])
    states = three_state_consensus(energy, proxy)
    assert states[:2].tolist() == ["uncertain_ecc", "uncertain_ecc"]
    assert states[2:].tolist() == ["unknown_ecc", "known_ecc"]
