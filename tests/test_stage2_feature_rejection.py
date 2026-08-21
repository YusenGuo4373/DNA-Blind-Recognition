from __future__ import annotations

import numpy as np
import pytest
import torch

from author_baseline.original_models import create_original_model
from author_baseline.recognizer import OneHotArchive, OriginalTaskModel
from incremental_validation.embedding_rejection import (
    aggregate_read_distances,
    archive_mean_std,
    extract_archive_embeddings,
    fit_pca,
    transform_embeddings,
)
from incremental_validation.comparison import KNOWN_TYPES
from incremental_validation.stage2_feature_rejection import (
    KnownFeatureModel,
    acceptance_threshold,
    conformal_p_values,
    extract_logit_features,
    leave_one_out_conformal_scores,
    select_class_score,
    split_known_archives,
)


def test_logit_feature_shape_and_fixed_values() -> None:
    logits = np.zeros((2, 2, 3, 4), dtype=np.float64)
    logits[0, ..., 0] = 2.0
    bundle = extract_logit_features(logits)
    assert bundle.features.shape == (2, 43)
    assert len(bundle.feature_names) == 43
    assert bundle.closed_index.tolist() == [0, 0]
    assert np.all(np.isfinite(bundle.features))
    assert bundle.max_soft_vote_probability[0] > bundle.max_soft_vote_probability[1]


def test_known_split_is_stable_disjoint_and_rejects_unknown() -> None:
    categories = np.repeat(KNOWN_TYPES, 6)
    archive_ids = np.asarray([f"{category}:{i}" for category in KNOWN_TYPES for i in range(6)])
    fit1, cal1, _ = split_known_archives(categories, archive_ids, 42)
    fit2, cal2, _ = split_known_archives(categories, archive_ids, 42)
    np.testing.assert_array_equal(fit1, fit2)
    np.testing.assert_array_equal(cal1, cal2)
    assert set(archive_ids[fit1]).isdisjoint(set(archive_ids[cal1]))
    with pytest.raises(ValueError, match="unsupported"):
        split_known_archives(np.asarray([*categories, "HEDGES"]), np.asarray([*archive_ids, "x"]), 42)


def test_mahalanobis_distances_and_class_selection() -> None:
    rng = np.random.default_rng(7)
    categories = np.repeat(KNOWN_TYPES, 8)
    features = np.vstack(
        [rng.normal(index * 4.0, 0.2, size=(8, 6)) for index in range(4)]
    )
    model = KnownFeatureModel.fit(features, categories)
    diagonal = model.diagonal_distances(features)
    shrinkage = model.shrinkage_distances(features)
    assert diagonal.shape == shrinkage.shape == (32, 4)
    predicted = np.repeat(np.arange(4), 8)
    np.testing.assert_allclose(
        select_class_score(diagonal, predicted, "predicted"), diagonal.min(axis=1),
        rtol=0.25, atol=3.0,
    )
    assert np.all(model.shrinkage >= 0.0)
    assert np.all(model.shrinkage <= 1.0)
    assert np.all(np.isfinite(model.condition_numbers))


def test_conformal_pvalues_and_known_only_threshold() -> None:
    calibration = [np.asarray([1.0, 2.0, 3.0]) for _ in range(4)]
    distances = np.asarray([[0.5, 2.0, 4.0, 1.0]])
    pvalues = conformal_p_values(distances, calibration)
    np.testing.assert_allclose(pvalues[0], [1.0, 0.75, 0.25, 1.0])
    categories = np.repeat(KNOWN_TYPES, 3)
    matrix = np.tile(np.asarray([[1.0], [2.0], [3.0]]), (4, 4))
    closed = np.repeat(np.arange(4), 3)
    scores = leave_one_out_conformal_scores(matrix, categories, closed, "predicted")
    threshold = acceptance_threshold(scores, 0.95)
    assert np.mean(scores <= threshold) >= 0.95


def test_q_m_aggregation_matches_explicit_means() -> None:
    logits = np.arange(2 * 3 * 4, dtype=np.float64).reshape(1, 2, 3, 4) / 10.0
    bundle = extract_logit_features(logits)
    flat = logits.reshape(-1, 4)
    maximum = flat.max(axis=1)
    expected_energy = -(maximum + np.log(np.exp(flat - maximum[:, None]).sum(axis=1))).mean()
    assert bundle.energy[0] == pytest.approx(expected_energy)


def test_absent_reporting_category_is_not_a_feature_error() -> None:
    logits = np.zeros((1, 1, 1, 4), dtype=np.float64)
    bundle = extract_logit_features(logits)
    assert bundle.features.shape == (1, 43)


def test_fc_prehook_does_not_change_author_transformer_logits() -> None:
    torch.manual_seed(4)
    model = create_original_model("transformer", num_classes=4).eval()
    task = OriginalTaskModel("code_type", model, KNOWN_TYPES, device="cpu", batch_size=2)
    bases = np.arange(32, dtype=np.int64) % 4
    one_hot = np.eye(4, dtype=np.float32)[bases].T[None, None]
    archive = OneHotArchive(one_hot, np.ones((1, 1, 32), dtype=np.bool_))
    baseline = task.read_logits(archive).numpy()
    hooked, embedding = extract_archive_embeddings(task, archive)
    assert np.array_equal(baseline, hooked)
    assert embedding.shape == (1, 1, 128)


def test_pca_archive_features_and_read_aggregation() -> None:
    rng = np.random.default_rng(10)
    embeddings = rng.normal(size=(4, 2, 3, 128))
    mean, components, eigenvalues = fit_pca(embeddings, 16)
    transformed = transform_embeddings(embeddings, mean, components)
    assert transformed.shape == (4, 2, 3, 16)
    assert np.all(np.diff(eigenvalues) <= 1e-12)
    archive_features = archive_mean_std(transformed)
    assert archive_features.shape == (4, 32)
    distances = np.arange(4 * 2 * 3 * 4, dtype=np.float64).reshape(-1, 4)
    aggregated = aggregate_read_distances(distances, (4, 2, 3))
    np.testing.assert_allclose(aggregated, distances.reshape(4, 2, 3, 4).mean(2).mean(1))
