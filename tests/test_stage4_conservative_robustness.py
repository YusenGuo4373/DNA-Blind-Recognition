from __future__ import annotations

import numpy as np
import pytest
import torch

from incremental_validation.stage4_conservative_robustness import (
    bootstrap_archive_metrics,
    conservative_outputs,
    freeze_module,
    noise_seed,
    NULL_PARAMETER_OUTPUT,
    prefix_presence,
    prefix_soft_vote,
    recalibrate_energy_threshold,
    _reference_fingerprints,
    seven_class_confusion,
)


def test_q_and_m_prefixes_match_explicit_two_level_aggregation() -> None:
    rng = np.random.default_rng(2)
    logits = rng.normal(size=(3, 5, 7, 4))
    presence = rng.random(size=(3, 5, 7))
    probabilities, energy = prefix_soft_vote(logits, 2, 3)
    selected = logits[:, :2, :3]
    softmax = np.exp(selected - selected.max(-1, keepdims=True))
    softmax /= softmax.sum(-1, keepdims=True)
    np.testing.assert_allclose(probabilities, softmax.mean(2).mean(1))
    np.testing.assert_allclose(prefix_presence(presence, 2, 3), presence[:, :2, :3].mean(2).mean(1))
    assert energy.shape == (3,)


def test_fixed_threshold_is_pure_input_and_not_modified() -> None:
    categories = np.asarray(["BCH", "NoECC-Random"])
    presence = np.asarray([[[0.9]], [[0.1]]])
    logits = np.zeros((2, 1, 1, 4)); logits[0, 0, 0, 0] = 5.0
    tau1, tau2 = 0.5, -1.0
    first = conservative_outputs(categories, presence, logits, tau1, tau2, 1, 1)
    second = conservative_outputs(categories, presence, logits, tau1, tau2, 1, 1)
    assert tau1 == 0.5 and tau2 == -1.0
    np.testing.assert_array_equal(first["output"], second["output"])


def test_noise_seed_is_stable_and_isolates_all_channel_coordinates() -> None:
    value = noise_seed(43, 0.05, "BCH", 1, 2, 3)
    assert value == noise_seed(43, 0.05, "BCH", 1, 2, 3)
    assert value != noise_seed(44, 0.05, "BCH", 1, 2, 3)
    assert value != noise_seed(43, 0.10, "BCH", 1, 2, 3)
    assert value != noise_seed(43, 0.05, "BCH", 1, 2, 4)


def test_stable_seed_reference_pools_have_zero_cross_seed_overlap() -> None:
    def pool(seed: int) -> dict[str, np.ndarray]:
        rng = np.random.default_rng(noise_seed(seed, 0.05, "BCH", 0, 0, 0))
        return {"BCH": rng.integers(0, 4, size=(12, 384), dtype=np.uint8)}

    assert not (_reference_fingerprints(pool(43)) & _reference_fingerprints(pool(44)))


def test_freeze_module_disables_gradients_and_switches_to_eval() -> None:
    module = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.Dropout())
    frozen = freeze_module(module)
    assert frozen is module
    assert not module.training
    assert all(not parameter.requires_grad for parameter in module.parameters())


def test_parameter_recognition_interfaces_remain_null() -> None:
    assert NULL_PARAMETER_OUTPUT == {"code_rate": None, "code_length": None}


def test_bootstrap_resamples_archives_not_reads() -> None:
    categories = np.asarray(["BCH", "BCH", "Convolutional", "Convolutional", "LDPC", "LDPC", "Polar", "Polar", "NoECC-Random", "NoECC-Random", "NoECC-Constrained", "NoECC-Constrained", "HEDGES", "HEDGES", "DNA-Aeon", "DNA-Aeon"])
    closed = np.asarray(["BCH", "BCH", "Convolutional", "Convolutional", "LDPC", "LDPC", "Polar", "Polar", "BCH", "BCH", "BCH", "BCH", "BCH", "BCH", "BCH", "BCH"])
    output = closed.copy(); output[8:12] = "no_ecc"; output[12:] = "unknown_ecc"
    result = {"output": output, "closed_output": closed, "ecc_score": np.ones(16), "energy": np.zeros(16), "closed_index": np.zeros(16), "type_probabilities": np.zeros((16, 4))}
    audit = bootstrap_archive_metrics(categories, result, 43, repetitions=20)
    assert audit["unit"] == "archive"
    assert audit["repetitions"] == 20


def test_seven_class_confusion_shape() -> None:
    categories = ["NoECC-Random", "HEDGES", "BCH", "Convolutional", "LDPC", "Polar"]
    outputs = ["no_ecc", "unknown_ecc", "BCH", "Convolutional", "LDPC", "Polar"]
    matrix = np.asarray(seven_class_confusion(categories, outputs))
    assert matrix.shape == (7, 7)
    assert matrix.trace() == 6


def test_recalibration_rejects_target_inner_codes() -> None:
    with pytest.raises(ValueError, match="only four known"):
        recalibrate_energy_threshold(None, {"HEDGES": np.zeros((1, 384), dtype=np.uint8)}, 43, None)
