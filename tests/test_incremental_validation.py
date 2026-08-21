from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from author_baseline.recognizer import OneHotArchive, OriginalTaskModel
from incremental_validation.collector import TorchPresenceDetector, collect_shared_logits
from incremental_validation.comparison import (
    KNOWN_TYPES,
    calibrate_thresholds,
    compare_shared_logits,
    summarize_comparisons,
)


def known_logits(class_index: int, molecules: int = 2, reads: int = 3) -> np.ndarray:
    logits = np.zeros((molecules, reads, 4), dtype=np.float64)
    logits[..., class_index] = 5.0
    return logits


class FixedReadModel(nn.Module):
    def __init__(self, values: tuple[float, ...]):
        super().__init__()
        self.register_buffer("values", torch.tensor(values, dtype=torch.float32))
        self.calls = 0

    def forward(self, x, mask):
        del mask
        self.calls += 1
        return self.values.unsqueeze(0).expand(x.shape[0], -1)


def one_hot_archive() -> OneHotArchive:
    bases = np.arange(16) % 4
    read = np.eye(4, dtype=np.float32)[bases].T
    return OneHotArchive(
        np.tile(read[None, None], (2, 3, 1, 1)),
        np.ones((2, 3, 16), dtype=np.float32),
    )


class SharedLogitExperimentTests(unittest.TestCase):
    def validation_arrays(self):
        categories = np.array(KNOWN_TYPES + ("NoECC-Random", "NoECC-Constrained"))
        presence = np.stack(
            [np.full((2, 3), 0.9) for _ in KNOWN_TYPES]
            + [np.full((2, 3), 0.1), np.full((2, 3), 0.2)]
        )
        logits = np.stack(
            [known_logits(index) for index in range(4)]
            + [np.zeros((2, 3, 4)), np.zeros((2, 3, 4))]
        )
        return categories, presence, logits

    def test_calibration_rejects_fountain(self) -> None:
        categories, presence, logits = self.validation_arrays()
        categories = np.concatenate((categories, ("Fountain",)))
        presence = np.concatenate((presence, np.full((1, 2, 3), 0.9)), axis=0)
        logits = np.concatenate((logits, np.zeros((1, 2, 3, 4))), axis=0)
        with self.assertRaisesRegex(ValueError, "forbidden"):
            calibrate_thresholds(categories, presence, logits)

    def test_shared_logits_comparison_improves_false_known_rates(self) -> None:
        categories, presence, logits = self.validation_arrays()
        thresholds, calibration = calibrate_thresholds(categories, presence, logits)
        self.assertEqual(calibration["stage2_unknown_samples_used"], 0.0)

        test_categories = list(KNOWN_TYPES) + ["NoECC-Random", "NoECC-Constrained", "Fountain"]
        test_presence = [np.full((2, 3), 0.9) for _ in KNOWN_TYPES]
        test_presence.extend((np.full((2, 3), 0.1), np.full((2, 3), 0.2), np.full((2, 3), 0.9)))
        test_logits = [known_logits(index) for index in range(4)]
        test_logits.extend(
            (
                known_logits(0),
                known_logits(1),
                np.zeros((2, 3, 4), dtype=np.float64),
            )
        )
        comparisons = [
            compare_shared_logits(category, p, logits_value, thresholds, archive_id=index)
            for index, (category, p, logits_value) in enumerate(
                zip(test_categories, test_presence, test_logits)
            )
        ]
        summary = summarize_comparisons(comparisons)
        self.assertTrue(summary["shared_type_logits"])
        self.assertEqual(summary["closed_set"]["no_ecc_output_as_known_rate"], 1.0)
        self.assertEqual(summary["closed_set"]["unknown_ecc_output_as_known_rate"], 1.0)
        self.assertEqual(summary["cascade"]["no_ecc_output_as_known_rate"], 0.0)
        self.assertEqual(summary["cascade"]["unknown_ecc_output_as_known_rate"], 0.0)
        self.assertEqual(summary["cascade"]["no_ecc_specificity"], 1.0)
        self.assertEqual(summary["cascade"]["unknown_ecc_gate_recall"], 1.0)
        self.assertEqual(summary["cascade"]["unknown_energy_rejection_rate"], 1.0)
        self.assertEqual(summary["cascade"]["unknown_ecc_recall"], 1.0)
        self.assertEqual(summary["cascade"]["known_ecc_acceptance_rate"], 1.0)
        self.assertEqual(summary["cascade"]["known_type_macro_f1_end_to_end"], 1.0)
        self.assertTrue(all(item.code_rate is None and item.code_length is None for item in comparisons))

    def test_inner_codes_supervise_tau1_but_not_tau2(self) -> None:
        categories, presence, logits = self.validation_arrays()
        baseline, _ = calibrate_thresholds(categories, presence, logits)
        categories = np.concatenate((categories, ("HEDGES", "DNA-Aeon")))
        presence = np.concatenate(
            (presence, np.full((2, 2, 3), 0.95, dtype=np.float64)), axis=0
        )
        extreme = np.full((2, 2, 3, 4), -100.0, dtype=np.float64)
        logits = np.concatenate((logits, extreme), axis=0)
        thresholds, calibration = calibrate_thresholds(categories, presence, logits)
        self.assertEqual(calibration["stage1_supervised_inner_code_samples"], 2.0)
        self.assertEqual(calibration["stage2_unknown_samples_used"], 0.0)
        self.assertEqual(thresholds.unknown_energy, baseline.unknown_energy)

    def test_collector_calls_author_model_once_per_archive(self) -> None:
        type_network = FixedReadModel((0.0, 0.0, 5.0, 0.0))
        type_model = OriginalTaskModel(
            "code_type", type_network, KNOWN_TYPES, device="cpu", batch_size=32
        )
        presence_network = FixedReadModel((0.0, 2.0))
        presence = TorchPresenceDetector(presence_network, device="cpu", batch_size=32)
        dataset = collect_shared_logits(
            [one_hot_archive(), one_hot_archive()],
            ["LDPC", "Fountain"],
            ["a", "b"],
            presence,
            type_model,
        )
        self.assertEqual(dataset.presence_probabilities.shape, (2, 2, 3))
        self.assertEqual(dataset.type_logits.shape, (2, 2, 3, 4))
        self.assertEqual(type_network.calls, 2)


if __name__ == "__main__":
    unittest.main()
