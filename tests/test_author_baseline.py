from __future__ import annotations

import unittest

import numpy as np
import torch
from torch import nn

from author_baseline.cascade import CascadeThresholds, HierarchicalAuthorAdapter
from author_baseline.original_models import AUTHOR_MODEL_NAMES, create_original_model
from author_baseline.recognizer import OneHotArchive, OriginalBlindRecognizer, OriginalTaskModel
from author_baseline.soft_voting import author_soft_vote
from author_baseline.vendor_guard import verify_vendor_snapshot


class FixedPresence:
    def __init__(self, probability: float):
        self.probability = probability

    def predict_probabilities(self, archive: OneHotArchive) -> np.ndarray:
        molecules, reads, _ = archive.validate()
        return np.full((molecules, reads), self.probability, dtype=np.float64)


class CountingLogitModel(nn.Module):
    def __init__(self, logits: tuple[float, ...]):
        super().__init__()
        self.register_buffer("fixed_logits", torch.tensor(logits, dtype=torch.float32))
        self.calls = 0

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        del mask
        self.calls += 1
        return self.fixed_logits.unsqueeze(0).expand(x.shape[0], -1)


def archive() -> OneHotArchive:
    bases = np.arange(16) % 4
    one_read = np.eye(4, dtype=np.float32)[bases].T
    one_hot = np.tile(one_read[None, None, :, :], (2, 3, 1, 1))
    mask = np.ones((2, 3, 16), dtype=np.float32)
    return OneHotArchive(one_hot, mask)


class VendorAndModelTests(unittest.TestCase):
    def test_vendor_snapshot_is_unchanged(self) -> None:
        verification = verify_vendor_snapshot()
        self.assertTrue(verification.valid, verification.to_dict())

    def test_original_models_accept_four_channel_one_hot(self) -> None:
        x = torch.zeros((2, 4, 400), dtype=torch.float32)
        x[:, 0, :] = 1.0
        mask = torch.ones((2, 400), dtype=torch.float32)
        for name in AUTHOR_MODEL_NAMES:
            model = create_original_model(name, num_classes=4).eval()
            with torch.inference_mode():
                logits = model(x, mask)
            self.assertEqual(tuple(logits.shape), (2, 4))

    def test_author_soft_vote_matches_flattened_original_formula(self) -> None:
        logits = torch.randn((5, 7, 4), generator=torch.Generator().manual_seed(42))
        vote = author_soft_vote(logits)
        expected = torch.softmax(logits.reshape(-1, 4), dim=1).mean(dim=0)
        torch.testing.assert_close(vote.archive_probabilities, expected)


class CascadeTests(unittest.TestCase):
    def task(self, task: str, logits: tuple[float, ...], labels: tuple[str, ...]):
        model = CountingLogitModel(logits)
        adapter = OriginalTaskModel(task, model, labels, device="cpu", batch_size=32)
        return adapter, model

    def test_no_ecc_stops_before_author_model(self) -> None:
        type_task, type_model = self.task(
            "code_type", (0.0, 0.0, 0.0, 0.0), ("BCH", "Convolutional", "LDPC", "Polar")
        )
        recognizer = OriginalBlindRecognizer(type_task)
        cascade = HierarchicalAuthorAdapter(
            FixedPresence(0.1), recognizer, CascadeThresholds(0.5, -2.0)
        )
        decision = cascade.predict(archive())
        self.assertEqual(decision.status, "no_ecc")
        self.assertEqual(type_model.calls, 0)

    def test_unknown_stops_before_parameter_models(self) -> None:
        type_task, type_model = self.task(
            "code_type", (0.0, 0.0, 0.0, 0.0), ("BCH", "Convolutional", "LDPC", "Polar")
        )
        rate_task, rate_model = self.task("code_rate", (1.0, 0.0), ("1/2", "3/4"))
        recognizer = OriginalBlindRecognizer(type_task, code_rate=rate_task)
        cascade = HierarchicalAuthorAdapter(
            FixedPresence(0.9), recognizer, CascadeThresholds(0.5, -2.0)
        )
        decision = cascade.predict(archive())
        self.assertEqual(decision.status, "unknown_ecc")
        self.assertGreater(type_model.calls, 0)
        self.assertEqual(rate_model.calls, 0)

    def test_known_runs_original_type_rate_and_length_models(self) -> None:
        type_task, _ = self.task(
            "code_type", (0.0, 0.0, 5.0, 0.0), ("BCH", "Convolutional", "LDPC", "Polar")
        )
        rate_task, rate_model = self.task("code_rate", (0.0, 4.0), ("1/2", "3/4"))
        length_task, length_model = self.task("code_length", (3.0, 0.0), ("128", "256"))
        recognizer = OriginalBlindRecognizer(type_task, rate_task, length_task)
        cascade = HierarchicalAuthorAdapter(
            FixedPresence(0.9), recognizer, CascadeThresholds(0.5, 0.0)
        )
        decision = cascade.predict(archive())
        self.assertEqual(decision.status, "known_ecc")
        self.assertEqual(decision.code_type, "LDPC")
        self.assertEqual(decision.code_rate, "3/4")
        self.assertEqual(decision.code_length, "128")
        self.assertGreater(rate_model.calls, 0)
        self.assertGreater(length_model.calls, 0)


if __name__ == "__main__":
    unittest.main()
