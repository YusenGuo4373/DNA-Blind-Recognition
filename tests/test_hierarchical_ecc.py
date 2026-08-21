from __future__ import annotations

import unittest
import importlib.util

import numpy as np

from hierarchical_ecc.coding import (
    BCH_SPECS,
    LDPCCode,
    bases_to_bits,
    bch_encode,
    bits_to_bases,
    convolutional_encode,
    generate_constrained_uncoded,
    lt_encode,
    max_homopolymer_length,
    polar_encode,
    polar_encode_with_positions,
    polynomial_remainder,
    stable_seed,
)
from hierarchical_ecc.config import ExperimentConfig, KNOWN_CODE_TYPES, NO_ECC_TYPES
from hierarchical_ecc.data import (
    ReferenceFactory,
    assert_split_isolation,
    inject_ids,
    mask_pad_read,
)
from hierarchical_ecc.metrics import binary_aupr, binary_auroc, confusion_matrix, macro_f1_score
from hierarchical_ecc.voting import (
    Thresholds,
    hierarchical_decision,
    select_presence_threshold,
    select_unknown_threshold,
    two_level_soft_vote,
)


TORCH_AVAILABLE = importlib.util.find_spec("torch") is not None


def bits_as_int(bits: np.ndarray) -> int:
    value = 0
    for bit in bits:
        value = (value << 1) | int(bit)
    return value


class CodingTests(unittest.TestCase):
    def test_stable_seed(self) -> None:
        self.assertEqual(stable_seed("a", 1), stable_seed("a", 1))
        self.assertNotEqual(stable_seed("train", 1), stable_seed("test", 1))

    def test_base_mapping_round_trip(self) -> None:
        bits = np.array((0, 0, 0, 1, 1, 0, 1, 1), dtype=np.uint8)
        bases = bits_to_bases(bits)
        np.testing.assert_array_equal(bases, (0, 1, 2, 3))
        np.testing.assert_array_equal(bases_to_bits(bases), bits)

    def test_bch_codewords_are_divisible(self) -> None:
        rng = np.random.default_rng(1)
        for spec in BCH_SPECS:
            for _ in range(10):
                codeword = bch_encode(rng.integers(0, 2, spec.k, dtype=np.uint8), spec)
                self.assertEqual(polynomial_remainder(bits_as_int(codeword), spec.generator), 0)

    def test_ldpc_syndrome(self) -> None:
        code = LDPCCode.create(64, 48, seed=4)
        message = np.random.default_rng(5).integers(0, 2, 48, dtype=np.uint8)
        codeword = code.encode(message)
        self.assertFalse(np.any((code.H @ codeword) & 1))

    def test_convolutional_lengths(self) -> None:
        message = np.zeros(300, dtype=np.uint8)
        expected = {"1_2": 600, "1_3": 900, "1_4": 1200, "3_4": 400}
        for rate, length in expected.items():
            self.assertEqual(convolutional_encode(message, rate).size, length)

    def test_polar_linearity(self) -> None:
        rng = np.random.default_rng(6)
        left = rng.integers(0, 2, 43, dtype=np.uint8)
        right = rng.integers(0, 2, 43, dtype=np.uint8)
        np.testing.assert_array_equal(
            polar_encode(left ^ right, 128), polar_encode(left, 128) ^ polar_encode(right, 128)
        )
        positions = np.sort(rng.choice(128, size=43, replace=False))
        np.testing.assert_array_equal(
            polar_encode_with_positions(left ^ right, 128, positions),
            polar_encode_with_positions(left, 128, positions)
            ^ polar_encode_with_positions(right, 128, positions),
        )

    def test_lt_is_pure_xor(self) -> None:
        source = np.random.default_rng(7).integers(0, 2, size=(8, 768), dtype=np.uint8)
        droplets = lt_encode(source, 20, seed=8)
        for droplet in droplets:
            expected = np.bitwise_xor.reduce(source[np.asarray(droplet.indices)], axis=0)
            np.testing.assert_array_equal(droplet.bits, expected)


class DataTests(unittest.TestCase):
    def setUp(self) -> None:
        self.config = ExperimentConfig()

    def test_reference_lengths_and_constraints(self) -> None:
        factory = ReferenceFactory(self.config)
        for category in KNOWN_CODE_TYPES + NO_ECC_TYPES:
            reference = factory.make_reference(category, "unit", 0, 0)
            self.assertEqual(reference.size, 384)
            self.assertTrue(np.all(reference < 4))
        constrained = factory.make_reference("NoECC-Constrained", "unit", 2, 3)
        gc = np.mean((constrained == 2) | (constrained == 3))
        self.assertGreaterEqual(gc, 0.45)
        self.assertLessEqual(gc, 0.55)
        self.assertLessEqual(max_homopolymer_length(constrained), 3)

    def test_standalone_constrained_generator(self) -> None:
        values = generate_constrained_uncoded(384, np.random.default_rng(9))
        self.assertLessEqual(max_homopolymer_length(values), 3)

    def test_two_bit_mask_semantics(self) -> None:
        features, mask = mask_pad_read(np.array((0, 1, 2, 3), dtype=np.uint8), 6)
        np.testing.assert_array_equal(features[:, :4], ((0, 0, 1, 1), (0, 1, 0, 1)))
        self.assertTrue(np.all(features[:, 4:] == 0))
        np.testing.assert_array_equal(mask, (True, True, True, True, False, False))

    def test_ids_empirical_rates(self) -> None:
        rng = np.random.default_rng(10)
        template = rng.integers(0, 4, 100_000, dtype=np.uint8)
        _, stats = inject_ids(template, 0.1, 0.1, 0.1, rng)
        self.assertAlmostEqual(stats.insertions / stats.input_bases, 0.1, delta=0.006)
        self.assertAlmostEqual(stats.deletions / stats.input_bases, 0.1, delta=0.006)
        self.assertAlmostEqual(
            stats.substitutions / (stats.input_bases - stats.deletions), 0.1, delta=0.006
        )

    def test_fountain_is_test_only(self) -> None:
        factory = ReferenceFactory(self.config)
        with self.assertRaises(ValueError):
            factory.make_fountain_archive("train", 0, 2)
        with self.assertRaises(ValueError):
            factory.make_fountain_archive("calibration", 0, 2)
        with self.assertRaises(ValueError):
            factory.make_fountain_archive("calibration-seed-42", 0, 2)
        references, indices = factory.make_fountain_archive("test", 0, 3)
        self.assertEqual(references.shape, (3, 384))
        self.assertEqual(len(indices), 3)

    def test_split_isolation(self) -> None:
        assert_split_isolation(self.config)


class VotingAndMetricTests(unittest.TestCase):
    def test_two_level_vote(self) -> None:
        values = np.arange(24, dtype=np.float64).reshape(2, 3, 4)
        np.testing.assert_allclose(two_level_soft_vote(values), values.mean(axis=1).mean(axis=0))

    def test_presence_threshold_uses_macro_f1(self) -> None:
        threshold, score = select_presence_threshold(
            np.array((0.1, 0.2, 0.8, 0.9)), np.array((0, 0, 1, 1))
        )
        self.assertGreater(threshold, 0.2)
        self.assertLess(threshold, 0.8)
        self.assertEqual(score, 1.0)

    def test_unknown_threshold_known_quantile(self) -> None:
        scores = np.arange(20, dtype=np.float64)
        threshold = select_unknown_threshold(scores, 0.95)
        self.assertGreaterEqual(np.mean(scores <= threshold), 0.95)

    def test_hierarchical_stop_unknown_and_known(self) -> None:
        no_ecc = hierarchical_decision(
            np.full((2, 3), 0.1), None, Thresholds(0.5, -2.0)
        )
        self.assertEqual(no_ecc.status, "no_ecc")
        self.assertIsNone(no_ecc.unknown_score)

        logits = np.zeros((2, 3, 4), dtype=np.float64)
        unknown = hierarchical_decision(
            np.full((2, 3), 0.9), logits, Thresholds(0.5, -2.0)
        )
        self.assertEqual(unknown.status, "unknown_ecc")
        self.assertIsNone(unknown.code_type)

        logits[..., 2] = 5.0
        known = hierarchical_decision(
            np.full((2, 3), 0.9), logits, Thresholds(0.5, 0.0)
        )
        self.assertEqual(known.status, "known_ecc")
        self.assertEqual(known.code_type, "LDPC")
        self.assertIsNone(known.code_rate)
        self.assertIsNone(known.code_length)

    def test_metrics(self) -> None:
        truth = np.array((0, 0, 1, 1))
        scores = np.array((0.1, 0.2, 0.8, 0.9))
        self.assertEqual(binary_auroc(truth, scores), 1.0)
        self.assertEqual(binary_aupr(truth, scores), 1.0)
        matrix = confusion_matrix(truth, truth, (0, 1))
        np.testing.assert_array_equal(matrix, ((2, 0), (0, 2)))
        self.assertEqual(macro_f1_score(truth, truth, (0, 1)), 1.0)


@unittest.skipUnless(TORCH_AVAILABLE, "PyTorch is not installed in this interpreter")
class TorchIntegrationTests(unittest.TestCase):
    def test_transformer_ignores_padded_values(self) -> None:
        import torch

        from hierarchical_ecc.config import ModelConfig
        from hierarchical_ecc.models import DNAReadTransformer

        torch.manual_seed(11)
        model = DNAReadTransformer(
            ModelConfig(
                d_model=16,
                nhead=4,
                num_layers=1,
                dim_feedforward=32,
                dropout=0.0,
                max_length=8,
            ),
            output_dim=4,
        ).eval()
        valid_mask = torch.tensor(((1, 1, 1, 1, 0, 0, 0, 0),), dtype=torch.bool)
        clean = torch.zeros((1, 2, 8), dtype=torch.float32)
        clean[:, :, :4] = torch.tensor((((0, 0, 1, 1), (0, 1, 0, 1)),))
        dirty_padding = clean.clone()
        dirty_padding[:, :, 4:] = torch.rand((1, 2, 4))
        with torch.inference_mode():
            clean_output = model(clean, valid_mask)
            dirty_output = model(dirty_padding, valid_mask)
        torch.testing.assert_close(clean_output, dirty_output, rtol=0.0, atol=1e-6)

    def test_training_dataset_is_deterministic_and_has_no_fountain(self) -> None:
        from hierarchical_ecc.data import ReadClassificationDataset

        dataset = ReadClassificationDataset(
            ExperimentConfig(),
            task="presence",
            split="train",
            seed=42,
            known_reads_per_type=1,
            no_ecc_reads_per_subtype=1,
        )
        categories = tuple(segment[2] for segment in dataset._segments)
        self.assertNotIn("Fountain", categories)
        first = dataset[0]
        repeated = dataset[0]
        np.testing.assert_array_equal(first["x"].numpy(), repeated["x"].numpy())
        np.testing.assert_array_equal(
            first["valid_mask"].numpy(), repeated["valid_mask"].numpy()
        )


if __name__ == "__main__":
    unittest.main()
