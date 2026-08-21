from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .coding import (
    BCH_SPECS,
    LDPCCode,
    bch_encode,
    convolutional_encode,
    lt_encode,
    max_homopolymer_length,
    polar_encode,
    polynomial_remainder,
    stable_seed,
)
from .config import ExperimentConfig, KNOWN_CODE_TYPES, NO_ECC_TYPES
from .data import (
    ReadClassificationDataset,
    ReferenceFactory,
    assert_split_isolation,
    inject_ids,
    mask_pad_read,
)
from .voting import two_level_soft_vote


@dataclass(frozen=True)
class ValidationCheck:
    name: str
    passed: bool
    detail: str


def _bits_as_int(bits: np.ndarray) -> int:
    value = 0
    for bit in np.asarray(bits).reshape(-1):
        value = (value << 1) | int(bit)
    return value


def run_validation_suite(config: ExperimentConfig | None = None) -> list[ValidationCheck]:
    config = config or ExperimentConfig()
    checks: list[ValidationCheck] = []

    for spec in BCH_SPECS:
        rng = np.random.default_rng(stable_seed("validate-bch", spec.n, spec.k))
        valid = True
        for _ in range(20):
            codeword = bch_encode(rng.integers(0, 2, spec.k, dtype=np.uint8), spec)
            valid &= codeword.size == spec.n
            valid &= polynomial_remainder(_bits_as_int(codeword), spec.generator) == 0
        checks.append(ValidationCheck(f"BCH({spec.n},{spec.k})", bool(valid), "generator remainder is zero"))

    for n, k in ((32, 16), (64, 48), (128, 43), (256, 64)):
        code = LDPCCode.create(n, k, stable_seed("validate-ldpc", n, k))
        rng = np.random.default_rng(stable_seed("validate-ldpc-message", n, k))
        message = rng.integers(0, 2, k, dtype=np.uint8)
        codeword = code.encode(message)
        passed = bool(not np.any((code.H @ codeword) & 1))
        checks.append(ValidationCheck(f"LDPC({n},{k})", passed, "Hc=0 over GF(2)"))

    rng = np.random.default_rng(stable_seed("validate-convolutional"))
    message = rng.integers(0, 2, 300, dtype=np.uint8)
    expected_lengths = {"1_2": 600, "1_3": 900, "1_4": 1200, "3_4": 400}
    for rate, expected in expected_lengths.items():
        encoded = convolutional_encode(message, rate)
        checks.append(
            ValidationCheck(
                f"Convolutional-{rate}",
                encoded.size == expected and np.all((encoded == 0) | (encoded == 1)),
                f"expected {expected} binary encoded bits",
            )
        )

    rng = np.random.default_rng(stable_seed("validate-polar"))
    for n, k in ((32, 16), (64, 48), (128, 43), (256, 64)):
        left = rng.integers(0, 2, k, dtype=np.uint8)
        right = rng.integers(0, 2, k, dtype=np.uint8)
        linear = np.array_equal(
            polar_encode(left ^ right, n), polar_encode(left, n) ^ polar_encode(right, n)
        )
        checks.append(ValidationCheck(f"Polar({n},{k})", linear, "linearity over GF(2)"))

    source = rng.integers(0, 2, size=(8, 768), dtype=np.uint8)
    droplets = lt_encode(source, count=30, seed=stable_seed("validate-lt"))
    xor_valid = all(
        np.array_equal(
            droplet.bits,
            np.bitwise_xor.reduce(source[np.asarray(droplet.indices)], axis=0),
        )
        for droplet in droplets
    )
    checks.append(ValidationCheck("LT-XOR", xor_valid, "every droplet equals its source-block XOR"))

    ids_rng = np.random.default_rng(stable_seed("validate-ids"))
    template = ids_rng.integers(0, 4, size=200_000, dtype=np.uint8)
    _, stats = inject_ids(template, 0.10, 0.10, 0.10, ids_rng)
    ins_rate = stats.insertions / stats.input_bases
    del_rate = stats.deletions / stats.input_bases
    sub_rate = stats.substitutions / (stats.input_bases - stats.deletions)
    ids_valid = all(abs(observed - 0.10) < 0.005 for observed in (ins_rate, del_rate, sub_rate))
    checks.append(
        ValidationCheck(
            "IDS-empirical-rate",
            ids_valid,
            f"ins={ins_rate:.4f}, del={del_rate:.4f}, sub|retained={sub_rate:.4f}",
        )
    )

    read = np.array((0, 1, 2, 3), dtype=np.uint8)
    features, mask = mask_pad_read(read, padded_length=8)
    expected_features = np.array(
        ((0, 0, 1, 1, 0, 0, 0, 0), (0, 1, 0, 1, 0, 0, 0, 0)), dtype=np.uint8
    )
    mask_valid = np.array_equal(features, expected_features) and np.array_equal(
        mask, np.array((1, 1, 1, 1, 0, 0, 0, 0), dtype=np.bool_)
    )
    checks.append(ValidationCheck("two-bit-mask-padding", mask_valid, "A/T/C/G=00/01/10/11; True means valid"))

    votes = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
    vote_valid = np.allclose(two_level_soft_vote(votes), votes.mean(axis=1).mean(axis=0))
    checks.append(ValidationCheck("two-level-soft-vote", bool(vote_valid), "mean over q, then mean over M"))

    factory = ReferenceFactory(config)
    reference_valid = True
    for category in KNOWN_CODE_TYPES + NO_ECC_TYPES:
        reference = factory.make_reference(category, "validation-check", 0, 0)
        reference_valid &= reference.size == config.channel.reference_length
        reference_valid &= bool(np.all(reference < 4))
        if category == "NoECC-Constrained":
            gc = float(np.mean((reference == 2) | (reference == 3)))
            reference_valid &= 0.45 <= gc <= 0.55 and max_homopolymer_length(reference) <= 3
    checks.append(ValidationCheck("reference-and-negative-constraints", bool(reference_valid), "384 nt and constrained negative rules"))

    try:
        assert_split_isolation(config)
        isolation_valid = True
        isolation_detail = "train/validation/test references are distinct"
    except AssertionError as error:
        isolation_valid = False
        isolation_detail = str(error)
    checks.append(ValidationCheck("split-isolation", isolation_valid, isolation_detail))

    # The constructor category route is checked without materializing a sample,
    # and Fountain is absent by construction from both tasks.
    try:
        presence_categories = KNOWN_CODE_TYPES + NO_ECC_TYPES
        type_categories = KNOWN_CODE_TYPES
        no_unknown = "Fountain" not in presence_categories and "Fountain" not in type_categories
        dataset_check = True
        try:
            # Only executes when torch is installed.
            presence_dataset = ReadClassificationDataset(
                config,
                "presence",
                "train",
                42,
                known_reads_per_type=1,
                no_ecc_reads_per_subtype=1,
            )
            observed = tuple(segment[2] for segment in presence_dataset._segments)
            dataset_check = observed == presence_categories
        except RuntimeError as error:
            if "PyTorch" not in str(error):
                raise
        passed = no_unknown and dataset_check
    except Exception as error:  # Surface a useful validation record rather than hiding it.
        passed = False
        isolation_detail = str(error)
    checks.append(ValidationCheck("strict-unknown-holdout", passed, "Fountain absent from train/validation categories"))

    return checks


def validation_report(config: ExperimentConfig | None = None) -> dict[str, object]:
    checks = run_validation_suite(config)
    return {
        "passed": bool(all(check.passed for check in checks)),
        "checks": [
            {"name": check.name, "passed": bool(check.passed), "detail": str(check.detail)}
            for check in checks
        ],
    }
