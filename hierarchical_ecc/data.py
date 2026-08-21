from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Iterable

import numpy as np

from .coding import (
    BCH_SPECS,
    LDPCCode,
    bch_encode,
    bits_to_bases,
    concatenate_codewords,
    convolutional_encode,
    generate_constrained_uncoded,
    lt_encode,
    polar_encode_with_positions,
    stable_seed,
)
from .config import (
    ExperimentConfig,
    KNOWN_CODE_TYPES,
    NO_ECC_TYPES,
    UNKNOWN_CODE_TYPE,
)

try:
    import torch
    from torch.utils.data import Dataset
except ImportError:  # Pure-NumPy validation remains usable without PyTorch.
    torch = None

    class Dataset:  # type: ignore[no-redef]
        pass


ALL_EVALUATION_CATEGORIES = KNOWN_CODE_TYPES + NO_ECC_TYPES + (UNKNOWN_CODE_TYPE,)
TYPE_TO_INDEX = {name: index for index, name in enumerate(KNOWN_CODE_TYPES)}


@dataclass(frozen=True)
class IDSStats:
    input_bases: int
    insertions: int
    deletions: int
    substitutions: int


@dataclass(frozen=True)
class ArchiveReads:
    category: str
    split: str
    archive_id: int
    error_rate: float
    bases: np.ndarray
    features: np.ndarray
    valid_mask: np.ndarray

    @property
    def molecules(self) -> int:
        return int(self.features.shape[0])

    @property
    def reads_per_molecule(self) -> int:
        return int(self.features.shape[1])


def inject_ids(
    template: np.ndarray,
    p_ins: float,
    p_del: float,
    p_sub: float,
    rng: np.random.Generator,
    max_length: int | None = None,
) -> tuple[np.ndarray, IDSStats]:
    """Apply independent per-template-base insertion, deletion and substitution.

    Insertions are emitted immediately before their associated template base.
    An inserted base is retained even when that template base is deleted.  This
    makes the three requested event probabilities independent and measurable.
    """

    template = np.asarray(template, dtype=np.uint8).reshape(-1)
    if any(probability < 0.0 or probability > 1.0 for probability in (p_ins, p_del, p_sub)):
        raise ValueError("IDS probabilities must be in [0, 1]")

    output: list[int] = []
    insertions = deletions = substitutions = 0
    for base in template:
        if rng.random() < p_ins:
            output.append(int(rng.integers(0, 4)))
            insertions += 1
        if rng.random() < p_del:
            deletions += 1
            continue
        emitted = int(base)
        if rng.random() < p_sub:
            # Choose uniformly from the other three bases.
            emitted = (emitted + 1 + int(rng.integers(0, 3))) % 4
            substitutions += 1
        output.append(emitted)

    result = np.asarray(output, dtype=np.uint8)
    if max_length is not None:
        result = result[:max_length]
    return result, IDSStats(
        input_bases=int(template.size),
        insertions=insertions,
        deletions=deletions,
        substitutions=substitutions,
    )


def sample_noisy_read(
    reference: np.ndarray,
    error_rate: float,
    min_read_length: int,
    max_read_length: int,
    padded_length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    reference = np.asarray(reference, dtype=np.uint8).reshape(-1)
    upper = min(max_read_length, int(reference.size))
    if min_read_length > upper:
        raise ValueError("minimum read length exceeds the available reference length")
    template_length = int(rng.integers(min_read_length, upper + 1))
    start = int(rng.integers(0, reference.size - template_length + 1))
    template = reference[start : start + template_length]
    read, _ = inject_ids(
        template,
        p_ins=error_rate,
        p_del=error_rate,
        p_sub=error_rate,
        rng=rng,
        max_length=padded_length,
    )
    return read


def mask_pad_read(read: np.ndarray, padded_length: int) -> tuple[np.ndarray, np.ndarray]:
    """Return original two-bit base features [2, L] and a valid mask [L].

    The mapping is A=00, T=01, C=10 and G=11.  Padding is also all-zero,
    so the mask is the sole authority for excluding padded positions.
    """

    read = np.asarray(read, dtype=np.uint8).reshape(-1)[:padded_length]
    features = np.zeros((2, padded_length), dtype=np.uint8)
    valid_mask = np.zeros(padded_length, dtype=np.bool_)
    if read.size:
        features[0, : read.size] = (read >> 1) & 1
        features[1, : read.size] = read & 1
        valid_mask[: read.size] = True
    return features, valid_mask


class ReferenceFactory:
    """Deterministically synthesize 384-nt references for every experiment arm."""

    _ldpc_dimensions = ((32, 16), (64, 48), (128, 43), (256, 64))
    _polar_dimensions = ((32, 16), (64, 48), (128, 43), (256, 64))
    _conv_rates = ("1_2", "1_3", "1_4", "3_4")

    def __init__(self, config: ExperimentConfig):
        self.config = config
        if config.channel.reference_length * 2 != config.fountain.block_bits:
            raise ValueError("reference_length*2 must equal the LT droplet block length")

    @staticmethod
    @lru_cache(maxsize=64)
    def _ldpc_code(split: str, variant: int, n: int, k: int) -> LDPCCode:
        return LDPCCode.create(n=n, k=k, seed=stable_seed("ldpc-encoder", split, variant, n, k))

    @staticmethod
    @lru_cache(maxsize=64)
    def _polar_positions(split: str, variant: int, n: int, k: int) -> tuple[int, ...]:
        rng = np.random.default_rng(stable_seed("polar-encoder", split, variant, n, k))
        return tuple(sorted(int(value) for value in rng.choice(n, size=k, replace=False)))

    def make_reference(
        self,
        category: str,
        split: str,
        archive_id: int,
        molecule_id: int,
    ) -> np.ndarray:
        if category == UNKNOWN_CODE_TYPE:
            raise ValueError("generate fountain references with make_fountain_archive")
        if category not in KNOWN_CODE_TYPES + NO_ECC_TYPES:
            raise ValueError(f"unknown category {category}")

        target_bits = self.config.channel.reference_length * 2
        rng = np.random.default_rng(
            stable_seed("reference", split, category, archive_id, molecule_id)
        )

        if category == "BCH":
            spec = BCH_SPECS[(archive_id + molecule_id) % len(BCH_SPECS)]
            bits = concatenate_codewords(
                lambda message: bch_encode(message, spec), spec.k, target_bits, rng
            )
            return bits_to_bases(bits)

        if category == "Convolutional":
            rate = self._conv_rates[(archive_id + molecule_id) % len(self._conv_rates)]
            # A long independently generated payload avoids repeated trellis resets.
            payload_length = target_bits
            encoded = convolutional_encode(
                rng.integers(0, 2, size=payload_length, dtype=np.uint8), rate=rate
            )
            if encoded.size < target_bits:
                raise RuntimeError("convolutional encoder produced too few bits")
            start = int(rng.integers(0, encoded.size - target_bits + 1))
            return bits_to_bases(encoded[start : start + target_bits])

        if category == "LDPC":
            variant = (archive_id + molecule_id) % len(self._ldpc_dimensions)
            n, k = self._ldpc_dimensions[variant]
            code = self._ldpc_code(split, variant, n, k)
            bits = concatenate_codewords(code.encode, k, target_bits, rng)
            return bits_to_bases(bits)

        if category == "Polar":
            variant = (archive_id + molecule_id) % len(self._polar_dimensions)
            n, k = self._polar_dimensions[variant]
            information_positions = np.asarray(
                self._polar_positions(split, variant, n, k), dtype=np.int64
            )
            bits = concatenate_codewords(
                lambda message: polar_encode_with_positions(
                    message, n=n, information_positions=information_positions
                ),
                k,
                target_bits,
                rng,
            )
            return bits_to_bases(bits)

        if category == "NoECC-Random":
            return rng.integers(
                0, 4, size=self.config.channel.reference_length, dtype=np.uint8
            )

        return generate_constrained_uncoded(
            length=self.config.channel.reference_length,
            rng=rng,
            gc_min=0.45,
            gc_max=0.55,
            max_homopolymer=3,
        )

    def make_fountain_archive(
        self,
        split: str,
        archive_id: int,
        molecules: int,
    ) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
        """Make pure XOR droplets with no RS, seed header or biochemical code."""

        normalized_split = split.lower()
        if normalized_split.startswith(("train", "validation", "val", "calibration")):
            raise ValueError("fountain data are strict test-only unknowns")
        source_rng = np.random.default_rng(stable_seed("lt-source", split, archive_id))
        source_blocks = source_rng.integers(
            0,
            2,
            size=(self.config.fountain.source_blocks, self.config.fountain.block_bits),
            dtype=np.uint8,
        )
        droplets = lt_encode(
            source_blocks,
            count=molecules,
            seed=stable_seed("lt-droplets", split, archive_id),
            c=self.config.fountain.robust_soliton_c,
            delta=self.config.fountain.robust_soliton_delta,
        )
        references = np.stack([bits_to_bases(droplet.bits) for droplet in droplets])
        indices = tuple(droplet.indices for droplet in droplets)
        return references, indices


class ReadClassificationDataset(Dataset):
    """On-demand deterministic read dataset for one of the two classifiers."""

    def __init__(
        self,
        config: ExperimentConfig,
        task: str,
        split: str,
        seed: int,
        known_reads_per_type: int | None = None,
        no_ecc_reads_per_subtype: int | None = None,
    ):
        if torch is None:
            raise RuntimeError("PyTorch is required to construct a training dataset")
        if task not in {"presence", "type"}:
            raise ValueError("task must be 'presence' or 'type'")
        if split not in {"train", "validation", "val"}:
            raise ValueError("training datasets only support train/validation splits")

        self.config = config
        self.task = task
        self.split = "validation" if split == "val" else split
        self.seed = int(seed)
        self.factory = ReferenceFactory(config)

        train = self.split == "train"
        known_count = known_reads_per_type
        if known_count is None:
            known_count = (
                config.training.train_known_reads_per_type
                if train
                else config.training.val_known_reads_per_type
            )
        no_ecc_count = no_ecc_reads_per_subtype
        if no_ecc_count is None:
            no_ecc_count = (
                config.training.train_no_ecc_reads_per_subtype
                if train
                else config.training.val_no_ecc_reads_per_subtype
            )

        categories = KNOWN_CODE_TYPES if task == "type" else KNOWN_CODE_TYPES + NO_ECC_TYPES
        self._segments: list[tuple[int, int, str]] = []
        cursor = 0
        for category in categories:
            count = known_count if category in KNOWN_CODE_TYPES else no_ecc_count
            self._segments.append((cursor, cursor + int(count), category))
            cursor += int(count)
        self._length = cursor

    def __len__(self) -> int:
        return self._length

    def category_for_index(self, index: int) -> tuple[str, int]:
        if index < 0:
            index += self._length
        if index < 0 or index >= self._length:
            raise IndexError(index)
        for start, stop, category in self._segments:
            if start <= index < stop:
                return category, index - start
        raise IndexError(index)

    def __getitem__(self, index: int) -> dict[str, Any]:
        category, local_index = self.category_for_index(index)
        # The seed is part of every payload and channel key, while split-specific
        # namespaces prevent train/validation payload or encoder reuse.
        archive_id = local_index // self.config.voting.default_molecules
        molecule_id = local_index
        reference = self.factory.make_reference(
            category,
            f"{self.split}-seed-{self.seed}",
            archive_id,
            molecule_id,
        )
        rate_rng = np.random.default_rng(
            stable_seed("train-rate", self.split, self.seed, category, local_index)
        )
        # Validation follows the training channel grid; 0.15 and 0.20 remain
        # strictly held-out test severities.
        rates = self.config.channel.train_error_rates
        error_rate = float(rates[int(rate_rng.integers(0, len(rates)))])
        channel_rng = np.random.default_rng(
            stable_seed("train-read", self.split, self.seed, category, local_index)
        )
        read = sample_noisy_read(
            reference,
            error_rate,
            self.config.channel.min_read_length,
            self.config.channel.max_read_length,
            self.config.channel.padded_length,
            channel_rng,
        )
        features, valid_mask = mask_pad_read(read, self.config.channel.padded_length)
        if self.task == "presence":
            target = float(category in KNOWN_CODE_TYPES)
            target_tensor = torch.tensor(target, dtype=torch.float32)
        else:
            target_tensor = torch.tensor(TYPE_TO_INDEX[category], dtype=torch.long)
        return {
            "x": torch.from_numpy(features).to(dtype=torch.float32),
            "valid_mask": torch.from_numpy(valid_mask),
            "target": target_tensor,
        }


def generate_archive_reads(
    config: ExperimentConfig,
    category: str,
    split: str,
    archive_id: int,
    error_rate: float,
    molecules: int,
    reads_per_molecule: int,
) -> ArchiveReads:
    if category not in ALL_EVALUATION_CATEGORIES:
        raise ValueError(f"unknown evaluation category {category}")
    if molecules <= 0 or reads_per_molecule <= 0:
        raise ValueError("molecules and reads_per_molecule must be positive")

    factory = ReferenceFactory(config)
    if category == UNKNOWN_CODE_TYPE:
        references, _ = factory.make_fountain_archive(split, archive_id, molecules)
    else:
        references = np.stack(
            [
                factory.make_reference(category, split, archive_id, molecule_id)
                for molecule_id in range(molecules)
            ]
        )

    padded_length = config.channel.padded_length
    features = np.zeros((molecules, reads_per_molecule, 2, padded_length), dtype=np.uint8)
    valid_mask = np.zeros((molecules, reads_per_molecule, padded_length), dtype=np.bool_)
    bases = np.full((molecules, reads_per_molecule, padded_length), 255, dtype=np.uint8)
    for molecule_id, reference in enumerate(references):
        for read_id in range(reads_per_molecule):
            rng = np.random.default_rng(
                stable_seed(
                    "evaluation-read",
                    split,
                    category,
                    archive_id,
                    molecule_id,
                    read_id,
                    error_rate,
                )
            )
            read = sample_noisy_read(
                reference,
                error_rate,
                config.channel.min_read_length,
                config.channel.max_read_length,
                padded_length,
                rng,
            )
            encoded, mask = mask_pad_read(read, padded_length)
            features[molecule_id, read_id] = encoded
            valid_mask[molecule_id, read_id] = mask
            bases[molecule_id, read_id, : min(read.size, padded_length)] = read[:padded_length]
    return ArchiveReads(
        category=category,
        split=split,
        archive_id=archive_id,
        error_rate=float(error_rate),
        bases=bases,
        features=features,
        valid_mask=valid_mask,
    )


def categories_in_training(task: str) -> tuple[str, ...]:
    if task == "presence":
        return KNOWN_CODE_TYPES + NO_ECC_TYPES
    if task == "type":
        return KNOWN_CODE_TYPES
    raise ValueError("task must be 'presence' or 'type'")


def assert_split_isolation(
    config: ExperimentConfig,
    categories: Iterable[str] = KNOWN_CODE_TYPES + NO_ECC_TYPES,
) -> None:
    """Fail if deterministic fixtures accidentally collide across data splits."""

    factory = ReferenceFactory(config)
    for category in categories:
        train = factory.make_reference(category, "train-seed-42", 0, 0)
        validation = factory.make_reference(category, "validation-seed-42", 0, 0)
        test = factory.make_reference(category, "test-seed-42", 0, 0)
        if np.array_equal(train, validation) or np.array_equal(train, test) or np.array_equal(validation, test):
            raise AssertionError(f"reference collision across splits for {category}")
