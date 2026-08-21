from __future__ import annotations

"""Lightweight, process-safe Stage-7 channel generation.

This module deliberately imports neither PyTorch nor the model stack.  Windows
worker processes can therefore generate deterministic IDS reads without each
loading a copy of the frozen neural networks.
"""

from typing import Any

import numpy as np

from hierarchical_ecc.coding import stable_seed


def inject_ids_with_stats(
    template: np.ndarray,
    error_rate: float,
    rng: np.random.Generator,
    max_length: int,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    """Bit-for-bit equivalent to the repository's scalar IDS channel plus stats."""

    values = np.asarray(template, dtype=np.uint8).reshape(-1)
    output: list[int] = []
    insertions = deletions = substitutions = 0
    for base in values:
        if rng.random() < error_rate:
            output.append(int(rng.integers(0, 4)))
            insertions += 1
        if rng.random() < error_rate:
            deletions += 1
            continue
        emitted = int(base)
        if rng.random() < error_rate:
            emitted = (emitted + 1 + int(rng.integers(0, 3))) % 4
            substitutions += 1
        output.append(emitted)
    read = np.asarray(output, dtype=np.uint8)[: int(max_length)]
    return read, (int(values.size), insertions, deletions, substitutions)


def sample_noisy_read_with_stats(
    reference: np.ndarray,
    error_rate: float,
    min_read_length: int,
    max_read_length: int,
    padded_length: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    values = np.asarray(reference, dtype=np.uint8).reshape(-1)
    upper = min(int(max_read_length), int(values.size))
    if int(min_read_length) > upper:
        raise ValueError("minimum read length exceeds the available reference length")
    template_length = int(rng.integers(int(min_read_length), upper + 1))
    start = int(rng.integers(0, values.size - template_length + 1))
    return inject_ids_with_stats(
        values[start : start + template_length],
        float(error_rate),
        rng,
        int(padded_length),
    )


def generate_archive_payload(spec: dict[str, Any]) -> dict[str, np.ndarray]:
    """Generate one archive using the frozen Stage-5/6 channel semantics."""

    references = np.asarray(spec["references"], dtype=np.uint8)
    molecules = int(spec["molecules"])
    reads = int(spec["reads_per_molecule"])
    padded_length = int(spec["padded_length"])
    if references.shape != (molecules, int(spec["reference_length"])):
        raise ValueError("reference payload does not match [M,reference_length]")
    bases = np.zeros((molecules, reads, padded_length), dtype=np.uint8)
    mask = np.zeros((molecules, reads, padded_length), dtype=np.bool_)
    lengths = np.zeros((molecules, reads), dtype=np.int16)
    template_lengths = np.zeros((molecules, reads), dtype=np.int16)
    insertions = np.zeros((molecules, reads), dtype=np.int16)
    deletions = np.zeros((molecules, reads), dtype=np.int16)
    substitutions = np.zeros((molecules, reads), dtype=np.int16)
    noise_category = str(spec["noise_category"])
    split = str(spec["split"])
    archive_index = int(spec["archive_index"])
    error_rate = float(spec["error_rate"])
    for molecule_index, reference in enumerate(references):
        for read_index in range(reads):
            rng = np.random.default_rng(
                stable_seed(
                    "official-inner-read",
                    split,
                    noise_category,
                    archive_index,
                    molecule_index,
                    read_index,
                    error_rate,
                )
            )
            read, stats = sample_noisy_read_with_stats(
                reference,
                error_rate,
                int(spec["min_read_length"]),
                int(spec["max_read_length"]),
                padded_length,
                rng,
            )
            valid = min(read.size, padded_length)
            bases[molecule_index, read_index, :valid] = read[:valid]
            mask[molecule_index, read_index, :valid] = True
            lengths[molecule_index, read_index] = valid
            template_lengths[molecule_index, read_index] = stats[0]
            insertions[molecule_index, read_index] = stats[1]
            deletions[molecule_index, read_index] = stats[2]
            substitutions[molecule_index, read_index] = stats[3]
    return {
        "bases": bases,
        "mask": mask,
        "read_lengths": lengths,
        "template_lengths": template_lengths,
        "insertions": insertions,
        "deletions": deletions,
        "substitutions": substitutions,
    }
