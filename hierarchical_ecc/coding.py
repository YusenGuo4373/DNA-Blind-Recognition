from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from typing import Iterable, Sequence

import numpy as np


BASE_NAMES = ("A", "T", "C", "G")
KNOWN_TYPE_TO_ID = {"BCH": 0, "Convolutional": 1, "LDPC": 2, "Polar": 3}


def stable_seed(*parts: object, bits: int = 64) -> int:
    """Return a process-independent integer seed.

    Python's built-in ``hash()`` is deliberately randomized between processes,
    so every stochastic component in this package uses this helper instead.
    """

    payload = "\x1f".join(str(part) for part in parts).encode("utf-8")
    digest = hashlib.sha256(payload).digest()
    return int.from_bytes(digest[: bits // 8], "little", signed=False)


def bits_to_bases(bits: np.ndarray) -> np.ndarray:
    bits = np.asarray(bits, dtype=np.uint8).reshape(-1)
    if bits.size % 2:
        bits = np.pad(bits, (0, 1))
    pairs = bits.reshape(-1, 2)
    return ((pairs[:, 0] << 1) | pairs[:, 1]).astype(np.uint8)


def bases_to_bits(bases: np.ndarray) -> np.ndarray:
    bases = np.asarray(bases, dtype=np.uint8).reshape(-1)
    bits = np.empty(bases.size * 2, dtype=np.uint8)
    bits[0::2] = (bases >> 1) & 1
    bits[1::2] = bases & 1
    return bits


def gf2_rank(matrix: np.ndarray) -> int:
    a = np.asarray(matrix, dtype=np.uint8).copy() & 1
    rows, cols = a.shape
    rank = 0
    for col in range(cols):
        pivots = np.flatnonzero(a[rank:, col])
        if pivots.size == 0:
            continue
        pivot = rank + int(pivots[0])
        if pivot != rank:
            a[[rank, pivot]] = a[[pivot, rank]]
        for row in range(rows):
            if row != rank and a[row, col]:
                a[row] ^= a[rank]
        rank += 1
        if rank == rows:
            break
    return rank


def _poly_degree(value: int) -> int:
    return value.bit_length() - 1


def polynomial_remainder(dividend: int, divisor: int) -> int:
    if divisor <= 0:
        raise ValueError("divisor must be a non-zero binary polynomial")
    divisor_degree = _poly_degree(divisor)
    remainder = int(dividend)
    while remainder and _poly_degree(remainder) >= divisor_degree:
        remainder ^= divisor << (_poly_degree(remainder) - divisor_degree)
    return remainder


@dataclass(frozen=True)
class BCHSpec:
    n: int
    k: int
    generator: int

    def validate(self) -> None:
        degree = _poly_degree(self.generator)
        if degree != self.n - self.k:
            raise ValueError(f"BCH generator degree {degree} != n-k {self.n - self.k}")
        # In GF(2), x^n - 1 equals x^n + 1.
        if polynomial_remainder((1 << self.n) | 1, self.generator) != 0:
            raise ValueError("generator does not divide x^n + 1")


# Valid primitive/narrow-sense binary BCH generator polynomials.  These are
# checked programmatically by BCHSpec.validate() before use.
BCH_SPECS = (
    BCHSpec(7, 4, 0b1011),
    BCHSpec(15, 11, 0b10011),
    BCHSpec(15, 7, 0b111010001),
    BCHSpec(31, 26, 0b100101),
)


def bch_encode(message: np.ndarray, spec: BCHSpec) -> np.ndarray:
    spec.validate()
    message = np.asarray(message, dtype=np.uint8).reshape(-1)
    if message.size != spec.k:
        raise ValueError(f"expected {spec.k} message bits, got {message.size}")
    message_int = 0
    for bit in message:
        message_int = (message_int << 1) | int(bit & 1)
    shifted = message_int << (spec.n - spec.k)
    codeword_int = shifted ^ polynomial_remainder(shifted, spec.generator)
    return np.array(
        [(codeword_int >> shift) & 1 for shift in range(spec.n - 1, -1, -1)],
        dtype=np.uint8,
    )


@dataclass(frozen=True)
class LDPCCode:
    n: int
    k: int
    G: np.ndarray
    H: np.ndarray

    @classmethod
    def create(cls, n: int, k: int, seed: int, column_weight: int = 3) -> "LDPCCode":
        if not 0 < k < n:
            raise ValueError("LDPC dimensions require 0 < k < n")
        m = n - k
        rng = np.random.default_rng(seed)
        a = np.zeros((m, k), dtype=np.uint8)
        for col in range(k):
            weight = min(max(1, column_weight + int(rng.integers(-1, 2))), m)
            rows = rng.choice(m, size=weight, replace=False)
            a[rows, col] = 1
        h = np.concatenate((a, np.eye(m, dtype=np.uint8)), axis=1)
        g = np.concatenate((np.eye(k, dtype=np.uint8), a.T), axis=1)
        code = cls(n=n, k=k, G=g, H=h)
        code.validate()
        return code

    def validate(self) -> None:
        if gf2_rank(self.H) != self.n - self.k:
            raise ValueError("LDPC parity-check matrix is not full row rank over GF(2)")
        if gf2_rank(self.G) != self.k:
            raise ValueError("LDPC generator matrix is not full row rank over GF(2)")
        if np.any((self.G @ self.H.T) & 1):
            raise ValueError("LDPC G H^T != 0 over GF(2)")

    def encode(self, message: np.ndarray) -> np.ndarray:
        message = np.asarray(message, dtype=np.uint8).reshape(-1)
        if message.size != self.k:
            raise ValueError(f"expected {self.k} message bits, got {message.size}")
        return ((message @ self.G) & 1).astype(np.uint8)


CONV_GENERATORS = {
    "1_2": (0o17, 0o15),
    "1_3": (0o17, 0o15, 0o13),
    "1_4": (0o17, 0o15, 0o13, 0o11),
}
CONV_PUNCTURE_3_4 = np.array(((1, 1, 0), (1, 0, 1)), dtype=np.uint8)


def convolutional_encode(
    message: np.ndarray,
    rate: str,
    constraint_length: int = 4,
) -> np.ndarray:
    message = np.asarray(message, dtype=np.uint8).reshape(-1)
    if rate == "3_4":
        generators = CONV_GENERATORS["1_2"]
        puncture = CONV_PUNCTURE_3_4
    elif rate in CONV_GENERATORS:
        generators = CONV_GENERATORS[rate]
        puncture = None
    else:
        raise ValueError(f"unsupported convolutional rate {rate}")

    masks = np.array(
        [[(generator >> bit) & 1 for bit in range(constraint_length)] for generator in generators],
        dtype=np.uint8,
    )
    state = np.zeros(constraint_length, dtype=np.uint8)
    output: list[int] = []
    for time_index, bit in enumerate(message):
        state[1:] = state[:-1]
        state[0] = bit & 1
        encoded = ((masks @ state) & 1).astype(np.uint8)
        if puncture is None:
            output.extend(int(value) for value in encoded)
        else:
            column = time_index % puncture.shape[1]
            output.extend(int(encoded[row]) for row in range(encoded.size) if puncture[row, column])
    return np.asarray(output, dtype=np.uint8)


def polar_bec_reliabilities(n: int, erasure_probability: float) -> np.ndarray:
    if n <= 0 or n & (n - 1):
        raise ValueError("Polar length must be a positive power of two")
    z = np.array([float(erasure_probability)], dtype=np.float64)
    while z.size < n:
        upper = 2.0 * z - z * z
        lower = z * z
        z = np.stack((upper, lower), axis=1).reshape(-1)
    return z


def polar_transform(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=np.uint8).copy().reshape(-1)
    if x.size <= 0 or x.size & (x.size - 1):
        raise ValueError("Polar transform length must be a power of two")
    step = 1
    while step < x.size:
        block = step * 2
        for start in range(0, x.size, block):
            x[start : start + step] ^= x[start + step : start + block]
        step = block
    return x


def polar_encode(message: np.ndarray, n: int, design_p: float = 0.5) -> np.ndarray:
    message = np.asarray(message, dtype=np.uint8).reshape(-1)
    if message.size > n:
        raise ValueError("Polar message is longer than the codeword")
    reliability = polar_bec_reliabilities(n, design_p)
    information_positions = np.sort(np.argsort(reliability)[: message.size])
    u = np.zeros(n, dtype=np.uint8)
    u[information_positions] = message
    return polar_transform(u)


def polar_encode_with_positions(
    message: np.ndarray,
    n: int,
    information_positions: np.ndarray,
) -> np.ndarray:
    """Encode a Polar subcode with an explicit, encoder-seeded frozen set."""

    message = np.asarray(message, dtype=np.uint8).reshape(-1)
    positions = np.asarray(information_positions, dtype=np.int64).reshape(-1)
    if message.size != positions.size:
        raise ValueError("message and information_positions sizes must match")
    if n <= 0 or n & (n - 1):
        raise ValueError("Polar length must be a positive power of two")
    if np.unique(positions).size != positions.size or np.any(positions < 0) or np.any(positions >= n):
        raise ValueError("information_positions must be unique indices in [0, n)")
    u = np.zeros(n, dtype=np.uint8)
    u[positions] = message
    return polar_transform(u)


def robust_soliton_distribution(k: int, c: float = 0.1, delta: float = 0.05) -> np.ndarray:
    if k < 2:
        return np.array([1.0], dtype=np.float64)
    degrees = np.arange(1, k + 1, dtype=np.float64)
    rho = np.empty(k, dtype=np.float64)
    rho[0] = 1.0 / k
    rho[1:] = 1.0 / (degrees[1:] * (degrees[1:] - 1.0))

    r = c * math.log(k / delta) * math.sqrt(k)
    pivot = min(k, max(1, int(math.floor(k / r))))
    tau = np.zeros(k, dtype=np.float64)
    for degree in range(1, pivot):
        tau[degree - 1] = r / (degree * k)
    tau[pivot - 1] = r * math.log(r / delta) / k
    probabilities = rho + tau
    probabilities /= probabilities.sum()
    return probabilities


@dataclass(frozen=True)
class LTDroplet:
    indices: tuple[int, ...]
    bits: np.ndarray


def lt_encode(
    source_blocks: np.ndarray,
    count: int,
    seed: int,
    c: float = 0.1,
    delta: float = 0.05,
) -> list[LTDroplet]:
    blocks = np.asarray(source_blocks, dtype=np.uint8)
    if blocks.ndim != 2:
        raise ValueError("source_blocks must have shape [K, block_bits]")
    if count <= 0:
        raise ValueError("count must be positive")
    rng = np.random.default_rng(seed)
    probabilities = robust_soliton_distribution(blocks.shape[0], c=c, delta=delta)
    droplets: list[LTDroplet] = []
    for _ in range(count):
        degree = int(rng.choice(np.arange(1, blocks.shape[0] + 1), p=probabilities))
        indices = tuple(sorted(int(index) for index in rng.choice(blocks.shape[0], size=degree, replace=False)))
        bits = np.bitwise_xor.reduce(blocks[np.asarray(indices)], axis=0).astype(np.uint8)
        droplets.append(LTDroplet(indices=indices, bits=bits))
    return droplets


def concatenate_codewords(
    encoder,
    message_length: int,
    target_bits: int,
    rng: np.random.Generator,
) -> np.ndarray:
    pieces: list[np.ndarray] = []
    total = 0
    while total < target_bits * 2:
        message = rng.integers(0, 2, size=message_length, dtype=np.uint8)
        codeword = np.asarray(encoder(message), dtype=np.uint8).reshape(-1)
        if codeword.size == 0:
            raise ValueError("encoder returned an empty codeword")
        pieces.append(codeword)
        total += codeword.size
    stream = np.concatenate(pieces)
    start = int(rng.integers(0, stream.size - target_bits + 1))
    return stream[start : start + target_bits]


def max_homopolymer_length(bases: Sequence[int]) -> int:
    values = np.asarray(bases).reshape(-1)
    if values.size == 0:
        return 0
    maximum = run = 1
    for previous, current in zip(values[:-1], values[1:]):
        if current == previous:
            run += 1
            maximum = max(maximum, run)
        else:
            run = 1
    return maximum


def generate_constrained_uncoded(
    length: int,
    rng: np.random.Generator,
    gc_min: float = 0.45,
    gc_max: float = 0.55,
    max_homopolymer: int = 3,
    attempts: int = 20_000,
) -> np.ndarray:
    for _ in range(attempts):
        bases = rng.integers(0, 4, size=length, dtype=np.uint8)
        gc_fraction = float(np.mean((bases == 2) | (bases == 3)))
        if gc_min <= gc_fraction <= gc_max and max_homopolymer_length(bases) <= max_homopolymer:
            return bases
    raise RuntimeError("could not generate a constrained uncoded sequence")
