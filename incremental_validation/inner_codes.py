from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import subprocess

import numpy as np

from author_baseline.recognizer import OneHotArchive
from hierarchical_ecc.coding import stable_seed
from hierarchical_ecc.config import ExperimentConfig
from hierarchical_ecc.data import sample_noisy_read

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
ADAPTER_BUILD = WORKSPACE_ROOT / "build" / "inner_code_adapters"
HEDGES_EXECUTABLE = ADAPTER_BUILD / "hedges_encode.exe"
DNA_AEON_EXECUTABLE = ADAPTER_BUILD / "dna_aeon_encode.exe"
DNA_AEON_ROOT = WORKSPACE_ROOT / "vendor" / "MW55_DNA-Aeon_snapshot" / "DNA-Aeon-main"
DNA_AEON_CODEBOOK = DNA_AEON_ROOT / "codewords" / "cw_40_60_hp3.fasta"
DNA_AEON_MOTIFS = DNA_AEON_ROOT / "codewords" / "cw_40_60_hp3.json"
UNKNOWN_INNER_CODES = ("HEDGES", "DNA-Aeon")


def _bases_to_one_hot(bases: np.ndarray, valid_mask: np.ndarray) -> np.ndarray:
    safe = np.where(valid_mask, bases, 0).astype(np.int64)
    encoded = np.eye(4, dtype=np.float32)[safe]
    encoded *= valid_mask[..., None]
    return np.moveaxis(encoded, -1, -2)


@dataclass(frozen=True)
class InnerCodeValidation:
    category: str
    count: int
    length: int
    gc_min: float
    gc_max: float
    max_homopolymer: int
    homopolymer_limit: int
    homopolymer_violations: int
    unique_count: int
    duplicate_count: int
    all_bases_legal: bool
    fixed_primer_prefix_matches: int
    fixed_primer_suffix_matches: int


def _parse_fasta(path: Path) -> list[str]:
    records: list[str] = []
    current: list[str] = []
    for line in path.read_text(encoding="ascii").splitlines():
        line = line.strip().upper()
        if not line:
            continue
        if line.startswith(">"):
            if current:
                records.append("".join(current))
                current = []
        else:
            current.append(line)
    if current:
        records.append("".join(current))
    return records


def _max_homopolymer(sequence: str) -> int:
    best = run = 0
    previous = ""
    for base in sequence:
        run = run + 1 if base == previous else 1
        previous = base
        best = max(best, run)
    return best


def generate_inner_code_references(
    category: str,
    count: int,
    seed: int,
    output_fasta: str | Path,
    namespace: str = "test",
) -> tuple[np.ndarray, InnerCodeValidation]:
    """Generate 384-nt references using the official inner-code executables."""

    if category not in UNKNOWN_INNER_CODES:
        raise ValueError(f"unsupported unknown inner code: {category}")
    if count <= 0:
        raise ValueError("count must be positive")
    output_fasta = Path(output_fasta).resolve()
    output_fasta.parent.mkdir(parents=True, exist_ok=True)
    if not namespace:
        raise ValueError("namespace must be non-empty")
    encoder_seed = stable_seed("official-inner-code", namespace, category, int(seed))
    if category == "HEDGES":
        executable = HEDGES_EXECUTABLE
        command = [str(executable), str(count), str(encoder_seed), str(output_fasta)]
        cwd = WORKSPACE_ROOT / "vendor" / "whpress_HEDGES"
    else:
        executable = DNA_AEON_EXECUTABLE
        command = [
            str(executable),
            str(count),
            str(encoder_seed),
            str(DNA_AEON_CODEBOOK),
            str(DNA_AEON_MOTIFS),
            str(output_fasta),
        ]
        cwd = WORKSPACE_ROOT
    if not executable.is_file():
        raise FileNotFoundError(f"official inner-code adapter is not built: {executable}")
    subprocess.run(command, cwd=str(cwd), check=True)
    sequences = _parse_fasta(output_fasta)
    if len(sequences) != count:
        raise RuntimeError(f"{category} generated {len(sequences)} sequences, expected {count}")
    if any(len(sequence) != 384 for sequence in sequences):
        raise RuntimeError(f"{category} generated a non-384-nt sequence")
    all_bases_legal = not any(set(sequence) - set("ACGT") for sequence in sequences)
    if not all_bases_legal:
        raise RuntimeError(f"{category} generated a non-DNA symbol")
    unique_count = len(set(sequences))
    if unique_count != count:
        raise RuntimeError(f"{category} generated duplicate molecular references")
    mapping = {"A": 0, "C": 1, "G": 2, "T": 3}
    references = np.asarray(
        [[mapping[base] for base in sequence] for sequence in sequences], dtype=np.uint8
    )
    gc = [sum(base in "GC" for base in sequence) / len(sequence) for sequence in sequences]
    maximum_runs = [_max_homopolymer(sequence) for sequence in sequences]
    homopolymer_limit = 4 if category == "HEDGES" else 3
    homopolymer_violations = sum(value > homopolymer_limit for value in maximum_runs)
    if homopolymer_violations:
        raise RuntimeError(
            f"{category} has {homopolymer_violations} homopolymer-limit violations"
        )
    hedges_left = "TCGAAGTCAGCGTGTATTGTATG"
    hedges_right = "TAGTGAGTGCGATTAAGCGTGTT"
    prefix_matches = sum(sequence.startswith(hedges_left) for sequence in sequences)
    suffix_matches = sum(sequence.endswith(hedges_right) for sequence in sequences)
    if category == "HEDGES" and (prefix_matches or suffix_matches):
        raise RuntimeError("pure HEDGES inner-code output still contains fixed flanking primers")
    validation = InnerCodeValidation(
        category=category,
        count=count,
        length=384,
        gc_min=float(min(gc)),
        gc_max=float(max(gc)),
        max_homopolymer=max(maximum_runs),
        homopolymer_limit=homopolymer_limit,
        homopolymer_violations=homopolymer_violations,
        unique_count=unique_count,
        duplicate_count=count - unique_count,
        all_bases_legal=all_bases_legal,
        fixed_primer_prefix_matches=prefix_matches,
        fixed_primer_suffix_matches=suffix_matches,
    )
    return references, validation


def archives_from_references(
    config: ExperimentConfig,
    category: str,
    split: str,
    references: np.ndarray,
    archives: int,
    molecules: int,
    reads_per_molecule: int,
    error_rate: float,
) -> list[OneHotArchive]:
    references = np.asarray(references, dtype=np.uint8)
    if references.shape != (archives * molecules, config.channel.reference_length):
        raise ValueError("references must have shape [archives*M,384]")
    result: list[OneHotArchive] = []
    padded_length = config.channel.padded_length
    for archive_id in range(archives):
        bases = np.full((molecules, reads_per_molecule, padded_length), 255, dtype=np.uint8)
        mask = np.zeros((molecules, reads_per_molecule, padded_length), dtype=np.bool_)
        for molecule_id in range(molecules):
            reference = references[archive_id * molecules + molecule_id]
            for read_id in range(reads_per_molecule):
                rng = np.random.default_rng(
                    stable_seed(
                        "official-inner-read",
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
                valid = min(read.size, padded_length)
                bases[molecule_id, read_id, :valid] = read[:valid]
                mask[molecule_id, read_id, :valid] = True
        result.append(OneHotArchive(_bases_to_one_hot(bases, mask), mask))
    return result
