from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any
import json


KNOWN_CODE_TYPES = ("BCH", "Convolutional", "LDPC", "Polar")
NO_ECC_TYPES = ("NoECC-Random", "NoECC-Constrained")
UNKNOWN_CODE_TYPE = "Fountain"


@dataclass(frozen=True)
class ChannelConfig:
    reference_length: int = 384
    min_read_length: int = 130
    max_read_length: int = 384
    padded_length: int = 400
    train_error_rates: tuple[float, ...] = (0.0, 0.01, 0.05, 0.10)
    test_error_rates: tuple[float, ...] = (0.0, 0.01, 0.05, 0.10, 0.15, 0.20)


@dataclass(frozen=True)
class VotingConfig:
    default_molecules: int = 20
    default_reads: int = 50
    molecule_sweep: tuple[int, ...] = (1, 5, 10, 20, 50)
    read_sweep: tuple[int, ...] = (1, 5, 10, 20, 50)
    known_acceptance: float = 0.95

    @property
    def max_molecules(self) -> int:
        return max(self.molecule_sweep)

    @property
    def max_reads(self) -> int:
        return max(self.read_sweep)


@dataclass(frozen=True)
class ModelConfig:
    d_model: int = 128
    nhead: int = 4
    num_layers: int = 2
    dim_feedforward: int = 256
    dropout: float = 0.10
    max_length: int = 400


@dataclass(frozen=True)
class TrainingConfig:
    seeds: tuple[int, ...] = (42, 43, 44)
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    gradient_clip: float = 1.0
    batch_size: int = 64
    num_workers: int = 0
    presence_epochs: int = 12
    type_epochs: int = 20
    early_stopping_patience: int = 3
    train_known_reads_per_type: int = 20_000
    train_no_ecc_reads_per_subtype: int = 40_000
    val_known_reads_per_type: int = 2_000
    val_no_ecc_reads_per_subtype: int = 4_000
    calibration_archives_per_category: int = 20
    test_archives_per_category: int = 50
    amp: bool = True


@dataclass(frozen=True)
class FountainConfig:
    source_blocks: int = 8
    block_bits: int = 768
    robust_soliton_c: float = 0.10
    robust_soliton_delta: float = 0.05


@dataclass(frozen=True)
class ExperimentConfig:
    channel: ChannelConfig = field(default_factory=ChannelConfig)
    voting: VotingConfig = field(default_factory=VotingConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    fountain: FountainConfig = field(default_factory=FountainConfig)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "ExperimentConfig":
        def tuples(mapping: dict[str, Any], names: tuple[str, ...]) -> dict[str, Any]:
            result = dict(mapping)
            for name in names:
                if name in result:
                    result[name] = tuple(result[name])
            return result

        channel = ChannelConfig(**tuples(values.get("channel", {}), ("train_error_rates", "test_error_rates")))
        voting = VotingConfig(**tuples(values.get("voting", {}), ("molecule_sweep", "read_sweep")))
        model = ModelConfig(**values.get("model", {}))
        training = TrainingConfig(**tuples(values.get("training", {}), ("seeds",)))
        fountain = FountainConfig(**values.get("fountain", {}))
        return cls(channel=channel, voting=voting, model=model, training=training, fountain=fountain)

    @classmethod
    def load(cls, path: str | Path) -> "ExperimentConfig":
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))
