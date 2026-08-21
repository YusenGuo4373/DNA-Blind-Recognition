from __future__ import annotations

from dataclasses import asdict, dataclass
from hashlib import sha256
from pathlib import Path
import re

import torch

from .original_models import load_original_checkpoint
from .recognizer import OriginalBlindRecognizer, OriginalTaskModel


DEFAULT_WEIGHT_ROOT = Path(__file__).resolve().parents[1] / "artifacts" / "model_weights"
TYPE_LABELS = ("BCH", "Convolutional", "LDPC", "Polar")
PRIMARY_MODEL = "transformer"
PRIMARY_TYPE_FILENAME = "transformer_model_f10.6033.pt"

_FILENAME = re.compile(
    r"^(?P<model>cnn|lstm|resnet|transformer)_model_f1(?P<f1>\d+\.\d+)\.pt$"
)

EXPECTED_SHA256 = {
    "length/cnn_model_f10.3070.pt": "a4c84fd82419dc4fd80f2c5bedbffe5357ebbeb65cab38553e2510832cb9cfe4",
    "length/lstm_model_f10.4656.pt": "ce00e765fa20b54d840add91d9efff1476c13251b6e5730a602b58f85186c1fc",
    "length/resnet_model_f10.3911.pt": "eb17da64608544955391c3a4df82c02e2ae8a424497cb2a8941fad44f3ed3bd8",
    "length/transformer_model_f10.5445.pt": "773911b7df8443e486c0a950aa20a31d13cc39c03958691c649634cf451a71c1",
    "rate/cnn_model_f10.6213.pt": "db5d2be3e945c4e52c4d9b8ac7eeb778e01a3a6dfb16811a2c19d5467ba2cba6",
    "rate/lstm_model_f10.6418.pt": "abd995aab16c4151a69281a81a6e97c63f8368af3b5ee0c76d76c4e7b3ccf46a",
    "rate/resnet_model_f10.5697.pt": "baa4350e17e2ce5c8d13b9b23a4e7b066b93fa38b65b63506bb9193ad342598e",
    "rate/transformer_model_f10.6961.pt": "dfb0322a5a92a0c55cbaa5690e6903411413df4bff8f54ec222dca53ec4f54b6",
    "type/cnn_model_f10.5787.pt": "1878259380fbd0e5def01c64711d8b70de8a475cdaf329b344a17015f5a572bd",
    "type/lstm_model_f10.6397.pt": "de8c2472ad0418bc84c238d0cd2cd4cadc6ac1d1981809d6e64302583724aa3b",
    "type/resnet_model_f10.5787.pt": "40e4cbf331af2b951fdc0d5cefa7aee8e2296320628039635425d2c9cbd3b3dd",
    "type/transformer_model_f10.6033.pt": "2091d8ba0a32da526288629cb960fb2fc29314bc5090effe39c487f3599feb12",
}


@dataclass(frozen=True)
class WeightRecord:
    task: str
    model: str
    f1_from_filename: float
    path: str
    sha256: str
    hash_matches: bool
    strict_load_compatible: bool
    error: str | None = None

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def _digest(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def inspect_author_weights(
    weight_root: str | Path = DEFAULT_WEIGHT_ROOT,
    device: str | torch.device = "cpu",
) -> list[WeightRecord]:
    """Hash and strict-load every supplied checkpoint against unchanged models.py."""

    root = Path(weight_root).resolve()
    records: list[WeightRecord] = []
    for task in ("length", "rate", "type"):
        for path in sorted((root / task).glob("*.pt")):
            match = _FILENAME.fullmatch(path.name)
            if match is None:
                continue
            relative = path.relative_to(root).as_posix()
            actual_hash = _digest(path)
            compatible = False
            error: str | None = None
            try:
                model = load_original_checkpoint(
                    match.group("model"), path, num_classes=4, device=device
                )
                compatible = True
                del model
                if torch.cuda.is_available() and torch.device(device).type == "cuda":
                    torch.cuda.empty_cache()
            except Exception as exc:  # surfaced in the machine-readable audit
                error = f"{type(exc).__name__}: {exc}"
            records.append(
                WeightRecord(
                    task=task,
                    model=match.group("model"),
                    f1_from_filename=float(match.group("f1")),
                    path=str(path),
                    sha256=actual_hash,
                    hash_matches=EXPECTED_SHA256.get(relative) == actual_hash,
                    strict_load_compatible=compatible,
                    error=error,
                )
            )
    return records


def build_primary_type_recognizer(
    weight_root: str | Path = DEFAULT_WEIGHT_ROOT,
    device: str | torch.device = "cpu",
    batch_size: int = 256,
) -> OriginalBlindRecognizer:
    """Build this round's author-core recognizer; rate/length stay disconnected."""

    checkpoint = Path(weight_root).resolve() / "type" / PRIMARY_TYPE_FILENAME
    if not checkpoint.is_file():
        raise FileNotFoundError(f"primary type checkpoint not found: {checkpoint}")
    model = load_original_checkpoint(
        PRIMARY_MODEL, checkpoint, num_classes=len(TYPE_LABELS), device=device
    )
    type_task = OriginalTaskModel(
        task="code_type",
        model=model,
        labels=TYPE_LABELS,
        device=device,
        batch_size=batch_size,
    )
    return OriginalBlindRecognizer(code_type=type_task, code_rate=None, code_length=None)
