from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from author_baseline.recognizer import OneHotArchive, OriginalTaskModel


class TorchPresenceDetector:
    """External 4-channel No-ECC detector; not part of the author recognizer."""

    def __init__(
        self,
        model: nn.Module,
        device: str | torch.device,
        batch_size: int = 256,
        ecc_class_index: int = 1,
    ):
        self.model = model.to(device).eval()
        self.device = torch.device(device)
        self.batch_size = int(batch_size)
        self.ecc_class_index = int(ecc_class_index)

    @torch.inference_mode()
    def predict_probabilities(self, archive: OneHotArchive) -> np.ndarray:
        molecules, reads, length = archive.validate()
        x = torch.as_tensor(archive.one_hot, dtype=torch.float32).reshape(-1, 4, length)
        mask = torch.as_tensor(archive.mask).reshape(-1, length)
        probabilities: list[torch.Tensor] = []
        for start in range(0, x.shape[0], self.batch_size):
            stop = min(start + self.batch_size, x.shape[0])
            logits = self.model(
                x[start:stop].to(self.device), mask[start:stop].to(self.device)
            )
            if logits.ndim == 1:
                logits = logits.unsqueeze(-1)
            if logits.ndim != 2:
                raise ValueError("presence model must return [batch], [batch,1], or [batch,classes]")
            if logits.shape[1] == 1:
                values = torch.sigmoid(logits[:, 0])
            else:
                if not 0 <= self.ecc_class_index < logits.shape[1]:
                    raise ValueError("ecc_class_index is outside presence model outputs")
                values = torch.softmax(logits, dim=1)[:, self.ecc_class_index]
            probabilities.append(values.detach().cpu())
        return torch.cat(probabilities).reshape(molecules, reads).numpy()


@dataclass(frozen=True)
class SharedLogitDataset:
    categories: np.ndarray
    presence_probabilities: np.ndarray
    type_logits: np.ndarray
    archive_ids: np.ndarray

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            categories=self.categories.astype(str),
            presence_probabilities=self.presence_probabilities.astype(np.float32),
            type_logits=self.type_logits.astype(np.float32),
            archive_ids=self.archive_ids.astype(str),
        )


def collect_shared_logits(
    archives: Sequence[OneHotArchive],
    categories: Sequence[str],
    archive_ids: Sequence[str | int],
    presence_detector,
    author_type_model: OriginalTaskModel,
) -> SharedLogitDataset:
    """Score each archive once and persist the exact tensors used by both flows."""

    if len(archives) == 0 or not (len(archives) == len(categories) == len(archive_ids)):
        raise ValueError("archives, categories, and archive_ids must be non-empty and aligned")
    expected_shape = archives[0].validate()[:2]
    presence_values: list[np.ndarray] = []
    type_values: list[np.ndarray] = []
    for archive in archives:
        if archive.validate()[:2] != expected_shape:
            raise ValueError("all archives in one shared-logit NPZ must use the same M and q")
        presence = np.asarray(presence_detector.predict_probabilities(archive), dtype=np.float64)
        if presence.shape != expected_shape:
            raise ValueError("presence detector returned a shape other than [M,q]")
        # This is the sole author-model call. Both downstream flows reuse it.
        logits = author_type_model.read_logits(archive).numpy()
        presence_values.append(presence)
        type_values.append(logits)
    return SharedLogitDataset(
        categories=np.asarray(categories, dtype=str),
        presence_probabilities=np.stack(presence_values),
        type_logits=np.stack(type_values),
        archive_ids=np.asarray(archive_ids, dtype=str),
    )
