from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn

from .soft_voting import AuthorSoftVote, author_soft_vote


@dataclass(frozen=True)
class OneHotArchive:
    """Unmodified author input contract: [M,q,4,L] plus True/1 valid mask."""

    one_hot: np.ndarray | torch.Tensor
    mask: np.ndarray | torch.Tensor

    def validate(self) -> tuple[int, int, int]:
        x_shape = tuple(self.one_hot.shape)
        mask_shape = tuple(self.mask.shape)
        if len(x_shape) != 4 or x_shape[2] != 4:
            raise ValueError("one_hot must have shape [M, q, 4, L]")
        if mask_shape != (x_shape[0], x_shape[1], x_shape[3]):
            raise ValueError("mask must have shape [M, q, L]")
        if x_shape[0] == 0 or x_shape[1] == 0 or x_shape[3] == 0:
            raise ValueError("archive dimensions must be non-empty")
        return x_shape[0], x_shape[1], x_shape[3]


@dataclass(frozen=True)
class TaskPrediction:
    task: str
    label: str
    index: int
    confidence: float
    probabilities: tuple[float, ...]
    read_logits: torch.Tensor


class OriginalTaskModel:
    def __init__(
        self,
        task: str,
        model: nn.Module,
        labels: tuple[str, ...],
        device: str | torch.device,
        batch_size: int = 256,
    ):
        if not labels:
            raise ValueError("labels must not be empty")
        self.task = task
        self.model = model.to(device).eval()
        self.labels = labels
        self.device = torch.device(device)
        self.batch_size = int(batch_size)

    @torch.inference_mode()
    def read_logits(self, archive: OneHotArchive) -> torch.Tensor:
        molecules, reads, length = archive.validate()
        x = torch.as_tensor(archive.one_hot, dtype=torch.float32).reshape(-1, 4, length)
        mask = torch.as_tensor(archive.mask).reshape(-1, length)
        outputs: list[torch.Tensor] = []
        for start in range(0, x.shape[0], self.batch_size):
            stop = min(start + self.batch_size, x.shape[0])
            logits = self.model(
                x[start:stop].to(self.device),
                mask[start:stop].to(self.device),
            )
            if logits.ndim != 2 or logits.shape[1] != len(self.labels):
                raise ValueError(
                    f"{self.task} model returned {tuple(logits.shape)}, expected [batch,{len(self.labels)}]"
                )
            outputs.append(logits.detach().cpu())
        return torch.cat(outputs, dim=0).reshape(molecules, reads, len(self.labels))

    def predict(self, archive: OneHotArchive) -> TaskPrediction:
        logits = self.read_logits(archive)
        vote = author_soft_vote(logits)
        return self.prediction_from_logits(logits, vote)

    def prediction_from_logits(
        self,
        logits: torch.Tensor,
        vote: AuthorSoftVote | None = None,
    ) -> TaskPrediction:
        vote = vote or author_soft_vote(logits)
        return TaskPrediction(
            task=self.task,
            label=self.labels[vote.predicted_index],
            index=vote.predicted_index,
            confidence=vote.confidence,
            probabilities=tuple(float(value) for value in vote.archive_probabilities.tolist()),
            read_logits=logits,
        )


class OriginalBlindRecognizer:
    """Original author classifiers kept separate from all open-set decisions."""

    def __init__(
        self,
        code_type: OriginalTaskModel,
        code_rate: OriginalTaskModel | None = None,
        code_length: OriginalTaskModel | None = None,
    ):
        self.code_type = code_type
        self.code_rate = code_rate
        self.code_length = code_length

    def predict_type(self, archive: OneHotArchive) -> TaskPrediction:
        return self.code_type.predict(archive)

    def predict_parameters(self, archive: OneHotArchive) -> dict[str, TaskPrediction | None]:
        return {
            "code_rate": None if self.code_rate is None else self.code_rate.predict(archive),
            "code_length": None if self.code_length is None else self.code_length.predict(archive),
        }

    def describe(self) -> dict[str, Any]:
        def task(model: OriginalTaskModel | None) -> dict[str, Any] | None:
            if model is None:
                return None
            return {
                "task": model.task,
                "class_name": model.model.__class__.__name__,
                "labels": list(model.labels),
                "parameters": sum(parameter.numel() for parameter in model.model.parameters()),
            }

        return {
            "code_type": task(self.code_type),
            "code_rate": task(self.code_rate),
            "code_length": task(self.code_length),
        }
