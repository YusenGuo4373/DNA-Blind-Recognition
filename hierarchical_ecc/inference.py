from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

from .data import ArchiveReads
from .models import DNAReadTransformer
from .voting import HierarchicalDecision, Thresholds, hierarchical_decision, two_level_soft_vote


@torch.inference_mode()
def predict_archive_reads(
    model: DNAReadTransformer,
    archive: ArchiveReads,
    device: str | torch.device,
    batch_size: int = 256,
) -> np.ndarray:
    """Run a model independently on each read and restore [M, q, C]."""

    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    device = torch.device(device)
    model.eval()
    model.to(device)
    molecules, reads = archive.features.shape[:2]
    flat_features = archive.features.reshape(-1, 2, archive.features.shape[-1])
    flat_masks = archive.valid_mask.reshape(-1, archive.valid_mask.shape[-1])
    outputs: list[np.ndarray] = []
    for start in range(0, flat_features.shape[0], batch_size):
        stop = min(start + batch_size, flat_features.shape[0])
        x = torch.from_numpy(flat_features[start:stop]).to(
            device=device, dtype=torch.float32, non_blocking=True
        )
        valid_mask = torch.from_numpy(flat_masks[start:stop]).to(
            device=device, dtype=torch.bool, non_blocking=True
        )
        outputs.append(model(x, valid_mask).detach().cpu().numpy())
    values = np.concatenate(outputs, axis=0)
    return values.reshape(molecules, reads, -1)


@dataclass(frozen=True)
class ArchiveModelScores:
    presence_probabilities: np.ndarray
    type_logits: np.ndarray | None

    def prefix(self, molecules: int, reads: int) -> "ArchiveModelScores":
        if molecules <= 0 or reads <= 0:
            raise ValueError("prefix sizes must be positive")
        if molecules > self.presence_probabilities.shape[0] or reads > self.presence_probabilities.shape[1]:
            raise ValueError("requested prefix exceeds scored archive")
        logits = None
        if self.type_logits is not None:
            logits = self.type_logits[:molecules, :reads]
        return ArchiveModelScores(
            presence_probabilities=self.presence_probabilities[:molecules, :reads],
            type_logits=logits,
        )


def score_archive(
    presence_model: DNAReadTransformer,
    archive: ArchiveReads,
    device: str | torch.device,
    batch_size: int = 256,
    type_model: DNAReadTransformer | None = None,
) -> ArchiveModelScores:
    presence_logits = predict_archive_reads(
        presence_model, archive, device=device, batch_size=batch_size
    )[..., 0]
    presence_probabilities = 1.0 / (1.0 + np.exp(-np.clip(presence_logits, -80.0, 80.0)))
    type_logits = None
    if type_model is not None:
        type_logits = predict_archive_reads(
            type_model, archive, device=device, batch_size=batch_size
        )
    return ArchiveModelScores(presence_probabilities, type_logits)


def infer_archive(
    presence_model: DNAReadTransformer,
    type_model: DNAReadTransformer,
    archive: ArchiveReads,
    thresholds: Thresholds,
    device: str | torch.device,
    batch_size: int = 256,
) -> HierarchicalDecision:
    """Strict cascade: do not invoke the type model when stage one says no ECC."""

    presence_scores = score_archive(
        presence_model,
        archive,
        device=device,
        batch_size=batch_size,
        type_model=None,
    )
    if float(two_level_soft_vote(presence_scores.presence_probabilities)) < thresholds.presence:
        return hierarchical_decision(
            presence_scores.presence_probabilities,
            type_logits=None,
            thresholds=thresholds,
        )
    type_logits = predict_archive_reads(type_model, archive, device=device, batch_size=batch_size)
    return hierarchical_decision(
        presence_scores.presence_probabilities,
        type_logits=type_logits,
        thresholds=thresholds,
    )
