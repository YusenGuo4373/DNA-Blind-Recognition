from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class AuthorSoftVote:
    molecule_probabilities: torch.Tensor
    archive_probabilities: torch.Tensor
    predicted_index: int
    confidence: float


def author_soft_vote(logits: torch.Tensor) -> AuthorSoftVote:
    """Author soft vote: softmax per read and arithmetic probability mean.

    The original vote script flattens ``group_size × num_copies`` reads and
    executes ``torch.softmax(logits, dim=1).mean(dim=0)``.  Keeping explicit
    [M,q,C] dimensions gives the identical result while exposing both levels.
    """

    if logits.ndim != 3 or logits.shape[0] == 0 or logits.shape[1] == 0:
        raise ValueError("logits must have non-empty shape [M, q, classes]")
    probabilities = torch.softmax(logits, dim=-1)
    molecule_probabilities = probabilities.mean(dim=1)
    archive_probabilities = molecule_probabilities.mean(dim=0)
    predicted_index = int(torch.argmax(archive_probabilities).item())
    confidence = float(archive_probabilities[predicted_index].item())
    return AuthorSoftVote(
        molecule_probabilities=molecule_probabilities,
        archive_probabilities=archive_probabilities,
        predicted_index=predicted_index,
        confidence=confidence,
    )
