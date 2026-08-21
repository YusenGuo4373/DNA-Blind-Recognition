from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Protocol

import numpy as np
import torch

from .recognizer import OneHotArchive, OriginalBlindRecognizer
from .soft_voting import author_soft_vote


class PresenceDetector(Protocol):
    """External component; never part of the author blind recognizer."""

    def predict_probabilities(self, archive: OneHotArchive) -> np.ndarray:
        """Return per-read ECC probabilities with shape [M,q]."""


@dataclass(frozen=True)
class CascadeThresholds:
    ecc_presence: float
    unknown_energy: float
    energy_temperature: float = 1.0


@dataclass(frozen=True)
class CascadeDecision:
    status: str
    code_type: str | None
    code_rate: str | None
    code_length: str | None
    ecc_score: float
    unknown_score: float | None
    type_probabilities: tuple[float, ...] | None
    type_confidence: float | None
    max_logit: float | None
    q: int
    M: int
    thresholds: CascadeThresholds

    def to_dict(self) -> dict[str, object]:
        result = asdict(self)
        result["thresholds"] = asdict(self.thresholds)
        return result


def mean_energy(logits: torch.Tensor, temperature: float = 1.0) -> float:
    if temperature <= 0:
        raise ValueError("energy_temperature must be positive")
    energies = -temperature * torch.logsumexp(logits / temperature, dim=-1)
    # Explicit q then M averaging; identical sizes make this the author-style mean.
    return float(energies.mean(dim=1).mean(dim=0).item())


class HierarchicalAuthorAdapter:
    """Place no-ECC and unknown-ECC gates around the untouched author model."""

    def __init__(
        self,
        presence_detector: PresenceDetector,
        recognizer: OriginalBlindRecognizer,
        thresholds: CascadeThresholds,
    ):
        self.presence_detector = presence_detector
        self.recognizer = recognizer
        self.thresholds = thresholds

    def predict(self, archive: OneHotArchive) -> CascadeDecision:
        molecules, reads, _ = archive.validate()
        probabilities = np.asarray(
            self.presence_detector.predict_probabilities(archive), dtype=np.float64
        )
        if probabilities.shape != (molecules, reads):
            raise ValueError("external presence detector must return [M,q] probabilities")
        ecc_score = float(probabilities.mean(axis=1).mean(axis=0))
        if ecc_score < self.thresholds.ecc_presence:
            return self._decision("no_ecc", ecc_score, molecules, reads)

        # The first call into the author core happens only after the external ECC gate.
        type_logits = self.recognizer.code_type.read_logits(archive)
        unknown_score = mean_energy(type_logits, self.thresholds.energy_temperature)
        vote = author_soft_vote(type_logits)
        if unknown_score > self.thresholds.unknown_energy:
            return self._decision(
                "unknown_ecc",
                ecc_score,
                molecules,
                reads,
                unknown_score=unknown_score,
                type_probabilities=tuple(float(value) for value in vote.archive_probabilities.tolist()),
                type_confidence=vote.confidence,
                max_logit=float(type_logits.mean(dim=1).mean(dim=0).max().item()),
            )

        type_prediction = self.recognizer.code_type.prediction_from_logits(type_logits, vote)
        parameters = self.recognizer.predict_parameters(archive)
        rate_prediction = parameters["code_rate"]
        length_prediction = parameters["code_length"]
        return self._decision(
            "known_ecc",
            ecc_score,
            molecules,
            reads,
            code_type=type_prediction.label,
            code_rate=None if rate_prediction is None else rate_prediction.label,
            code_length=None if length_prediction is None else length_prediction.label,
            unknown_score=unknown_score,
            type_probabilities=type_prediction.probabilities,
            type_confidence=type_prediction.confidence,
            max_logit=float(type_logits.mean(dim=1).mean(dim=0).max().item()),
        )

    def _decision(
        self,
        status: str,
        ecc_score: float,
        molecules: int,
        reads: int,
        code_type: str | None = None,
        code_rate: str | None = None,
        code_length: str | None = None,
        unknown_score: float | None = None,
        type_probabilities: tuple[float, ...] | None = None,
        type_confidence: float | None = None,
        max_logit: float | None = None,
    ) -> CascadeDecision:
        return CascadeDecision(
            status=status,
            code_type=code_type,
            code_rate=code_rate,
            code_length=code_length,
            ecc_score=ecc_score,
            unknown_score=unknown_score,
            type_probabilities=type_probabilities,
            type_confidence=type_confidence,
            max_logit=max_logit,
            q=reads,
            M=molecules,
            thresholds=self.thresholds,
        )
