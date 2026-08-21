from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .config import ModelConfig


class DNAReadTransformer(nn.Module):
    """Original-style per-read Transformer with padding-aware mean pooling."""

    def __init__(self, config: ModelConfig, output_dim: int):
        super().__init__()
        if output_dim <= 0:
            raise ValueError("output_dim must be positive")
        self.config = config
        self.output_dim = int(output_dim)
        self.input_projection = nn.Linear(2, config.d_model)
        self.position_embedding = nn.Embedding(config.max_length, config.d_model)
        layer = nn.TransformerEncoderLayer(
            d_model=config.d_model,
            nhead=config.nhead,
            dim_feedforward=config.dim_feedforward,
            dropout=config.dropout,
            activation="gelu",
            batch_first=True,
            norm_first=False,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=config.num_layers)
        self.final_norm = nn.LayerNorm(config.d_model)
        self.classifier = nn.Linear(config.d_model, output_dim)

    def forward(self, x: torch.Tensor, valid_mask: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[1] != 2:
            raise ValueError("x must have shape [batch, 2, length]")
        if valid_mask.shape != (x.shape[0], x.shape[2]):
            raise ValueError("valid_mask must have shape [batch, length]")
        if x.shape[2] > self.config.max_length:
            raise ValueError("input is longer than configured max_length")

        valid_mask = valid_mask.to(dtype=torch.bool)
        if torch.any(~valid_mask.any(dim=1)):
            raise ValueError("each read must contain at least one valid base")
        sequence = x.transpose(1, 2)
        positions = torch.arange(sequence.shape[1], device=x.device)
        sequence = self.input_projection(sequence) + self.position_embedding(positions)[None, :, :]
        sequence = self.encoder(sequence, src_key_padding_mask=~valid_mask)
        sequence = self.final_norm(sequence)
        weights = valid_mask.unsqueeze(-1).to(dtype=sequence.dtype)
        pooled = (sequence * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)
        return self.classifier(pooled)


def build_presence_model(config: ModelConfig) -> DNAReadTransformer:
    return DNAReadTransformer(config, output_dim=1)


def build_type_model(config: ModelConfig) -> DNAReadTransformer:
    return DNAReadTransformer(config, output_dim=4)


def save_checkpoint(
    path: str | Path,
    model: DNAReadTransformer,
    task: str,
    seed: int,
    epoch: int,
    validation_score: float,
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "task": task,
            "seed": int(seed),
            "epoch": int(epoch),
            "validation_score": float(validation_score),
            "model_config": asdict(model.config),
            "output_dim": model.output_dim,
            "state_dict": model.state_dict(),
        },
        path,
    )


def load_checkpoint(path: str | Path, device: str | torch.device = "cpu") -> tuple[DNAReadTransformer, dict[str, Any]]:
    checkpoint = torch.load(Path(path), map_location=device, weights_only=False)
    config = ModelConfig(**checkpoint["model_config"])
    model = DNAReadTransformer(config, output_dim=int(checkpoint["output_dim"]))
    model.load_state_dict(checkpoint["state_dict"])
    model.to(device)
    return model, checkpoint
