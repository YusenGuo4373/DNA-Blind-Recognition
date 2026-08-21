from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import csv
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from .config import ExperimentConfig
from .data import ReadClassificationDataset
from .metrics import macro_f1_score
from .models import (
    DNAReadTransformer,
    build_presence_model,
    build_type_model,
    load_checkpoint,
    save_checkpoint,
)


@dataclass(frozen=True)
class EpochRecord:
    epoch: int
    train_loss: float
    validation_loss: float
    validation_macro_f1: float


@dataclass(frozen=True)
class TrainingResult:
    task: str
    seed: int
    checkpoint: Path
    best_epoch: int
    best_validation_macro_f1: float
    history: tuple[EpochRecord, ...]


def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def make_loader(
    dataset: ReadClassificationDataset,
    batch_size: int,
    num_workers: int,
    seed: int,
    shuffle: bool,
) -> DataLoader:
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        worker_init_fn=seed_worker,
        generator=generator,
        persistent_workers=num_workers > 0,
    )


def _loss_for_task(task: str) -> nn.Module:
    if task == "presence":
        return nn.BCEWithLogitsLoss()
    if task == "type":
        return nn.CrossEntropyLoss()
    raise ValueError("task must be 'presence' or 'type'")


def _batch_loss(
    task: str,
    model: DNAReadTransformer,
    batch: dict[str, torch.Tensor],
    criterion: nn.Module,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    x = batch["x"].to(device, non_blocking=True)
    valid_mask = batch["valid_mask"].to(device, non_blocking=True)
    target = batch["target"].to(device, non_blocking=True)
    logits = model(x, valid_mask)
    if task == "presence":
        logits = logits.squeeze(-1)
        loss = criterion(logits, target.to(dtype=logits.dtype))
    else:
        loss = criterion(logits, target)
    return loss, logits


@torch.inference_mode()
def evaluate_read_model(
    task: str,
    model: DNAReadTransformer,
    loader: DataLoader,
    device: torch.device,
) -> tuple[float, float]:
    criterion = _loss_for_task(task)
    model.eval()
    total_loss = 0.0
    total_items = 0
    truth: list[np.ndarray] = []
    prediction: list[np.ndarray] = []
    for batch in loader:
        loss, logits = _batch_loss(task, model, batch, criterion, device)
        batch_size = int(batch["target"].shape[0])
        total_loss += float(loss.item()) * batch_size
        total_items += batch_size
        labels = batch["target"].detach().cpu().numpy()
        if task == "presence":
            predicted = (torch.sigmoid(logits) >= 0.5).to(dtype=torch.int64).cpu().numpy()
            labels = labels.astype(np.int64)
            classes = (0, 1)
        else:
            predicted = torch.argmax(logits, dim=-1).cpu().numpy()
            classes = (0, 1, 2, 3)
        truth.append(labels)
        prediction.append(predicted)
    if total_items == 0:
        raise ValueError("validation loader is empty")
    score = macro_f1_score(np.concatenate(truth), np.concatenate(prediction), classes)
    return total_loss / total_items, score


def train_model(
    config: ExperimentConfig,
    task: str,
    seed: int,
    output_directory: str | Path,
    device: str | torch.device | None = None,
    train_known_reads_per_type: int | None = None,
    train_no_ecc_reads_per_subtype: int | None = None,
    val_known_reads_per_type: int | None = None,
    val_no_ecc_reads_per_subtype: int | None = None,
    epochs: int | None = None,
) -> TrainingResult:
    if task not in {"presence", "type"}:
        raise ValueError("task must be 'presence' or 'type'")
    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    set_global_seed(seed)

    train_dataset = ReadClassificationDataset(
        config,
        task=task,
        split="train",
        seed=seed,
        known_reads_per_type=train_known_reads_per_type,
        no_ecc_reads_per_subtype=train_no_ecc_reads_per_subtype,
    )
    validation_dataset = ReadClassificationDataset(
        config,
        task=task,
        split="validation",
        seed=seed,
        known_reads_per_type=val_known_reads_per_type,
        no_ecc_reads_per_subtype=val_no_ecc_reads_per_subtype,
    )
    train_loader = make_loader(
        train_dataset,
        config.training.batch_size,
        config.training.num_workers,
        seed=seed,
        shuffle=True,
    )
    validation_loader = make_loader(
        validation_dataset,
        config.training.batch_size,
        config.training.num_workers,
        seed=seed + 10_000,
        shuffle=False,
    )

    model = build_presence_model(config.model) if task == "presence" else build_type_model(config.model)
    model.to(device)
    criterion = _loss_for_task(task)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.training.learning_rate,
        weight_decay=config.training.weight_decay,
    )
    amp_enabled = bool(config.training.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=amp_enabled)
    maximum_epochs = epochs or (
        config.training.presence_epochs if task == "presence" else config.training.type_epochs
    )
    checkpoint = output_directory / f"{task}_seed_{seed}.pt"
    history: list[EpochRecord] = []
    best_score = -np.inf
    best_epoch = 0
    epochs_without_improvement = 0

    for epoch in range(1, maximum_epochs + 1):
        model.train()
        total_loss = 0.0
        total_items = 0
        for batch in train_loader:
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast(device_type=device.type, enabled=amp_enabled):
                loss, _ = _batch_loss(task, model, batch, criterion, device)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.training.gradient_clip)
            scaler.step(optimizer)
            scaler.update()
            batch_size = int(batch["target"].shape[0])
            total_loss += float(loss.item()) * batch_size
            total_items += batch_size

        validation_loss, validation_score = evaluate_read_model(
            task, model, validation_loader, device
        )
        record = EpochRecord(
            epoch=epoch,
            train_loss=total_loss / max(total_items, 1),
            validation_loss=validation_loss,
            validation_macro_f1=validation_score,
        )
        history.append(record)
        print(
            f"task={task} seed={seed} epoch={epoch}/{maximum_epochs} "
            f"train_loss={record.train_loss:.6f} val_loss={record.validation_loss:.6f} "
            f"val_macro_f1={record.validation_macro_f1:.4f}",
            flush=True,
        )
        if validation_score > best_score + 1e-12:
            best_score = validation_score
            best_epoch = epoch
            epochs_without_improvement = 0
            save_checkpoint(checkpoint, model, task, seed, epoch, validation_score)
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.training.early_stopping_patience:
                break

    history_path = output_directory / f"{task}_seed_{seed}_history.csv"
    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("epoch", "train_loss", "validation_loss", "validation_macro_f1"),
        )
        writer.writeheader()
        for record in history:
            writer.writerow(record.__dict__)

    # Verify that the stored best checkpoint is readable before handing it off.
    _, metadata = load_checkpoint(checkpoint, device="cpu")
    if int(metadata["epoch"]) != best_epoch:
        raise RuntimeError("best checkpoint epoch does not match training state")
    return TrainingResult(
        task=task,
        seed=seed,
        checkpoint=checkpoint,
        best_epoch=best_epoch,
        best_validation_macro_f1=float(best_score),
        history=tuple(history),
    )
