from __future__ import annotations

from pathlib import Path
from types import ModuleType
from typing import Any
import sys

import torch
from torch import nn

from .vendor_guard import DEFAULT_VENDOR_ROOT, require_clean_vendor


AUTHOR_MODEL_NAMES = ("cnn", "lstm", "transformer", "resnet")


def load_author_models_module(repository: str | Path = DEFAULT_VENDOR_ROOT) -> ModuleType:
    """Load the author's models.py verbatim, without copying its class definitions."""

    root = require_clean_vendor(repository)
    source = root / "models.py"
    module_name = "zhouph0313_dna_original_models"
    existing = sys.modules.get(module_name)
    if existing is not None:
        return existing
    # Compile directly instead of SourceFileLoader so importing never writes a
    # __pycache__ entry into the immutable vendor snapshot.
    module = ModuleType(module_name)
    module.__file__ = str(source)
    module.__package__ = ""
    sys.modules[module_name] = module
    code = compile(source.read_bytes(), str(source), "exec")
    exec(code, module.__dict__)
    return module


def create_original_model(
    model_name: str,
    num_classes: int,
    repository: str | Path = DEFAULT_VENDOR_ROOT,
) -> nn.Module:
    """Instantiate exactly the architecture selected in the author's vote script."""

    if num_classes <= 0:
        raise ValueError("num_classes must be positive")
    module = load_author_models_module(repository)
    normalized = model_name.lower()
    if normalized == "cnn":
        return module.CNN1DClassifier(num_classes=num_classes)
    if normalized == "lstm":
        return module.LSTMClassifier(
            input_dim=4, hidden_dim=128, num_layers=2, num_classes=num_classes
        )
    if normalized == "transformer":
        return module.TransformerClassifier(
            d_model=128, nhead=4, num_layers=2, num_classes=num_classes
        )
    if normalized == "resnet":
        return module.ResNet1DClassifier(num_classes=num_classes)
    raise ValueError(f"unsupported author model {model_name!r}; choose from {AUTHOR_MODEL_NAMES}")


def load_original_checkpoint(
    model_name: str,
    checkpoint_path: str | Path,
    num_classes: int,
    device: str | torch.device = "cpu",
    repository: str | Path = DEFAULT_VENDOR_ROOT,
    state_dict_key: str | None = None,
) -> nn.Module:
    model = create_original_model(model_name, num_classes, repository)
    # The supplied author assets are plain state_dict files.  Restricting the
    # loader to tensor weights avoids executing arbitrary pickle payloads.
    checkpoint: Any = torch.load(Path(checkpoint_path), map_location=device, weights_only=True)
    state_dict = checkpoint[state_dict_key] if state_dict_key is not None else checkpoint
    if not isinstance(state_dict, dict):
        raise TypeError("author checkpoint must be a raw state_dict or use state_dict_key")
    model.load_state_dict(state_dict, strict=True)
    return model.to(device).eval()
