"""Incremental open-set validation around the unchanged author recognizer."""

from .comparison import (
    IncrementalThresholds,
    SharedLogitComparison,
    calibrate_thresholds,
    compare_shared_logits,
    summarize_comparisons,
)
from .collector import SharedLogitDataset, TorchPresenceDetector, collect_shared_logits

__all__ = [
    "IncrementalThresholds",
    "SharedLogitComparison",
    "calibrate_thresholds",
    "compare_shared_logits",
    "summarize_comparisons",
    "SharedLogitDataset",
    "TorchPresenceDetector",
    "collect_shared_logits",
]
