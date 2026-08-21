"""Hierarchical blind ECC recognition for DNA reads.

The package is deliberately separate from the legacy experiment scripts in the
repository.  It exposes deterministic data generation, read-level Transformer
models, the paper-compatible two-level soft vote, and the three-way cascade:
``no_ecc`` / ``unknown_ecc`` / ``known_ecc``.
"""

from .config import ExperimentConfig, KNOWN_CODE_TYPES
from .voting import HierarchicalDecision, Thresholds, hierarchical_decision

__all__ = [
    "ExperimentConfig",
    "KNOWN_CODE_TYPES",
    "HierarchicalDecision",
    "Thresholds",
    "hierarchical_decision",
]
