"""Adapters around the unmodified zhouph0313/DNA author repository."""

from .cascade import CascadeDecision, CascadeThresholds, HierarchicalAuthorAdapter
from .recognizer import OneHotArchive, OriginalBlindRecognizer, OriginalTaskModel
from .vendor_guard import VendorVerification, verify_vendor_snapshot
from .weights import build_primary_type_recognizer, inspect_author_weights

__all__ = [
    "CascadeDecision",
    "CascadeThresholds",
    "HierarchicalAuthorAdapter",
    "OneHotArchive",
    "OriginalBlindRecognizer",
    "OriginalTaskModel",
    "VendorVerification",
    "verify_vendor_snapshot",
    "build_primary_type_recognizer",
    "inspect_author_weights",
]
