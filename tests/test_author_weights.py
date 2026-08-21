import numpy as np
import pytest

from author_baseline.recognizer import OneHotArchive
from author_baseline.weights import (
    DEFAULT_WEIGHT_ROOT,
    build_primary_type_recognizer,
    inspect_author_weights,
)


@pytest.mark.skipif(not DEFAULT_WEIGHT_ROOT.is_dir(), reason="author weights not supplied")
def test_all_supplied_weights_match_manifest_and_models() -> None:
    records = inspect_author_weights(device="cpu")
    assert len(records) == 12
    assert all(record.hash_matches for record in records)
    assert all(record.strict_load_compatible for record in records)


@pytest.mark.skipif(not DEFAULT_WEIGHT_ROOT.is_dir(), reason="author weights not supplied")
def test_primary_recognizer_connects_only_type_model() -> None:
    recognizer = build_primary_type_recognizer(device="cpu", batch_size=2)
    description = recognizer.describe()
    assert description["code_type"]["class_name"] == "TransformerClassifier"
    assert description["code_type"]["labels"] == ["BCH", "Convolutional", "LDPC", "Polar"]
    assert description["code_rate"] is None
    assert description["code_length"] is None

    bases = np.arange(32) % 4
    one_hot = np.eye(4, dtype=np.float32)[bases].T[None, None, :, :]
    archive = OneHotArchive(one_hot=one_hot, mask=np.ones((1, 1, 32), dtype=np.float32))
    prediction = recognizer.predict_type(archive)
    assert prediction.label in description["code_type"]["labels"]
    assert len(prediction.probabilities) == 4
    assert sum(prediction.probabilities) == pytest.approx(1.0)
