"""Integration tests — full preprocessing → prediction → dispatch pipeline.

Uses the real model.keras file.  Tests are skipped if the model is absent.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

pytest.importorskip("tensorflow", reason="TensorFlow not installed")

from core.dispatcher import get_dispatch_info
from core.predictor import LABELS, load_model, predict_image
from core.preprocessor import preprocess_image


def _model_exists(path: str) -> bool:
    return os.path.isfile(path)


@pytest.fixture(scope="module")
def real_model(real_model_path: str):
    if not _model_exists(real_model_path):
        pytest.skip(f"model.keras not found at {real_model_path}")
    return load_model(real_model_path)


# ---------------------------------------------------------------------------
# Preprocessing → prediction
# ---------------------------------------------------------------------------

class TestPreprocessThenPredict:
    def test_pipeline_returns_valid_label(self, real_model, sample_image_file: str):
        arr = preprocess_image(sample_image_file)
        label, probs = predict_image(arr, real_model)
        assert label in LABELS

    def test_probs_sum_to_one(self, real_model, sample_image_file: str):
        arr = preprocess_image(sample_image_file)
        _, probs = predict_image(arr, real_model)
        assert pytest.approx(float(probs.sum()), abs=1e-4) == 1.0

    def test_probs_length_three(self, real_model, sample_image_file: str):
        arr = preprocess_image(sample_image_file)
        _, probs = predict_image(arr, real_model)
        assert len(probs) == 3

    def test_pipeline_with_bytesio_source(self, real_model, sample_image_bytesio):
        arr = preprocess_image(sample_image_bytesio)
        label, _ = predict_image(arr, real_model)
        assert label in LABELS

    def test_pipeline_with_raw_bytes(self, real_model, sample_image_bytes: bytes):
        arr = preprocess_image(sample_image_bytes)
        label, _ = predict_image(arr, real_model)
        assert label in LABELS


# ---------------------------------------------------------------------------
# Prediction → dispatch
# ---------------------------------------------------------------------------

class TestPredictThenDispatch:
    def _run(self, real_model, image_path: str, area_info: dict, hotlines: dict):
        arr = preprocess_image(image_path)
        label, probs = predict_image(arr, real_model)
        dispatch = get_dispatch_info(label, area_info, hotlines)
        return label, probs, dispatch

    def test_dispatch_is_none_or_dict(self, real_model, sample_image_file, mock_location_data):
        area = mock_location_data["areas"]["Test Area"]
        hotlines = mock_location_data["general_emergency_hotlines"]
        label, _, dispatch = self._run(real_model, sample_image_file, area, hotlines)
        if label == "Accident":
            assert isinstance(dispatch, dict)
        else:
            assert dispatch is None

    def test_accident_dispatch_has_all_keys(self, real_model, mock_location_data):
        """Force an Accident prediction via a mock and check dispatch completeness."""
        from unittest.mock import MagicMock
        m = MagicMock()
        m.predict.return_value = np.array([[0.95, 0.03, 0.02]], dtype=np.float32)
        area = mock_location_data["areas"]["Test Area"]
        hotlines = mock_location_data["general_emergency_hotlines"]

        arr = np.random.rand(1, 224, 224, 3).astype(np.float32)
        label, _, dispatch = (
            lambda: (
                lambda lbl, pr: (lbl, pr, get_dispatch_info(lbl, area, hotlines))
            )(*predict_image(arr, m))
        )()
        assert label == "Accident"
        assert dispatch is not None
        assert "hospital" in dispatch
        assert "police" in dispatch
        assert "hotlines" in dispatch

    def test_non_accident_no_dispatch(self, mock_location_data):
        from unittest.mock import MagicMock
        m = MagicMock()
        m.predict.return_value = np.array([[0.02, 0.03, 0.95]], dtype=np.float32)
        area = mock_location_data["areas"]["Test Area"]
        hotlines = mock_location_data["general_emergency_hotlines"]

        arr = np.random.rand(1, 224, 224, 3).astype(np.float32)
        label, _, dispatch = (
            lambda: (
                lambda lbl, pr: (lbl, pr, get_dispatch_info(lbl, area, hotlines))
            )(*predict_image(arr, m))
        )()
        assert label == "NormalRoadActivity"
        assert dispatch is None


# ---------------------------------------------------------------------------
# Threshold integration
# ---------------------------------------------------------------------------

class TestThresholdIntegration:
    def test_threshold_produces_uncertain(self, real_model, sample_image_file: str):
        arr = preprocess_image(sample_image_file)
        label, _ = predict_image(arr, real_model, threshold=1.0)
        assert label == "Uncertain"

    def test_no_threshold_never_uncertain(self, real_model, sample_image_file: str):
        arr = preprocess_image(sample_image_file)
        label, _ = predict_image(arr, real_model, threshold=0.0)
        assert label in LABELS
