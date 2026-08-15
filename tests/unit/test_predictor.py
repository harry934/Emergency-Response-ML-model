"""Unit tests for interface/core/predictor.py."""
from __future__ import annotations

import numpy as np
import pytest
from unittest.mock import MagicMock

from core.predictor import LABELS, predict_image


VALID_INPUT = np.random.rand(1, 224, 224, 3).astype(np.float32)


class TestReturnType:
    def test_returns_tuple_of_two(self, mock_model_accident: MagicMock):
        result = predict_image(VALID_INPUT, mock_model_accident)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_label_is_string(self, mock_model_accident: MagicMock):
        label, _ = predict_image(VALID_INPUT, mock_model_accident)
        assert isinstance(label, str)

    def test_probs_is_ndarray(self, mock_model_accident: MagicMock):
        _, probs = predict_image(VALID_INPUT, mock_model_accident)
        assert isinstance(probs, np.ndarray)

    def test_probs_length_is_three(self, mock_model_accident: MagicMock):
        _, probs = predict_image(VALID_INPUT, mock_model_accident)
        assert len(probs) == 3


class TestLabelMapping:
    def test_accident_label(self, mock_model_accident: MagicMock):
        label, _ = predict_image(VALID_INPUT, mock_model_accident)
        assert label == "Accident"

    def test_heavy_traffic_label(self, mock_model_heavy_traffic: MagicMock):
        label, _ = predict_image(VALID_INPUT, mock_model_heavy_traffic)
        assert label == "HeavyTraffic"

    def test_normal_label(self, mock_model_normal: MagicMock):
        label, _ = predict_image(VALID_INPUT, mock_model_normal)
        assert label == "NormalRoadActivity"

    def test_label_is_in_labels_list(self, mock_model_accident: MagicMock):
        label, _ = predict_image(VALID_INPUT, mock_model_accident)
        assert label in LABELS


class TestProbabilities:
    def test_probs_sum_to_one(self, mock_model_accident: MagicMock):
        _, probs = predict_image(VALID_INPUT, mock_model_accident)
        assert pytest.approx(float(probs.sum()), abs=1e-5) == 1.0

    def test_probs_are_non_negative(self, mock_model_accident: MagicMock):
        _, probs = predict_image(VALID_INPUT, mock_model_accident)
        assert all(p >= 0.0 for p in probs)

    def test_probs_are_at_most_one(self, mock_model_accident: MagicMock):
        _, probs = predict_image(VALID_INPUT, mock_model_accident)
        assert all(p <= 1.0 for p in probs)

    def test_highest_prob_matches_label(self, mock_model_heavy_traffic: MagicMock):
        label, probs = predict_image(VALID_INPUT, mock_model_heavy_traffic)
        assert label == LABELS[int(np.argmax(probs))]


class TestFallbackMode:
    def test_none_model_returns_valid_label(self):
        label, _ = predict_image(VALID_INPUT, model=None)
        assert label in LABELS

    def test_none_model_probs_sum_to_one(self):
        _, probs = predict_image(VALID_INPUT, model=None)
        assert pytest.approx(float(probs.sum()), abs=1e-5) == 1.0

    def test_none_model_probs_length_three(self):
        _, probs = predict_image(VALID_INPUT, model=None)
        assert len(probs) == 3

    def test_none_model_model_predict_not_called(self):
        # No AttributeError should occur — there is no .predict on None
        label, probs = predict_image(VALID_INPUT, model=None)
        assert label in LABELS


class TestConfidenceThreshold:
    def test_uncertain_returned_below_threshold(self, mock_model_uncertain: MagicMock):
        # ~0.34 max confidence — threshold 0.5 should trigger Uncertain
        label, _ = predict_image(VALID_INPUT, mock_model_uncertain, threshold=0.5)
        assert label == "Uncertain"

    def test_label_returned_above_threshold(self, mock_model_accident: MagicMock):
        # 0.85 max confidence — threshold 0.5 should NOT trigger Uncertain
        label, _ = predict_image(VALID_INPUT, mock_model_accident, threshold=0.5)
        assert label == "Accident"

    def test_zero_threshold_never_uncertain(self, mock_model_uncertain: MagicMock):
        label, _ = predict_image(VALID_INPUT, mock_model_uncertain, threshold=0.0)
        assert label in LABELS  # never "Uncertain"

    def test_threshold_one_always_uncertain(self, mock_model_accident: MagicMock):
        # No real prediction can reach 100% confidence
        label, _ = predict_image(VALID_INPUT, mock_model_accident, threshold=1.0)
        assert label == "Uncertain"
