"""Integration tests — real model.keras load and shape checks.

These tests require the model file to be present at the repository root and
TensorFlow to be installed.  They are skipped automatically if either
condition is not met.
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pytest

# Skip entire module if TensorFlow is not installed
tf = pytest.importorskip("tensorflow", reason="TensorFlow not installed")


def _model_exists(path: str) -> bool:
    return os.path.isfile(path)


@pytest.fixture(scope="module")
def loaded_model(real_model_path: str):
    if not _model_exists(real_model_path):
        pytest.skip(f"model.keras not found at {real_model_path}")
    from core.predictor import load_model
    return load_model(real_model_path)


class TestModelLoads:
    def test_no_exception_on_load(self, loaded_model):
        assert loaded_model is not None

    def test_model_has_predict_method(self, loaded_model):
        assert hasattr(loaded_model, "predict")

    def test_model_has_layers(self, loaded_model):
        assert len(loaded_model.layers) > 0


class TestModelInputShape:
    def test_input_rank_is_four(self, loaded_model):
        shape = loaded_model.input_shape
        assert len(shape) == 4, f"Expected rank-4 input, got {shape}"

    def test_input_height_is_224(self, loaded_model):
        assert loaded_model.input_shape[1] == 224

    def test_input_width_is_224(self, loaded_model):
        assert loaded_model.input_shape[2] == 224

    def test_input_channels_is_3(self, loaded_model):
        assert loaded_model.input_shape[3] == 3

    def test_batch_dim_is_none(self, loaded_model):
        assert loaded_model.input_shape[0] is None


class TestModelOutputShape:
    def test_output_rank_is_two(self, loaded_model):
        shape = loaded_model.output_shape
        assert len(shape) == 2, f"Expected rank-2 output, got {shape}"

    def test_output_classes_is_three(self, loaded_model):
        assert loaded_model.output_shape[1] == 3

    def test_output_batch_dim_is_none(self, loaded_model):
        assert loaded_model.output_shape[0] is None


class TestModelInference:
    def test_inference_returns_array(self, loaded_model):
        dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
        result = loaded_model.predict(dummy, verbose=0)
        assert isinstance(result, np.ndarray)

    def test_inference_output_shape(self, loaded_model):
        dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
        result = loaded_model.predict(dummy, verbose=0)
        assert result.shape == (1, 3)

    def test_inference_probs_sum_to_one(self, loaded_model):
        dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
        probs = loaded_model.predict(dummy, verbose=0)[0]
        assert pytest.approx(float(probs.sum()), abs=1e-4) == 1.0

    def test_inference_all_non_negative(self, loaded_model):
        dummy = np.random.rand(1, 224, 224, 3).astype(np.float32)
        probs = loaded_model.predict(dummy, verbose=0)[0]
        assert all(p >= 0.0 for p in probs)

    def test_batch_inference(self, loaded_model):
        batch = np.random.rand(4, 224, 224, 3).astype(np.float32)
        result = loaded_model.predict(batch, verbose=0)
        assert result.shape == (4, 3)
