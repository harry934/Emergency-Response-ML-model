"""Unit tests for interface/core/preprocessor.py."""
from __future__ import annotations

import io
import numpy as np
import pytest
from PIL import Image

from core.preprocessor import TARGET_SIZE, preprocess_image


class TestOutputShape:
    def test_rgb_image_gives_correct_shape(self, sample_rgb_image: Image.Image):
        result = preprocess_image(sample_rgb_image)
        assert result.shape == (1, TARGET_SIZE[1], TARGET_SIZE[0], 3)

    def test_small_image_is_upscaled(self, tmp_path):
        small = Image.fromarray(
            (np.random.rand(32, 32, 3) * 255).astype("uint8"), "RGB"
        )
        buf = io.BytesIO()
        small.save(buf, format="PNG")
        buf.seek(0)
        result = preprocess_image(buf)
        assert result.shape == (1, 224, 224, 3)

    def test_large_image_is_downscaled(self):
        large = Image.fromarray(
            (np.random.rand(1024, 1024, 3) * 255).astype("uint8"), "RGB"
        )
        result = preprocess_image(large)
        assert result.shape == (1, 224, 224, 3)

    def test_custom_target_size(self, sample_rgb_image: Image.Image):
        result = preprocess_image(sample_rgb_image, target_size=(128, 128))
        assert result.shape == (1, 128, 128, 3)


class TestNormalisation:
    def test_pixel_values_in_zero_one(self, sample_rgb_image: Image.Image):
        result = preprocess_image(sample_rgb_image)
        assert float(result.min()) >= 0.0
        assert float(result.max()) <= 1.0

    def test_dtype_is_float32(self, sample_rgb_image: Image.Image):
        result = preprocess_image(sample_rgb_image)
        assert result.dtype == np.float32

    def test_pure_white_image_normalises_to_one(self):
        white = Image.fromarray(
            np.full((224, 224, 3), 255, dtype=np.uint8), "RGB"
        )
        result = preprocess_image(white)
        assert pytest.approx(float(result.max()), abs=1e-4) == 1.0

    def test_pure_black_image_normalises_to_zero(self):
        black = Image.fromarray(np.zeros((224, 224, 3), dtype=np.uint8), "RGB")
        result = preprocess_image(black)
        assert pytest.approx(float(result.max()), abs=1e-4) == 0.0


class TestChannelConversion:
    def test_rgba_converted_to_rgb(self, sample_rgba_image: Image.Image):
        result = preprocess_image(sample_rgba_image)
        assert result.shape[-1] == 3

    def test_grayscale_converted_to_rgb(self, sample_grayscale_image: Image.Image):
        result = preprocess_image(sample_grayscale_image)
        assert result.shape[-1] == 3

    def test_grayscale_output_shape(self, sample_grayscale_image: Image.Image):
        result = preprocess_image(sample_grayscale_image)
        assert result.shape == (1, 224, 224, 3)


class TestInputSources:
    def test_file_path_string(self, sample_image_file: str):
        result = preprocess_image(sample_image_file)
        assert result.shape == (1, 224, 224, 3)

    def test_bytes_input(self, sample_image_bytes: bytes):
        result = preprocess_image(sample_image_bytes)
        assert result.shape == (1, 224, 224, 3)

    def test_bytesio_input(self, sample_image_bytesio: io.BytesIO):
        result = preprocess_image(sample_image_bytesio)
        assert result.shape == (1, 224, 224, 3)

    def test_pil_image_input(self, sample_rgb_image: Image.Image):
        result = preprocess_image(sample_rgb_image)
        assert result.shape == (1, 224, 224, 3)


class TestErrorHandling:
    def test_none_raises_value_error(self):
        with pytest.raises(ValueError, match="None"):
            preprocess_image(None)  # type: ignore[arg-type]

    def test_corrupt_bytes_raises_value_error(self):
        with pytest.raises(ValueError):
            preprocess_image(b"this is not an image")

    def test_empty_bytes_raises_value_error(self):
        with pytest.raises(ValueError):
            preprocess_image(b"")
