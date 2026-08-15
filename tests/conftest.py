"""Shared pytest fixtures for the Emergency-Responce test suite."""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest
from PIL import Image

# ---------------------------------------------------------------------------
# Make interface/core importable from tests without installing the package
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
INTERFACE_DIR = REPO_ROOT / "interface"
sys.path.insert(0, str(INTERFACE_DIR))


# ---------------------------------------------------------------------------
# Location data
# ---------------------------------------------------------------------------

MOCK_LOCATION_DATA: dict[str, Any] = {
    "areas": {
        "Test Area": {
            "sub_locations": [
                {"name": "Point A", "lat": -1.25, "lon": 36.85},
                {"name": "Point B", "lat": -1.26, "lon": 36.86},
            ],
            "hospital": {
                "name": "Test Hospital",
                "phone": "0700000001",
                "lat": -1.27,
                "lon": 36.87,
            },
            "police": {
                "name": "Test Police Station",
                "phone": "0700000002",
                "lat": -1.28,
                "lon": 36.88,
            },
        },
        "Second Area": {
            "sub_locations": [
                {"name": "Point C", "lat": -1.30, "lon": 36.90},
            ],
            "hospital": {
                "name": "Second Hospital",
                "phone": "0700000003",
                "lat": -1.31,
                "lon": 36.91,
            },
            "police": {
                "name": "Second Police",
                "phone": "0700000004",
                "lat": -1.32,
                "lon": 36.92,
            },
        },
    },
    "general_emergency_hotlines": {
        "police_control_room": "999",
        "ambulance_services": ["Red Cross 111", "St. John 222"],
    },
}


@pytest.fixture()
def mock_location_data() -> dict[str, Any]:
    """Return a minimal valid location data dict matching the real schema."""
    return MOCK_LOCATION_DATA.copy()


@pytest.fixture()
def locations_json_file(tmp_path: Path, mock_location_data: dict) -> str:
    """Write mock location data to a temporary JSON file; return its path."""
    path = tmp_path / "locations.json"
    path.write_text(json.dumps(mock_location_data), encoding="utf-8")
    return str(path)


@pytest.fixture()
def missing_locations_file(tmp_path: Path) -> str:
    """Return a path to a JSON file that does not exist."""
    return str(tmp_path / "nonexistent.json")


@pytest.fixture()
def malformed_json_file(tmp_path: Path) -> str:
    """Return a path to a file containing invalid JSON."""
    path = tmp_path / "bad.json"
    path.write_text("{not valid json", encoding="utf-8")
    return str(path)


# ---------------------------------------------------------------------------
# Images
# ---------------------------------------------------------------------------

def _make_rgb_image(width: int = 224, height: int = 224) -> Image.Image:
    data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(data, mode="RGB")


@pytest.fixture()
def sample_rgb_image() -> Image.Image:
    """224×224 random RGB PIL image."""
    return _make_rgb_image()


@pytest.fixture()
def sample_rgba_image() -> Image.Image:
    """224×224 random RGBA PIL image."""
    data = np.random.randint(0, 256, (224, 224, 4), dtype=np.uint8)
    return Image.fromarray(data, mode="RGBA")


@pytest.fixture()
def sample_grayscale_image() -> Image.Image:
    """224×224 random grayscale PIL image."""
    data = np.random.randint(0, 256, (224, 224), dtype=np.uint8)
    return Image.fromarray(data, mode="L")


@pytest.fixture()
def sample_image_file(tmp_path: Path) -> str:
    """Save a small RGB JPEG to a temporary file; return its path string."""
    img = _make_rgb_image(64, 64)
    path = tmp_path / "test_road.jpg"
    img.save(str(path), format="JPEG")
    return str(path)


@pytest.fixture()
def sample_image_bytes() -> bytes:
    """Return raw JPEG bytes of a small RGB image."""
    buf = io.BytesIO()
    _make_rgb_image(64, 64).save(buf, format="JPEG")
    return buf.getvalue()


@pytest.fixture()
def sample_image_bytesio() -> io.BytesIO:
    """Return a BytesIO of a small RGB JPEG (file-like, seeked to 0)."""
    buf = io.BytesIO()
    _make_rgb_image(64, 64).save(buf, format="JPEG")
    buf.seek(0)
    return buf


# ---------------------------------------------------------------------------
# Model mocks
# ---------------------------------------------------------------------------

def _make_mock_model(probs: list[float] | None = None) -> MagicMock:
    """Return a MagicMock whose .predict() returns a known probability array."""
    if probs is None:
        probs = [0.85, 0.10, 0.05]  # high accident confidence by default
    arr = np.array([probs], dtype=np.float32)
    m = MagicMock()
    m.predict.return_value = arr
    return m


@pytest.fixture()
def mock_model_accident() -> MagicMock:
    """Mock model that predicts Accident with 85% confidence."""
    return _make_mock_model([0.85, 0.10, 0.05])


@pytest.fixture()
def mock_model_heavy_traffic() -> MagicMock:
    """Mock model that predicts HeavyTraffic with 80% confidence."""
    return _make_mock_model([0.10, 0.80, 0.10])


@pytest.fixture()
def mock_model_normal() -> MagicMock:
    """Mock model that predicts NormalRoadActivity with 90% confidence."""
    return _make_mock_model([0.05, 0.05, 0.90])


@pytest.fixture()
def mock_model_uncertain() -> MagicMock:
    """Mock model that returns near-uniform probabilities (uncertain)."""
    return _make_mock_model([0.34, 0.33, 0.33])


# ---------------------------------------------------------------------------
# Repository paths
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def repo_root() -> Path:
    return REPO_ROOT


@pytest.fixture(scope="session")
def real_model_path(repo_root: Path) -> str:
    return str(repo_root / "model.keras")


@pytest.fixture(scope="session")
def real_locations_path(repo_root: Path) -> str:
    return str(repo_root / "locations.json")
