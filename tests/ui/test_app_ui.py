"""UI tests for interface/app.py using streamlit.testing.v1.AppTest.

AppTest drives the Streamlit script without a browser or server.
Tests patch the model so no GPU/TF heavy lifting happens during CI.
"""
from __future__ import annotations

import io
import json
import os
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

# Ensure interface/ is on sys.path (also done in conftest, but explicit here
# in case this file is run standalone).
INTERFACE_DIR = Path(__file__).parent.parent.parent / "interface"
sys.path.insert(0, str(INTERFACE_DIR))

APP_PATH = str(INTERFACE_DIR / "app.py")

try:
    from streamlit.testing.v1 import AppTest  # Streamlit ≥ 1.18
    _HAS_APPTEST = True
except ImportError:
    _HAS_APPTEST = False

skip_no_apptest = pytest.mark.skipif(
    not _HAS_APPTEST,
    reason="streamlit.testing.v1.AppTest not available (upgrade Streamlit ≥ 1.18)",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_jpeg_bytes(width: int = 64, height: int = 64) -> bytes:
    data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(data, "RGB").save(buf, format="JPEG")
    return buf.getvalue()


def _make_mock_model(probs: list[float]) -> MagicMock:
    m = MagicMock()
    m.predict.return_value = np.array([probs], dtype=np.float32)
    return m


def _patch_model_load(probs: list[float]):
    """Context manager / decorator that replaces load_model in app."""
    return patch("core.predictor.load_model", return_value=_make_mock_model(probs))


# ---------------------------------------------------------------------------
# Basic startup
# ---------------------------------------------------------------------------

@skip_no_apptest
class TestAppStartup:
    def test_app_runs_without_exception(self):
        with _patch_model_load([0.05, 0.05, 0.90]):
            at = AppTest.from_file(APP_PATH, default_timeout=30)
            at.run()
        assert not at.exception, f"App raised: {at.exception}"

    def test_title_is_present(self):
        with _patch_model_load([0.05, 0.05, 0.90]):
            at = AppTest.from_file(APP_PATH, default_timeout=30)
            at.run()
        titles = [t.value for t in at.title]
        assert any("Emergency Response" in t for t in titles)

    def test_no_error_on_startup(self):
        with _patch_model_load([0.05, 0.05, 0.90]):
            at = AppTest.from_file(APP_PATH, default_timeout=30)
            at.run()
        # st.error should not fire during clean startup
        errors = [e.value for e in at.error]
        assert not any("locations.json not found" in e for e in errors)


# ---------------------------------------------------------------------------
# Widget tree
# ---------------------------------------------------------------------------

@skip_no_apptest
class TestWidgets:
    @pytest.fixture(autouse=True)
    def _app(self):
        with _patch_model_load([0.05, 0.05, 0.90]):
            at = AppTest.from_file(APP_PATH, default_timeout=30)
            at.run()
        self.at = at

    def test_area_selectbox_exists(self):
        assert len(self.at.selectbox) >= 1

    def test_sub_location_selectbox_exists(self):
        assert len(self.at.selectbox) >= 2

    def test_file_uploader_exists(self):
        assert len(self.at.file_uploader) >= 1

    def test_confidence_slider_exists(self):
        assert len(self.at.slider) >= 1

    def test_area_selectbox_has_options(self):
        # Options for the first selectbox come from locations.json
        options = self.at.selectbox[0].options
        assert len(options) >= 1

    def test_sub_location_selectbox_has_options(self):
        options = self.at.selectbox[1].options
        assert len(options) >= 1


# ---------------------------------------------------------------------------
# Area → sub-location cascade
# ---------------------------------------------------------------------------

@skip_no_apptest
class TestAreaCascade:
    def test_sub_locations_update_when_area_changes(self):
        with _patch_model_load([0.05, 0.05, 0.90]):
            at = AppTest.from_file(APP_PATH, default_timeout=30)
            at.run()

        initial_subs = list(at.selectbox[1].options)

        # Change to the second area (if it exists)
        area_options = at.selectbox[0].options
        if len(area_options) < 2:
            pytest.skip("Only one area in locations.json — cascade test not meaningful")

        with _patch_model_load([0.05, 0.05, 0.90]):
            at.selectbox[0].set_value(area_options[1]).run()

        new_subs = list(at.selectbox[1].options)
        # Sub-location lists for different areas should differ
        assert initial_subs != new_subs


# ---------------------------------------------------------------------------
# Normal / heavy traffic result
# ---------------------------------------------------------------------------

@skip_no_apptest
class TestNonAccidentResult:
    def _run_with_upload(self, probs: list[float]) -> "AppTest":
        with _patch_model_load(probs):
            at = AppTest.from_file(APP_PATH, default_timeout=30)
            at.run()
            at.file_uploader[0].upload(
                {"name": "road.jpg", "content": _make_jpeg_bytes(), "type": "image/jpeg"}
            )
            at.run()
        return at

    def test_no_dispatch_info_shown(self):
        at = self._run_with_upload([0.05, 0.05, 0.90])
        infos = [i.value for i in at.info]
        assert any("No emergency dispatch required" in i for i in infos)

    def test_result_subheader_shown(self):
        at = self._run_with_upload([0.05, 0.05, 0.90])
        subheaders = [s.value for s in at.subheader]
        assert any("Result:" in s for s in subheaders)

    def test_label_is_normal(self):
        at = self._run_with_upload([0.05, 0.05, 0.90])
        subheaders = [s.value for s in at.subheader]
        assert any("NormalRoadActivity" in s for s in subheaders)

    def test_no_accident_error_shown(self):
        at = self._run_with_upload([0.05, 0.90, 0.05])
        errors = [e.value for e in at.error]
        assert not any("Accident detected" in e for e in errors)


# ---------------------------------------------------------------------------
# Accident result
# ---------------------------------------------------------------------------

@skip_no_apptest
class TestAccidentResult:
    def _run_with_upload(self) -> "AppTest":
        with _patch_model_load([0.95, 0.03, 0.02]):
            at = AppTest.from_file(APP_PATH, default_timeout=30)
            at.run()
            at.file_uploader[0].upload(
                {"name": "crash.jpg", "content": _make_jpeg_bytes(), "type": "image/jpeg"}
            )
            at.run()
        return at

    def test_accident_error_banner_shown(self):
        at = self._run_with_upload()
        errors = [e.value for e in at.error]
        assert any("Accident detected" in e for e in errors)

    def test_result_subheader_shows_accident(self):
        at = self._run_with_upload()
        subheaders = [s.value for s in at.subheader]
        assert any("Accident" in s for s in subheaders)

    def test_dispatch_map_subheader_shown(self):
        at = self._run_with_upload()
        subheaders = [s.value for s in at.subheader]
        assert any("Map" in s or "Dispatch" in s for s in subheaders)


# ---------------------------------------------------------------------------
# Uncertain prediction
# ---------------------------------------------------------------------------

@skip_no_apptest
class TestUncertainResult:
    def test_warning_shown_below_threshold(self):
        """Set threshold to 1.0 (guaranteed uncertain) and check warning."""
        with _patch_model_load([0.85, 0.10, 0.05]):
            at = AppTest.from_file(APP_PATH, default_timeout=30)
            at.run()
            # Set threshold slider to maximum (1.0)
            at.slider[0].set_value(1.0)
            at.file_uploader[0].upload(
                {"name": "road.jpg", "content": _make_jpeg_bytes(), "type": "image/jpeg"}
            )
            at.run()
        warnings = [w.value for w in at.warning]
        assert any("threshold" in w.lower() or "uncertain" in w.lower() for w in warnings)


# ---------------------------------------------------------------------------
# Incident history
# ---------------------------------------------------------------------------

@skip_no_apptest
class TestIncidentHistory:
    def test_history_expander_appears_after_upload(self):
        with _patch_model_load([0.05, 0.05, 0.90]):
            at = AppTest.from_file(APP_PATH, default_timeout=30)
            at.run()
            at.file_uploader[0].upload(
                {"name": "road.jpg", "content": _make_jpeg_bytes(), "type": "image/jpeg"}
            )
            at.run()
        expanders = [e.label for e in at.expander]
        assert any("History" in e or "Incident" in e for e in expanders)
