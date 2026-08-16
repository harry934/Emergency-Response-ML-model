"""UI tests for interface/app.py using streamlit.testing.v1.AppTest.

AppTest drives the Streamlit script without a browser or server.
Tests patch the model and file uploader because Streamlit 1.51 exposes
file_uploader as UnknownElement without an upload() helper.
"""
from __future__ import annotations

import io
import sys
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from PIL import Image

INTERFACE_DIR = Path(__file__).parent.parent.parent / "interface"
sys.path.insert(0, str(INTERFACE_DIR))

APP_PATH = str(INTERFACE_DIR / "app.py")

try:
    from streamlit.testing.v1 import AppTest
    _HAS_APPTEST = True
except ImportError:
    _HAS_APPTEST = False

skip_no_apptest = pytest.mark.skipif(
    not _HAS_APPTEST,
    reason="streamlit.testing.v1.AppTest not available (upgrade Streamlit >= 1.18)",
)


def _make_jpeg_bytes(width: int = 64, height: int = 64) -> bytes:
    data = np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)
    buf = io.BytesIO()
    Image.fromarray(data, "RGB").save(buf, format="JPEG")
    return buf.getvalue()


def _make_upload_file(name: str = "road.jpg") -> io.BytesIO:
    buf = io.BytesIO(_make_jpeg_bytes())
    buf.seek(0)
    buf.name = name
    return buf


def _make_mock_model(probs: list[float]) -> MagicMock:
    m = MagicMock()
    m.predict.return_value = np.array([probs], dtype=np.float32)
    return m


@contextmanager
def _app_context(probs: list[float], threshold: float | None = None, uploaded: bool = False):
    """Patch model (and optionally uploader / threshold) then yield an AppTest."""
    import streamlit as st

    st.cache_resource.clear()
    st.cache_data.clear()

    patches = [patch("core.predictor.load_model", return_value=_make_mock_model(probs))]
    if uploaded:
        patches.append(patch("streamlit.file_uploader", return_value=_make_upload_file()))
    if threshold is not None:
        patches.append(patch("streamlit.slider", return_value=threshold))

    from contextlib import ExitStack

    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        yield at


@skip_no_apptest
class TestAppStartup:
    def test_app_runs_without_exception(self):
        with _app_context([0.05, 0.05, 0.90]) as at:
            assert not at.exception, f"App raised: {at.exception}"

    def test_title_is_present(self):
        with _app_context([0.05, 0.05, 0.90]) as at:
            markdown = [m.value for m in at.markdown]
            assert any("Road Incident Detection" in m for m in markdown)

    def test_no_error_on_startup(self):
        with _app_context([0.05, 0.05, 0.90]) as at:
            errors = [e.value for e in at.error]
            assert not any("locations.json not found" in e for e in errors)


@skip_no_apptest
class TestWidgets:
    @pytest.fixture(autouse=True)
    def _app(self):
        with _app_context([0.05, 0.05, 0.90]) as at:
            self.at = at

    def test_area_selectbox_exists(self):
        assert len(self.at.selectbox) >= 1

    def test_sub_location_selectbox_exists(self):
        assert len(self.at.selectbox) >= 2

    def test_file_uploader_exists(self):
        assert len(self.at.get("file_uploader")) >= 1

    def test_confidence_slider_exists(self):
        assert len(self.at.slider) >= 1

    def test_area_selectbox_has_options(self):
        options = self.at.selectbox[0].options
        assert len(options) >= 1

    def test_sub_location_selectbox_has_options(self):
        options = self.at.selectbox[1].options
        assert len(options) >= 1


@skip_no_apptest
class TestAreaCascade:
    def test_sub_locations_update_when_area_changes(self):
        with _app_context([0.05, 0.05, 0.90]) as at:
            initial_subs = list(at.selectbox[1].options)
            area_options = at.selectbox[0].options
            if len(area_options) < 2:
                pytest.skip("Only one area in locations.json")

            at.selectbox[0].set_value(area_options[1]).run()
            new_subs = list(at.selectbox[1].options)
            assert initial_subs != new_subs


@skip_no_apptest
class TestNonAccidentResult:
    def test_no_dispatch_info_shown(self):
        with _app_context([0.05, 0.05, 0.90], uploaded=True) as at:
            markdown = [m.value for m in at.markdown]
            assert any("No emergency dispatch required" in m for m in markdown)

    def test_result_metric_shown(self):
        with _app_context([0.05, 0.05, 0.90], uploaded=True) as at:
            assert len(at.metric) >= 1

    def test_label_is_normal(self):
        with _app_context([0.05, 0.05, 0.90], uploaded=True) as at:
            metrics = [m.label for m in at.metric]
            values = [m.value for m in at.metric]
            assert "Classification" in metrics
            assert any("Normal Activity" in str(v) for v in values)

    def test_no_accident_error_shown(self):
        with _app_context([0.05, 0.90, 0.05], uploaded=True) as at:
            markdown = [m.value for m in at.markdown]
            assert not any("Accident detected" in m for m in markdown)


@skip_no_apptest
class TestAccidentResult:
    def test_accident_status_shown(self):
        with _app_context([0.95, 0.03, 0.02], uploaded=True) as at:
            markdown = [m.value for m in at.markdown]
            assert any("Accident detected" in m for m in markdown)

    def test_result_metric_shows_accident(self):
        with _app_context([0.95, 0.03, 0.02], uploaded=True) as at:
            values = [m.value for m in at.metric]
            assert any("Accident" in str(v) for v in values)

    def test_dispatch_map_section_shown(self):
        with _app_context([0.95, 0.03, 0.02], uploaded=True) as at:
            markdown = [m.value for m in at.markdown]
            assert any("Response map" in m for m in markdown)


@skip_no_apptest
class TestUncertainResult:
    def test_uncertain_status_shown_below_threshold(self):
        with _app_context([0.85, 0.10, 0.05], threshold=1.0, uploaded=True) as at:
            markdown = [m.value for m in at.markdown]
            assert any("threshold" in m.lower() for m in markdown)


@skip_no_apptest
class TestIncidentHistory:
    def test_history_expander_appears_after_upload(self):
        with _app_context([0.05, 0.05, 0.90], uploaded=True) as at:
            expanders = [e.label for e in at.expander]
            assert any("Session history" in e for e in expanders)
