"""UI tests for interface/app.py using streamlit.testing.v1.AppTest."""
from __future__ import annotations

import io
import re
import sys
from contextlib import ExitStack, contextmanager
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

EMOJI_RE = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F000-\U0001F2FF\U0000FE0F]"
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
def _app_context(probs: list[float], uploaded: bool = False):
    import streamlit as st

    st.cache_resource.clear()
    st.cache_data.clear()

    patches = [patch("core.predictor.load_model", return_value=_make_mock_model(probs))]
    if uploaded:
        patches.append(patch("streamlit.file_uploader", return_value=_make_upload_file()))
    with ExitStack() as stack:
        for p in patches:
            stack.enter_context(p)
        at = AppTest.from_file(APP_PATH, default_timeout=30)
        at.run()
        yield at


def _all_text(at) -> list[str]:
    texts = [m.value for m in at.markdown]
    texts += [t.value for t in at.title]
    texts += [c.value for c in at.caption]
    for group in (at.error, at.warning, at.success, at.info):
        texts += [e.value for e in group]
    return texts


@skip_no_apptest
class TestAppStartup:
    def test_app_runs_without_exception(self):
        with _app_context([0.05, 0.05, 0.90]) as at:
            assert not at.exception, f"App raised: {at.exception}"

    def test_title_is_present(self):
        with _app_context([0.05, 0.05, 0.90]) as at:
            assert any("Road Incident Detection" in t.value for t in at.title)

    def test_no_error_on_startup(self):
        with _app_context([0.05, 0.05, 0.90]) as at:
            assert len(at.error) == 0

    def test_empty_state_hint_shown(self):
        with _app_context([0.05, 0.05, 0.90]) as at:
            assert any("Upload an image" in c.value for c in at.caption)


@skip_no_apptest
class TestWidgets:
    @pytest.fixture(autouse=True)
    def _app(self):
        with _app_context([0.05, 0.05, 0.90]) as at:
            self.at = at

    def test_area_and_camera_selectboxes(self):
        labels = [s.label for s in self.at.selectbox]
        assert "Area" in labels
        assert "Camera" in labels

    def test_file_uploader_exists(self):
        assert len(self.at.get("file_uploader")) >= 1

    def test_no_settings_controls(self):
        assert len(self.at.slider) == 0
        assert not any(e.label == "Settings" for e in self.at.expander)

    def test_no_sidebar_content(self):
        assert len(self.at.sidebar.children) == 0

    def test_section_headings_present_without_numbers(self):
        markdown = " ".join(m.value for m in self.at.markdown)
        assert "Location" in markdown
        assert "Upload image" in markdown
        assert not re.search(r"\d\. (Location|Upload image)", markdown)


@skip_no_apptest
class TestAreaCascade:
    def test_cameras_update_when_area_changes(self):
        with _app_context([0.05, 0.05, 0.90]) as at:
            areas = at.selectbox[0].options
            if len(areas) < 2:
                pytest.skip("Only one area in locations.json")
            before = list(at.selectbox[1].options)
            at.selectbox[0].set_value(areas[1]).run()
            assert list(at.selectbox[1].options) != before


@skip_no_apptest
class TestNonAccidentResult:
    def test_success_message_shown(self):
        with _app_context([0.05, 0.05, 0.90], uploaded=True) as at:
            assert any("No emergency response required" in s.value for s in at.success)

    def test_metrics_shown(self):
        with _app_context([0.05, 0.05, 0.90], uploaded=True) as at:
            metrics = {m.label: m.value for m in at.metric}
            assert metrics["Classification"] == "Normal Activity"
            assert metrics["Confidence"] == "90.0%"

    def test_three_progress_bars(self):
        with _app_context([0.05, 0.05, 0.90], uploaded=True) as at:
            assert len(at.get("progress")) == 3

    def test_no_emergency_section(self):
        with _app_context([0.05, 0.90, 0.05], uploaded=True) as at:
            assert len(at.error) == 0
            assert not any("Emergency response" in m.value for m in at.markdown)


@skip_no_apptest
class TestAccidentResult:
    def test_accident_error_shown(self):
        with _app_context([0.95, 0.03, 0.02], uploaded=True) as at:
            assert any("Accident detected" in e.value for e in at.error)

    def test_emergency_section_shown(self):
        with _app_context([0.95, 0.03, 0.02], uploaded=True) as at:
            markdown = " ".join(m.value for m in at.markdown)
            assert "Emergency response" in markdown
            assert "Nearest hospital" in markdown
            assert "Nearest police station" in markdown


@skip_no_apptest
class TestUncertainResult:
    def test_warning_shown_below_threshold(self):
        with _app_context([0.40, 0.35, 0.25], uploaded=True) as at:
            assert any("threshold" in w.value.lower() for w in at.warning)


@skip_no_apptest
class TestHistory:
    def test_history_expander_appears_after_upload(self):
        with _app_context([0.05, 0.05, 0.90], uploaded=True) as at:
            assert any(e.label == "History" for e in at.expander)


class TestMapTiles:
    def test_no_tile_provider_requiring_api_key(self):
        source = Path(APP_PATH).read_text(encoding="utf-8")
        assert "CartoDB" not in source
        assert "server.arcgisonline.com" in source


@skip_no_apptest
class TestNoEmojis:
    @pytest.mark.parametrize("probs", [[0.05, 0.05, 0.90], [0.95, 0.03, 0.02]])
    def test_rendered_text_has_no_emojis(self, probs):
        with _app_context(probs, uploaded=True) as at:
            offenders = [t for t in _all_text(at) if EMOJI_RE.search(t)]
            assert not offenders, f"Emojis found in: {offenders}"
