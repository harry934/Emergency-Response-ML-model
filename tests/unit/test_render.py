"""Unit tests for interface/core/render.py and interface/core/icons.py."""
from __future__ import annotations

import pytest

from core.icons import ICONS, icon, icon_label
from core.render import ICON_CSS, format_label, section_heading

REQUIRED_ICONS = [
    "camera", "map_pin", "upload", "activity", "hospital",
    "shield", "phone", "alert_triangle", "history",
]


class TestFormatLabel:
    @pytest.mark.parametrize(
        "raw, expected",
        [
            ("Accident", "Accident"),
            ("HeavyTraffic", "Heavy Traffic"),
            ("NormalRoadActivity", "Normal Activity"),
            ("Uncertain", "Uncertain"),
            ("SomethingElse", "SomethingElse"),
        ],
    )
    def test_display_names(self, raw, expected):
        assert format_label(raw) == expected


class TestIcons:
    @pytest.mark.parametrize("name", REQUIRED_ICONS)
    def test_required_icon_exists(self, name):
        assert name in ICONS

    @pytest.mark.parametrize("name", list(ICONS))
    def test_icon_is_valid_svg(self, name):
        svg = ICONS[name]
        assert svg.startswith("<svg ")
        assert svg.endswith("</svg>")
        assert 'stroke="currentColor"' in svg

    def test_icon_lookup(self):
        assert icon("phone") == ICONS["phone"]

    def test_unknown_icon_raises(self):
        with pytest.raises(KeyError):
            icon("does-not-exist")

    def test_icon_label_wraps_svg_and_text(self):
        html = icon_label("camera", "Camera")
        assert html.startswith('<span class="ico"><svg ')
        assert html.endswith("</span>Camera")


class TestSectionHeading:
    def test_contains_title_and_svg(self):
        html = section_heading("map_pin", "Location")
        assert "Location" in html
        assert "<svg " in html
        assert 'class="section-heading"' in html

    def test_css_defines_icon_class(self):
        assert ".ico" in ICON_CSS
        assert ".section-heading" in ICON_CSS

    def test_icon_colour_follows_theme_text(self):
        assert "color: inherit" in ICON_CSS


class TestThemeConfig:
    def test_light_and_dark_themes_defined(self):
        import tomllib
        from pathlib import Path

        config_path = Path(__file__).resolve().parents[2] / ".streamlit" / "config.toml"
        theme = tomllib.loads(config_path.read_text(encoding="utf-8"))["theme"]
        for mode in ("light", "dark"):
            assert {"backgroundColor", "secondaryBackgroundColor", "textColor"} <= theme[mode].keys()
        assert "base" not in theme
