"""Unit tests for interface/core/render.py (confidence display)."""
from __future__ import annotations

from core.predictor import LABELS
from core.render import BASE_COLORS, build_confidence_html, format_label


def _html(probs):
    return build_confidence_html(LABELS, probs)


class TestHtmlContainsAllLabels:
    def test_contains_accident(self):
        html = _html([0.9, 0.05, 0.05])
        assert "Accident" in html

    def test_contains_heavy_traffic_display_name(self):
        html = _html([0.05, 0.9, 0.05])
        assert "Heavy Traffic" in html

    def test_contains_normal_activity_display_name(self):
        html = _html([0.05, 0.05, 0.9])
        assert "Normal Activity" in html

    def test_all_three_classes_present(self):
        html = _html([0.5, 0.3, 0.2])
        assert "Accident" in html
        assert "Heavy Traffic" in html
        assert "Normal Activity" in html


class TestPercentageValues:
    def test_percentages_displayed_in_html(self):
        html = _html([0.85, 0.10, 0.05])
        assert "85%" in html
        assert "10%" in html
        assert "5%" in html

    def test_zero_percent_shown(self):
        html = _html([1.0, 0.0, 0.0])
        assert "0%" in html


class TestHtmlStructure:
    def test_output_is_string(self):
        html = _html([0.33, 0.33, 0.34])
        assert isinstance(html, str)

    def test_output_not_empty(self):
        html = _html([0.33, 0.33, 0.34])
        assert len(html) > 0

    def test_uses_score_row_class(self):
        html = _html([0.5, 0.3, 0.2])
        assert html.count('class="score-row"') == 3

    def test_base_colors_present(self):
        html = _html([0.5, 0.3, 0.2])
        for color in BASE_COLORS.values():
            assert color in html

    def test_no_risk_badges(self):
        html = _html([0.9, 0.05, 0.05])
        assert "High" not in html
        assert "Medium" not in html
        assert "risk-badge" not in html


class TestFormatLabel:
    def test_accident_label(self):
        assert format_label("Accident") == "Accident"

    def test_heavy_traffic_label(self):
        assert format_label("HeavyTraffic") == "Heavy Traffic"

    def test_normal_label(self):
        assert format_label("NormalRoadActivity") == "Normal Activity"

    def test_unknown_label_passthrough(self):
        assert format_label("Unknown") == "Unknown"


class TestEdgeCases:
    def test_single_label_single_prob(self):
        html = build_confidence_html(["Accident"], [1.0])
        assert "Accident" in html
        assert "100%" in html

    def test_uniform_probs(self):
        html = _html([1 / 3, 1 / 3, 1 / 3])
        assert isinstance(html, str)

    def test_unknown_label_falls_back_gracefully(self):
        html = build_confidence_html(["UnknownClass"], [0.5])
        assert "UnknownClass" in html
