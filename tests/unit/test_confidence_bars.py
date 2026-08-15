"""Unit tests for interface/core/render.py (confidence bar HTML generation)."""
from __future__ import annotations

import pytest

from core.predictor import LABELS
from core.render import BASE_COLORS, build_confidence_html


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

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


class TestRiskBadgeHigh:
    """Confidence ≥ 70% → 'High' risk badge."""

    def test_accident_70_percent_is_high(self):
        html = build_confidence_html(["Accident"], [0.70])
        assert "High" in html

    def test_accident_90_percent_is_high(self):
        html = _html([0.90, 0.05, 0.05])
        # First occurrence of High corresponds to Accident bar
        assert "High" in html

    def test_heavy_traffic_75_percent_is_high(self):
        html = _html([0.05, 0.75, 0.20])
        assert "High" in html


class TestRiskBadgeMedium:
    """40% ≤ confidence < 70% → 'Medium' risk badge."""

    def test_40_percent_is_medium(self):
        html = build_confidence_html(["Accident"], [0.40])
        assert "Medium" in html

    def test_65_percent_is_medium(self):
        html = build_confidence_html(["Accident"], [0.65])
        assert "Medium" in html


class TestRiskBadgeLow:
    """Confidence < 40% → 'Low' risk badge."""

    def test_10_percent_is_low(self):
        html = build_confidence_html(["Accident"], [0.10])
        assert "Low" in html

    def test_zero_percent_is_low(self):
        html = build_confidence_html(["Accident"], [0.0])
        assert "Low" in html

    def test_39_percent_is_low(self):
        html = build_confidence_html(["Accident"], [0.39])
        assert "Low" in html


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

    def test_contains_style_tag(self):
        html = _html([0.5, 0.3, 0.2])
        assert "<style>" in html

    def test_contains_legend(self):
        html = _html([0.5, 0.3, 0.2])
        assert "legend" in html

    def test_base_colors_present(self):
        html = _html([0.5, 0.3, 0.2])
        for color in BASE_COLORS.values():
            assert color in html

    def test_three_pred_rows(self):
        html = _html([0.5, 0.3, 0.2])
        assert html.count('class="pred-row"') == 3


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
