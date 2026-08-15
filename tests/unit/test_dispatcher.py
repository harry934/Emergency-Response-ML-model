"""Unit tests for interface/core/dispatcher.py."""
from __future__ import annotations

import pytest

from core.dispatcher import ACCIDENT_LABEL, get_dispatch_info


@pytest.fixture()
def area_info(mock_location_data):
    return mock_location_data["areas"]["Test Area"]


@pytest.fixture()
def hotlines(mock_location_data):
    return mock_location_data["general_emergency_hotlines"]


class TestAccidentDispatch:
    def test_returns_dict_for_accident(self, area_info, hotlines):
        result = get_dispatch_info("Accident", area_info, hotlines)
        assert result is not None
        assert isinstance(result, dict)

    def test_dispatch_has_hospital_key(self, area_info, hotlines):
        result = get_dispatch_info("Accident", area_info, hotlines)
        assert "hospital" in result

    def test_dispatch_has_police_key(self, area_info, hotlines):
        result = get_dispatch_info("Accident", area_info, hotlines)
        assert "police" in result

    def test_dispatch_has_hotlines_key(self, area_info, hotlines):
        result = get_dispatch_info("Accident", area_info, hotlines)
        assert "hotlines" in result

    def test_hospital_matches_area_info(self, area_info, hotlines):
        result = get_dispatch_info("Accident", area_info, hotlines)
        assert result["hospital"] == area_info["hospital"]

    def test_police_matches_area_info(self, area_info, hotlines):
        result = get_dispatch_info("Accident", area_info, hotlines)
        assert result["police"] == area_info["police"]

    def test_hotlines_matches_general(self, area_info, hotlines):
        result = get_dispatch_info("Accident", area_info, hotlines)
        assert result["hotlines"] == hotlines


class TestNonAccidentReturnsNone:
    def test_heavy_traffic_returns_none(self, area_info, hotlines):
        result = get_dispatch_info("HeavyTraffic", area_info, hotlines)
        assert result is None

    def test_normal_returns_none(self, area_info, hotlines):
        result = get_dispatch_info("NormalRoadActivity", area_info, hotlines)
        assert result is None

    def test_uncertain_returns_none(self, area_info, hotlines):
        result = get_dispatch_info("Uncertain", area_info, hotlines)
        assert result is None

    def test_empty_string_returns_none(self, area_info, hotlines):
        result = get_dispatch_info("", area_info, hotlines)
        assert result is None


class TestMissingKeys:
    def test_missing_hospital_raises_key_error(self, hotlines):
        bad_area = {"police": {"name": "P", "phone": "0", "lat": 0, "lon": 0}}
        with pytest.raises(KeyError):
            get_dispatch_info("Accident", bad_area, hotlines)

    def test_missing_police_raises_key_error(self, hotlines):
        bad_area = {"hospital": {"name": "H", "phone": "0", "lat": 0, "lon": 0}}
        with pytest.raises(KeyError):
            get_dispatch_info("Accident", bad_area, hotlines)

    def test_empty_area_info_raises_key_error(self, hotlines):
        with pytest.raises(KeyError):
            get_dispatch_info("Accident", {}, hotlines)


class TestConstants:
    def test_accident_label_constant(self):
        assert ACCIDENT_LABEL == "Accident"
