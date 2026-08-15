"""Unit tests for interface/core/location_loader.py."""
from __future__ import annotations

import json

import pytest

from core.location_loader import REQUIRED_TOP_KEYS, load_locations


class TestValidLoad:
    def test_returns_dict(self, locations_json_file: str):
        result = load_locations(locations_json_file)
        assert isinstance(result, dict)

    def test_has_areas_key(self, locations_json_file: str):
        result = load_locations(locations_json_file)
        assert "areas" in result

    def test_has_general_hotlines_key(self, locations_json_file: str):
        result = load_locations(locations_json_file)
        assert "general_emergency_hotlines" in result

    def test_areas_is_dict(self, locations_json_file: str):
        result = load_locations(locations_json_file)
        assert isinstance(result["areas"], dict)

    def test_each_area_has_sub_locations(self, locations_json_file: str):
        result = load_locations(locations_json_file)
        for area in result["areas"].values():
            assert "sub_locations" in area

    def test_each_area_has_hospital(self, locations_json_file: str):
        result = load_locations(locations_json_file)
        for area in result["areas"].values():
            assert "hospital" in area

    def test_each_area_has_police(self, locations_json_file: str):
        result = load_locations(locations_json_file)
        for area in result["areas"].values():
            assert "police" in area

    def test_sub_location_has_lat_lon(self, locations_json_file: str):
        result = load_locations(locations_json_file)
        for area in result["areas"].values():
            for sub in area["sub_locations"]:
                assert "lat" in sub
                assert "lon" in sub


class TestMissingFile:
    def test_raises_file_not_found(self, missing_locations_file: str):
        with pytest.raises(FileNotFoundError):
            load_locations(missing_locations_file)


class TestMalformedJson:
    def test_raises_json_decode_error(self, malformed_json_file: str):
        with pytest.raises(json.JSONDecodeError):
            load_locations(malformed_json_file)


class TestMissingTopLevelKeys:
    def test_missing_areas_raises_key_error(self, tmp_path):
        path = tmp_path / "no_areas.json"
        path.write_text(
            json.dumps({"general_emergency_hotlines": {"police_control_room": "999"}}),
            encoding="utf-8",
        )
        with pytest.raises(KeyError, match="areas"):
            load_locations(str(path))

    def test_missing_hotlines_raises_key_error(self, tmp_path):
        path = tmp_path / "no_hotlines.json"
        path.write_text(json.dumps({"areas": {}}), encoding="utf-8")
        with pytest.raises(KeyError, match="general_emergency_hotlines"):
            load_locations(str(path))

    def test_empty_json_object_raises_key_error(self, tmp_path):
        path = tmp_path / "empty.json"
        path.write_text("{}", encoding="utf-8")
        with pytest.raises(KeyError):
            load_locations(str(path))


class TestEdgeCases:
    def test_area_with_empty_sub_locations(self, tmp_path):
        data = {
            "areas": {
                "Empty Area": {
                    "sub_locations": [],
                    "hospital": {"name": "H", "phone": "0", "lat": 0, "lon": 0},
                    "police": {"name": "P", "phone": "0", "lat": 0, "lon": 0},
                }
            },
            "general_emergency_hotlines": {"police_control_room": "999", "ambulance_services": []},
        }
        path = tmp_path / "empty_subs.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        result = load_locations(str(path))
        assert result["areas"]["Empty Area"]["sub_locations"] == []

    def test_required_top_keys_constant(self):
        assert "areas" in REQUIRED_TOP_KEYS
        assert "general_emergency_hotlines" in REQUIRED_TOP_KEYS

    def test_real_locations_file(self, real_locations_path: str):
        """Smoke test that the committed locations.json loads correctly."""
        result = load_locations(real_locations_path)
        assert "areas" in result
