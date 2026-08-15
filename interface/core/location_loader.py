"""Load and validate the locations JSON configuration file."""
from __future__ import annotations

import json
from typing import Any

REQUIRED_TOP_KEYS = ("areas", "general_emergency_hotlines")
REQUIRED_AREA_KEYS = ("sub_locations", "hospital", "police")


def load_locations(file_path: str) -> dict[str, Any]:
    """Load ``locations.json`` and validate its top-level schema.

    Parameters
    ----------
    file_path:
        Absolute or relative path to the JSON file.

    Returns
    -------
    dict
        Parsed location data.

    Raises
    ------
    FileNotFoundError
        If the file does not exist at *file_path*.
    json.JSONDecodeError
        If the file contains invalid JSON.
    KeyError
        If required top-level keys are missing.
    """
    with open(file_path, "r", encoding="utf-8") as fh:
        data: dict[str, Any] = json.load(fh)

    for key in REQUIRED_TOP_KEYS:
        if key not in data:
            raise KeyError(f"locations.json is missing required key: '{key}'")

    return data
