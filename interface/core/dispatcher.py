"""Dispatch logic — decides whether to alert emergency services."""
from __future__ import annotations

from typing import Any

ACCIDENT_LABEL = "Accident"


def get_dispatch_info(
    label: str,
    area_info: dict[str, Any],
    general_hotlines: dict[str, Any],
) -> dict[str, Any] | None:
    """Return dispatch information when an accident is detected, else ``None``.

    Parameters
    ----------
    label:
        Predicted class label from the model.
    area_info:
        Entry for the selected area from ``locations.json``, must contain
        ``"hospital"`` and ``"police"`` keys.
    general_hotlines:
        The ``"general_emergency_hotlines"`` dict from ``locations.json``.

    Returns
    -------
    dict or None
        ``{"hospital": ..., "police": ..., "hotlines": ...}`` on accident, or
        ``None`` for HeavyTraffic / NormalRoadActivity.

    Raises
    ------
    KeyError
        If ``area_info`` is missing ``"hospital"`` or ``"police"`` keys.
    """
    if label != ACCIDENT_LABEL:
        return None

    _ = area_info["hospital"]  # raise KeyError early if missing
    _ = area_info["police"]

    return {
        "hospital": area_info["hospital"],
        "police": area_info["police"],
        "hotlines": general_hotlines,
    }
