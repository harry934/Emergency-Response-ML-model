"""Small UI helpers: display names, section headings and icon alignment CSS."""
from __future__ import annotations

from core.icons import icon_label

_DISPLAY_NAMES: dict[str, str] = {
    "Accident": "Accident",
    "HeavyTraffic": "Heavy Traffic",
    "NormalRoadActivity": "Normal Activity",
    "Uncertain": "Uncertain",
}

ICON_CSS = """
<style>
.ico { display: inline-flex; vertical-align: -3px; margin-right: 8px; color: inherit; opacity: 0.7; }
.section-heading { font-size: 1.05rem; font-weight: 600; margin: 0 0 0.5rem 0; }
.app-logo { display: flex; align-items: center; gap: 12px; }
.app-logo svg { width: 40px; height: 40px; }
</style>
"""


def format_label(label: str) -> str:
    """Return a human-readable name for a prediction class."""
    return _DISPLAY_NAMES.get(label, label)


def section_heading(icon_name: str, title: str) -> str:
    """Return HTML for a section heading with a leading SVG icon."""
    return f'<p class="section-heading">{icon_label(icon_name, title)}</p>'
