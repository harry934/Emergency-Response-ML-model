"""Inline SVG icons (Lucide-style strokes, MIT licensed paths)."""
from __future__ import annotations

_SVG_OPEN = (
    '<svg xmlns="http://www.w3.org/2000/svg" width="18" height="18" viewBox="0 0 24 24" '
    'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" '
    'stroke-linejoin="round" aria-hidden="true">'
)

_PATHS: dict[str, str] = {
    "camera": (
        '<path d="M14.5 4h-5L7 7H4a2 2 0 0 0-2 2v9a2 2 0 0 0 2 2h16a2 2 0 0 0 2-2V9'
        'a2 2 0 0 0-2-2h-3l-2.5-3z"/><circle cx="12" cy="13" r="3"/>'
    ),
    "map_pin": (
        '<path d="M20 10c0 6-8 12-8 12s-8-6-8-12a8 8 0 0 1 16 0Z"/>'
        '<circle cx="12" cy="10" r="3"/>'
    ),
    "upload": (
        '<path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>'
        '<polyline points="17 8 12 3 7 8"/><line x1="12" x2="12" y1="3" y2="15"/>'
    ),
    "activity": '<path d="M22 12h-4l-3 9L9 3l-3 9H2"/>',
    "hospital": (
        '<path d="M12 6v4"/><path d="M14 8h-4"/>'
        '<path d="M18 22V4a2 2 0 0 0-2-2H8a2 2 0 0 0-2 2v18"/>'
        '<path d="M2 22h20"/><path d="M10 22v-4h4v4"/>'
    ),
    "shield": '<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/>',
    "phone": (
        '<path d="M22 16.92v3a2 2 0 0 1-2.18 2 19.79 19.79 0 0 1-8.63-3.07 19.5 19.5 0 0 1-6-6'
        ' 19.79 19.79 0 0 1-3.07-8.67A2 2 0 0 1 4.11 2h3a2 2 0 0 1 2 1.72c.13.96.36 1.9.7 2.81'
        'a2 2 0 0 1-.45 2.11L8.09 9.91a16 16 0 0 0 6 6l1.27-1.27a2 2 0 0 1 2.11-.45c.91.34 1.85.57'
        ' 2.81.7A2 2 0 0 1 22 16.92z"/>'
    ),
    "alert_triangle": (
        '<path d="m21.73 18-8-14a2 2 0 0 0-3.48 0l-8 14A2 2 0 0 0 4 21h16a2 2 0 0 0 1.73-3Z"/>'
        '<line x1="12" x2="12" y1="9" y2="13"/><line x1="12" x2="12.01" y1="17" y2="17"/>'
    ),
    "history": (
        '<path d="M3 12a9 9 0 1 0 9-9 9.75 9.75 0 0 0-6.74 2.74L3 8"/>'
        '<path d="M3 3v5h5"/><path d="M12 7v5l4 2"/>'
    ),
    "settings": (
        '<line x1="4" x2="20" y1="6" y2="6"/><line x1="4" x2="20" y1="12" y2="12"/>'
        '<line x1="4" x2="20" y1="18" y2="18"/><circle cx="9" cy="6" r="2"/>'
        '<circle cx="15" cy="12" r="2"/><circle cx="7" cy="18" r="2"/>'
    ),
}

ICONS: dict[str, str] = {name: f"{_SVG_OPEN}{path}</svg>" for name, path in _PATHS.items()}


def icon(name: str) -> str:
    """Return the SVG markup for *name* (raises KeyError for unknown icons)."""
    return ICONS[name]


def icon_label(name: str, text: str) -> str:
    """Return an inline icon followed by *text*, for use in headings."""
    return f'<span class="ico">{ICONS[name]}</span>{text}'
