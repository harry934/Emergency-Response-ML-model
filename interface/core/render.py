"""HTML rendering helpers -- confidence bars and risk badges (flat solid theme)."""
from __future__ import annotations

import math

BASE_COLORS: dict[str, str] = {
    "Accident": "#dc2626",
    "HeavyTraffic": "#d97706",
    "NormalRoadActivity": "#16a34a",
}

_DISPLAY_NAMES: dict[str, str] = {
    "Accident": "Accident",
    "HeavyTraffic": "Heavy Traffic",
    "NormalRoadActivity": "Normal Activity",
}

_ICONS: dict[str, str] = {
    "Accident": "🚨",
    "HeavyTraffic": "⚠️",
    "NormalRoadActivity": "✅",
}

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.conf-container {
    background-color: #1e293b;
    border: 1px solid #334155;
    border-radius: 8px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.75rem;
}
.conf-header {
    font-family: 'Inter', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    color: #94a3b8;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    margin-bottom: 0.85rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.pred-row {
    display: flex;
    align-items: center;
    margin: 0.5rem 0;
    gap: 10px;
    font-family: 'Inter', sans-serif;
}
.pred-icon { font-size: 0.95rem; width: 22px; text-align: center; flex-shrink: 0; }
.pred-label { width: 120px; font-weight: 600; color: #f1f5f9; font-size: 0.82rem; flex-shrink: 0; }
.pred-bar-wrap {
    flex: 1;
    height: 18px;
    background-color: #0f172a;
    border-radius: 4px;
    overflow: hidden;
    position: relative;
    border: 1px solid #334155;
}
.pred-fill {
    height: 100%;
    border-radius: 3px;
    position: relative;
    display: flex;
    align-items: center;
}
.bar-inner-text {
    position: absolute;
    left: 6px;
    font-weight: 600;
    font-size: 10px;
    color: #ffffff;
}
.pred-pct {
    width: 44px;
    text-align: right;
    font-family: 'Inter', monospace;
    font-size: 0.82rem;
    font-weight: 700;
    flex-shrink: 0;
}
.risk-chip {
    font-size: 0.7rem;
    font-weight: 600;
    border-radius: 4px;
    padding: 2px 8px;
    flex-shrink: 0;
    text-align: center;
    min-width: 54px;
}
.conf-legend {
    display: flex;
    gap: 1rem;
    margin-top: 0.75rem;
    padding-top: 0.65rem;
    border-top: 1px solid #334155;
    flex-wrap: wrap;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 0.72rem;
    color: #94a3b8;
    font-family: 'Inter', sans-serif;
}
.legend-dot { width: 8px; height: 8px; border-radius: 2px; }
</style>
"""


def _risk_level(pct: int) -> tuple[str, str, str]:
    """Return (colour_hex, bg_hex, label) for a given integer percentage."""
    if pct >= 70:
        return "#dc2626", "#450a0a", "High"
    if pct >= 40:
        return "#d97706", "#451a03", "Medium"
    return "#16a34a", "#052e16", "Low"


def build_confidence_html(labels: list[str], probs: list[float] | "np.ndarray") -> str:  # noqa: F821
    """Build a solid-theme HTML string for the confidence bar visualisation."""
    rows: list[str] = [_CSS, '<div class="conf-container">']
    rows.append('<div class="conf-header"><span>📊</span> Confidence Breakdown</div>')

    for lab, p in zip(labels, probs):
        pct = int(math.floor(float(p) * 100 + 0.5))  # round-half-up
        base = BASE_COLORS.get(lab, "#2563eb")
        rcolor, rbg, rlabel = _risk_level(pct)
        icon = _ICONS.get(lab, "•")
        display_name = _DISPLAY_NAMES.get(lab, lab)
        inner_text = f"{pct}%" if pct > 15 else ""

        fill_style = f"width:{pct}%;background-color:{base};"
        pct_style = f"color:{base};"
        chip_style = f"background-color:{rbg};color:{rcolor};border:1px solid {rcolor};"

        rows.append(
            f'<div class="pred-row" title="{display_name}: {pct}% - Risk: {rlabel}">'
            f'<div class="pred-icon">{icon}</div>'
            f'<div class="pred-label">{display_name}</div>'
            f'<div class="pred-bar-wrap">'
            f'<div class="pred-fill" style="{fill_style}">'
            f'<span class="bar-inner-text">{inner_text}</span>'
            f"</div></div>"
            f'<div class="pred-pct" style="{pct_style}">{pct}%</div>'
            f'<div class="risk-chip" style="{chip_style}">{rlabel}</div>'
            f"</div>"
        )

    rows.append(
        '<div class="conf-legend">'
        '<div class="legend-item"><div class="legend-dot" style="background-color:#16a34a;"></div>Low risk</div>'
        '<div class="legend-item"><div class="legend-dot" style="background-color:#d97706;"></div>Medium risk</div>'
        '<div class="legend-item"><div class="legend-dot" style="background-color:#dc2626;"></div>High risk</div>'
        "</div>"
    )
    rows.append("</div>")
    return "\n".join(rows)
