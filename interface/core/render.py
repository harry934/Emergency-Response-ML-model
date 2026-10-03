"""HTML rendering helpers -- confidence bars and risk badges (dark theme)."""
from __future__ import annotations

import math

BASE_COLORS: dict[str, str] = {
    "Accident":           "#e53935",
    "HeavyTraffic":       "#ff9800",
    "NormalRoadActivity": "#00c853",
}

GLOW_COLORS: dict[str, str] = {
    "Accident":           "rgba(229,57,53,0.35)",
    "HeavyTraffic":       "rgba(255,152,0,0.35)",
    "NormalRoadActivity": "rgba(0,200,83,0.35)",
}

_DISPLAY_NAMES: dict[str, str] = {
    "Accident":           "Accident",
    "HeavyTraffic":       "Heavy Traffic",
    "NormalRoadActivity": "Normal Activity",
}

_ICONS: dict[str, str] = {
    "Accident":           "🚨",
    "HeavyTraffic":       "🚦",
    "NormalRoadActivity": "✅",
}

_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
.conf-container {
    background: #1c2230;
    border: 1px solid #2d3650;
    border-radius: 12px;
    padding: 1rem 1.25rem;
    margin-bottom: 0.5rem;
}
.conf-header {
    font-family: 'Inter', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    color: #8b949e;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 6px;
}
.pred-row {
    display: flex;
    align-items: center;
    margin: 0.6rem 0;
    gap: 10px;
    font-family: 'Inter', sans-serif;
}
.pred-icon { font-size: 1rem; width: 24px; text-align: center; flex-shrink: 0; }
.pred-label { width: 120px; font-weight: 600; color: #e6edf3; font-size: 0.82rem; flex-shrink: 0; }
.pred-bar-wrap {
    flex: 1;
    height: 20px;
    background: #0d1117;
    border-radius: 10px;
    overflow: hidden;
    position: relative;
    border: 1px solid #2d3650;
}
.pred-fill {
    height: 100%;
    border-radius: 10px;
    transition: width 900ms cubic-bezier(0.4, 0, 0.2, 1);
    position: relative;
    display: flex;
    align-items: center;
}
.bar-inner-text {
    position: absolute;
    left: 8px;
    font-weight: 700;
    font-size: 11px;
    color: rgba(255,255,255,0.9);
    letter-spacing: 0.02em;
}
.pred-pct {
    width: 42px;
    text-align: right;
    font-family: 'Courier New', monospace;
    font-size: 0.82rem;
    font-weight: 700;
    flex-shrink: 0;
}
.risk-chip {
    font-size: 0.7rem;
    font-weight: 600;
    border-radius: 12px;
    padding: 2px 8px;
    flex-shrink: 0;
}
.conf-legend {
    display: flex;
    gap: 1rem;
    margin-top: 0.75rem;
    padding-top: 0.75rem;
    border-top: 1px solid #2d3650;
    flex-wrap: wrap;
}
.legend-item {
    display: flex;
    align-items: center;
    gap: 5px;
    font-size: 0.72rem;
    color: #8b949e;
    font-family: 'Inter', sans-serif;
}
.legend-dot { width: 8px; height: 8px; border-radius: 50%; }
</style>
"""


def _risk_level(pct: int) -> tuple[str, str, str]:
    """Return ``(colour_hex, bg_hex, label)`` for a given integer percentage."""
    if pct >= 70:
        return "#ff4444", "rgba(255,68,68,0.15)", "High"
    if pct >= 40:
        return "#ff9800", "rgba(255,152,0,0.15)", "Medium"
    return "#00c853", "rgba(0,200,83,0.15)", "Low"


def build_confidence_html(labels: list[str], probs: list[float] | "np.ndarray") -> str:  # noqa: F821
    """Build a dark-theme HTML string for the confidence bar visualisation.

    Parameters
    ----------
    labels:
        Ordered list of class names (length 3).
    probs:
        Corresponding probability values (each in ``[0, 1]``).

    Returns
    -------
    str
        Self-contained HTML / CSS string safe to pass to
        ``st.markdown(..., unsafe_allow_html=True)``.
    """
    rows: list[str] = [_CSS, '<div class="conf-container">']
    rows.append('<div class="conf-header"><span>📊</span> Confidence Breakdown</div>')

    for lab, p in zip(labels, probs):
        pct = int(math.floor(float(p) * 100 + 0.5))  # round-half-up
        base = BASE_COLORS.get(lab, "#00b4d8")
        glow = GLOW_COLORS.get(lab, "rgba(0,180,216,0.3)")
        rcolor, rbg, rlabel = _risk_level(pct)
        icon = _ICONS.get(lab, "●")
        display_name = _DISPLAY_NAMES.get(lab, lab)
        inner_text = f"{pct}%" if pct > 12 else ""

        fill_style = (
            f"width:{pct}%;"
            f"background:linear-gradient(90deg,{base}cc,{base});"
            f"box-shadow:0 0 10px {glow};"
        )
        pct_style = f"color:{base};"
        chip_style = f"background:{rbg};color:{rcolor};border:1px solid {rcolor}44;"

        rows.append(
            f'<div class="pred-row" title="{display_name}: {pct}% -- Risk: {rlabel}">'
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
        '<div class="legend-item"><div class="legend-dot" style="background:#00c853;box-shadow:0 0 4px #00c853;"></div>Low risk</div>'
        '<div class="legend-item"><div class="legend-dot" style="background:#ff9800;box-shadow:0 0 4px #ff9800;"></div>Medium risk</div>'
        '<div class="legend-item"><div class="legend-dot" style="background:#ff4444;box-shadow:0 0 4px #ff4444;"></div>High risk</div>'
        "</div>"
    )
    rows.append("</div>")
    return "\n".join(rows)
