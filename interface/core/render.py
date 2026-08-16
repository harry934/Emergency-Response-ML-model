"""UI rendering helpers — global styles and confidence display."""
from __future__ import annotations

import math

BASE_COLORS: dict[str, str] = {
    "Accident": "#c62828",
    "HeavyTraffic": "#e65100",
    "NormalRoadActivity": "#2e7d32",
}

_DISPLAY_NAMES: dict[str, str] = {
    "Accident": "Accident",
    "HeavyTraffic": "Heavy Traffic",
    "NormalRoadActivity": "Normal Activity",
    "Uncertain": "Uncertain",
}

LABEL_DISPLAY = _DISPLAY_NAMES


def get_app_styles() -> str:
    """Return global CSS for a clean, neutral application layout."""
    return """
    <style>
      .block-container { padding-top: 1.5rem; max-width: 1100px; }
      [data-testid="stSidebar"] { background-color: #f8f9fa; }
      [data-testid="stSidebar"] .block-container { padding-top: 1.25rem; }
      .app-header { margin-bottom: 1.5rem; }
      .app-title {
        font-size: 1.75rem;
        font-weight: 600;
        color: #1a1a1a;
        margin: 0 0 0.25rem 0;
        letter-spacing: -0.02em;
      }
      .app-subtitle {
        font-size: 0.95rem;
        color: #5f6368;
        margin: 0;
        line-height: 1.5;
      }
      .section-label {
        font-size: 0.75rem;
        font-weight: 600;
        letter-spacing: 0.04em;
        text-transform: uppercase;
        color: #5f6368;
        margin: 0 0 0.75rem 0;
      }
      .status-accident {
        padding: 0.875rem 1rem;
        border-left: 4px solid #c62828;
        background: #fce8e6;
        color: #3c4043;
        margin: 1rem 0;
        font-size: 0.95rem;
      }
      .status-normal {
        padding: 0.875rem 1rem;
        border-left: 4px solid #2e7d32;
        background: #e8f5e9;
        color: #3c4043;
        margin: 1rem 0;
        font-size: 0.95rem;
      }
      .status-uncertain {
        padding: 0.875rem 1rem;
        border-left: 4px solid #e65100;
        background: #fff3e0;
        color: #3c4043;
        margin: 1rem 0;
        font-size: 0.95rem;
      }
      .contact-block { margin-bottom: 1rem; }
      .contact-block h4 {
        font-size: 0.8rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.03em;
        color: #5f6368;
        margin: 0 0 0.25rem 0;
      }
      .contact-block p {
        margin: 0;
        color: #202124;
        font-size: 0.95rem;
        line-height: 1.5;
      }
      .score-row {
        display: grid;
        grid-template-columns: 140px 1fr 48px;
        align-items: center;
        gap: 12px;
        margin-bottom: 10px;
      }
      .score-label { font-size: 0.875rem; color: #3c4043; }
      .score-track {
        height: 8px;
        background: #e8eaed;
        border-radius: 2px;
        overflow: hidden;
      }
      .score-fill { height: 100%; border-radius: 2px; }
      .score-pct {
        font-size: 0.8rem;
        color: #5f6368;
        text-align: right;
        font-variant-numeric: tabular-nums;
      }
    </style>
    """


def format_label(label: str) -> str:
    """Return a human-readable label for a prediction class."""
    return _DISPLAY_NAMES.get(label, label)


def build_confidence_html(labels: list[str], probs: list[float] | "np.ndarray") -> str:  # noqa: F821
    """Build a minimal HTML confidence breakdown (no gradients or badges)."""
    rows: list[str] = ['<div class="confidence-scores">']
    for lab, p in zip(labels, probs):
        pct = int(math.floor(float(p) * 100 + 0.5))
        color = BASE_COLORS.get(lab, "#5f6368")
        name = _DISPLAY_NAMES.get(lab, lab)
        rows.append(
            f'<div class="score-row">'
            f'<span class="score-label">{name}</span>'
            f'<div class="score-track"><div class="score-fill" '
            f'style="width:{pct}%;background:{color}"></div></div>'
            f'<span class="score-pct">{pct}%</span>'
            f"</div>"
        )
    rows.append("</div>")
    return "\n".join(rows)
