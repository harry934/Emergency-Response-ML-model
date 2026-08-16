"""HTML rendering helpers — confidence bars and risk badges."""
from __future__ import annotations

import math

BASE_COLORS: dict[str, str] = {
    "Accident": "#e53935",
    "HeavyTraffic": "#fb8c00",
    "NormalRoadActivity": "#43a047",
}

_DISPLAY_NAMES: dict[str, str] = {
    "Accident": "Accident",
    "HeavyTraffic": "Heavy Traffic",
    "NormalRoadActivity": "Normal Activity",
}

_CSS = """
<style>
.pred-row{display:flex;align-items:center;margin:12px 0}
.pred-label{width:160px;font-weight:700;color:#222;font-size:14px}
.pred-bar{flex:1;height:22px;background:#f3f4f6;border-radius:12px;overflow:hidden;margin:0 12px;position:relative}
.pred-fill{height:100%;border-radius:12px 0 0 12px;box-shadow:0 2px 6px rgba(0,0,0,0.08);transition:width 800ms ease}
.pred-pct{width:64px;text-align:right;font-family:monospace;color:#111;font-size:13px}
.risk-badge{display:inline-flex;align-items:center;gap:8px;margin-left:10px}
.risk-dot{width:12px;height:12px;border-radius:50%}
.legend{display:flex;gap:12px;margin-top:10px;align-items:center}
.legend .item{display:flex;gap:8px;align-items:center;font-size:13px;color:#555}
.bar-inner-text{position:absolute;left:8px;top:0;bottom:0;display:flex;align-items:center;
  color:#fff;font-weight:600;font-size:12px;padding-left:6px}
</style>
"""


def _risk_level(pct: int) -> tuple[str, str]:
    """Return ``(colour_hex, label)`` for a given integer percentage."""
    if pct >= 70:
        return "#b71c1c", "High"
    if pct >= 40:
        return "#f57c00", "Medium"
    return "#2e7d32", "Low"


def build_confidence_html(labels: list[str], probs: list[float] | "np.ndarray") -> str:  # noqa: F821
    """Build an HTML string for the confidence bar visualisation.

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
    rows: list[str] = [_CSS, "<div>"]
    for lab, p in zip(labels, probs):
        pct = int(math.floor(float(p) * 100 + 0.5))  # round-half-up
        base = BASE_COLORS.get(lab, "#2196f3")
        rcolor, rlabel = _risk_level(pct)
        fill_style = f"background:linear-gradient(90deg,{base},{rcolor});width:{pct}%;"
        inner_text = f"{pct}%" if pct > 10 else ""
        display_name = _DISPLAY_NAMES.get(lab, lab)
        title = f"{lab}: {pct}% — Risk: {rlabel}"
        rows.append(
            f'<div class="pred-row" title="{title}">'
            f'<div class="pred-label">{display_name}</div>'
            f'<div class="pred-bar"><div class="pred-fill" style="{fill_style}">'
            f'<div class="bar-inner-text">{inner_text}</div></div></div>'
            f'<div class="pred-pct">{pct}%</div>'
            f'<div class="risk-badge"><div class="risk-dot" style="background:{rcolor}"></div>'
            f'<div style="color:#444;font-size:13px">{rlabel}</div></div>'
            f"</div>"
        )

    rows.append(
        '<div class="legend">'
        '<div class="item"><div class="risk-dot" style="background:#2e7d32"></div>Low</div>'
        '<div class="item"><div class="risk-dot" style="background:#f57c00"></div>Medium</div>'
        '<div class="item"><div class="risk-dot" style="background:#b71c1c"></div>High</div>'
        "</div>"
    )
    rows.append("</div>")
    return "\n".join(rows)
