import os
import time

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from core.dispatcher import get_dispatch_info
from core.location_loader import load_locations
from core.predictor import LABELS, load_model, predict_image
from core.preprocessor import preprocess_image
from core.render import build_confidence_html

# ---------------------------------------------------------------------------
# Page configuration (must be the first Streamlit call)
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.svg")
MODEL_PATH = os.path.join(BASE_DIR, "..", "model.keras")
LOCATIONS_PATH = os.path.join(BASE_DIR, "..", "locations.json")

st.set_page_config(
    page_title="Emergency Response | Road Incident Detection",
    page_icon="🚨",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Global CSS -- dark theme with emergency red/blue accents
# ---------------------------------------------------------------------------
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
:root {
    --bg-primary:   #0d1117;
    --bg-secondary: #161b22;
    --bg-card:      #1c2230;
    --bg-card-alt:  #1a1f2e;
    --border:       #2d3650;
    --accent-red:   #e53935;
    --accent-cyan:  #00b4d8;
    --text-primary: #e6edf3;
    --text-muted:   #8b949e;
    --success:      #00c853;
    --radius-md:    10px;
    --radius-sm:    6px;
}
.stApp, .main .block-container {
    background: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important;
    color: var(--text-primary) !important;
}
.main .block-container { padding-top: 1.5rem !important; max-width: 1400px !important; }
section[data-testid="stSidebar"] {
    background: var(--bg-secondary) !important;
    border-right: 1px solid var(--border) !important;
}
section[data-testid="stSidebar"] * {
    color: var(--text-primary) !important;
    font-family: 'Inter', sans-serif !important;
}
div[data-baseweb="select"] > div, div[data-baseweb="input"] > div {
    background: var(--bg-card) !important;
    border: 1px solid var(--border) !important;
    border-radius: var(--radius-sm) !important;
    color: var(--text-primary) !important;
}
div[data-baseweb="popover"] ul { background: var(--bg-card) !important; border: 1px solid var(--border) !important; }
div[data-baseweb="popover"] li { color: var(--text-primary) !important; }
div[data-baseweb="popover"] li:hover { background: var(--bg-card-alt) !important; }
div[data-baseweb="select"] * { color: var(--text-primary) !important; }
div[data-testid="stFileUploader"] {
    border: 2px dashed var(--border) !important;
    border-radius: var(--radius-md) !important;
    background: var(--bg-card) !important;
    transition: border-color 0.3s ease;
}
div[data-testid="stFileUploader"]:hover { border-color: var(--accent-cyan) !important; }
div[data-testid="stFileUploader"] * { color: var(--text-primary) !important; }
div[data-testid="stFileUploader"] button {
    background: var(--bg-card-alt) !important;
    border: 1px solid var(--border) !important;
    color: var(--text-primary) !important;
}
div[data-testid="stAlert"] { border-radius: var(--radius-md) !important; border: 1px solid var(--border) !important; }
.stButton > button {
    background: linear-gradient(135deg, var(--accent-red) 0%, #c62828 100%) !important;
    color: #fff !important; border: none !important;
    border-radius: var(--radius-sm) !important; font-weight: 600 !important;
    padding: 0.5rem 1.5rem !important; transition: all 0.2s ease !important;
    box-shadow: 0 2px 8px rgba(229,57,53,0.3) !important;
}
.stButton > button:hover {
    transform: translateY(-1px) !important;
    box-shadow: 0 4px 16px rgba(229,57,53,0.45) !important;
}
.stDownloadButton > button {
    background: var(--bg-card) !important;
    color: var(--accent-cyan) !important;
    border: 1px solid var(--accent-cyan) !important;
    border-radius: var(--radius-sm) !important; font-weight: 600 !important;
}
.stDownloadButton > button:hover { background: rgba(0,180,216,0.1) !important; }
div[data-testid="stExpander"] {
    background: var(--bg-card) !important; border: 1px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
}
div[data-testid="stExpander"] summary { color: var(--text-primary) !important; font-weight: 600 !important; }
div[data-testid="stImage"] img { border-radius: var(--radius-md) !important; border: 1px solid var(--border) !important; }
hr { border-color: var(--border) !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--accent-cyan); }
iframe { border-radius: var(--radius-md) !important; border: 1px solid var(--border) !important; }
h1,h2,h3,h4,h5,h6,p,label,span { font-family: 'Inter', sans-serif !important; }
@keyframes glow-pulse {
  0%,100% { box-shadow: 0 0 20px rgba(229,57,53,0.2); }
  50%      { box-shadow: 0 0 40px rgba(229,57,53,0.5); }
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("""
    <div style="padding:1rem 0 0.5rem 0; text-align:center;">
        <div style="
            width:60px; height:60px; border-radius:50%;
            background:linear-gradient(135deg,#e53935,#1565c0);
            display:flex; align-items:center; justify-content:center;
            margin:0 auto 0.75rem auto;
            box-shadow: 0 0 24px rgba(229,57,53,0.4);
            font-size:1.6rem;">
            🚨
        </div>
        <div style="font-weight:800;font-size:1rem;color:#e6edf3;letter-spacing:0.02em;">Emergency Response</div>
        <div style="font-size:0.72rem;color:#8b949e;margin-top:2px;letter-spacing:0.04em;">Road Incident Detection</div>
    </div>
    <hr style="border-color:#2d3650;margin:0.75rem 0;">
    """, unsafe_allow_html=True)

    st.markdown(
        "<div style='font-size:0.72rem;color:#8b949e;text-transform:uppercase;"
        "letter-spacing:0.08em;font-weight:600;margin-bottom:0.5rem;'>"
        "⚙️ Detection Settings</div>",
        unsafe_allow_html=True,
    )

    confidence_threshold = st.slider(
        "Confidence Threshold",
        min_value=0.0, max_value=1.0, value=0.5, step=0.05,
        help="Minimum confidence to trigger an alert. Below this = Uncertain.",
    )

    st.markdown(
        f"<div style='background:#1c2230;border:1px solid #2d3650;border-radius:8px;"
        f"padding:0.6rem 0.75rem;margin-top:0.5rem;font-size:0.8rem;color:#8b949e;'>"
        f"<span style='color:#00b4d8;font-weight:600;'>Active threshold:</span>"
        f"<span style='color:#e6edf3;font-weight:700;float:right;'>{confidence_threshold:.0%}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr style='border-color:#2d3650;margin:1rem 0;'>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:0.72rem;color:#8b949e;text-transform:uppercase;
        letter-spacing:0.08em;font-weight:600;margin-bottom:0.6rem;">
        📡 System Status
    </div>
    <div style="display:flex;flex-direction:column;gap:6px;">
        <div style="display:flex;align-items:center;gap:8px;font-size:0.82rem;color:#e6edf3;">
            <div style="width:8px;height:8px;border-radius:50%;background:#00c853;box-shadow:0 0 6px #00c853;"></div>
            ML Model
        </div>
        <div style="display:flex;align-items:center;gap:8px;font-size:0.82rem;color:#e6edf3;">
            <div style="width:8px;height:8px;border-radius:50%;background:#00c853;box-shadow:0 0 6px #00c853;"></div>
            Location Database
        </div>
        <div style="display:flex;align-items:center;gap:8px;font-size:0.82rem;color:#e6edf3;">
            <div style="width:8px;height:8px;border-radius:50%;background:#00c853;box-shadow:0 0 6px #00c853;"></div>
            Dispatch System
        </div>
    </div>
    <hr style="border-color:#2d3650;margin:1rem 0;">
    <div style="font-size:0.72rem;color:#8b949e;text-align:center;line-height:1.6;">
        Emergency Response Dashboard<br>v2.0 &bull; Road Incident Detection
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load model
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_model(path: str):
    return load_model(path)

model = None
with st.spinner("Loading model…"):
    try:
        model = _load_model(MODEL_PATH)
        _msg = st.empty()
        _msg.success("✅ Model loaded successfully!")
        time.sleep(1.5)
        _msg.empty()
    except Exception as exc:
        st.warning(f"⚠️ Running in demo mode — model unavailable.\n\n`{exc}`")

# ---------------------------------------------------------------------------
# Load location data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_locations(path: str):
    return load_locations(path)

try:
    location_data = _load_locations(LOCATIONS_PATH)
except FileNotFoundError:
    st.error(f"📂 `locations.json` not found at: `{LOCATIONS_PATH}`")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history: list[dict] = []

# ---------------------------------------------------------------------------
# Hero Header
# ---------------------------------------------------------------------------
st.markdown("""
<div style="
    background: linear-gradient(135deg, #161b22 0%, #1c2230 50%, #0d1117 100%);
    border: 1px solid #2d3650; border-radius: 16px;
    padding: 2rem 2.5rem; margin-bottom: 1.5rem;
    position: relative; overflow: hidden;">
    <div style="
        position:absolute; top:-40px; right:-40px;
        width:200px; height:200px; border-radius:50%;
        background:radial-gradient(circle, rgba(229,57,53,0.12) 0%, transparent 70%);
        pointer-events:none;"></div>
    <div style="
        position:absolute; bottom:-60px; left:30%;
        width:250px; height:250px; border-radius:50%;
        background:radial-gradient(circle, rgba(21,101,192,0.08) 0%, transparent 70%);
        pointer-events:none;"></div>
    <div style="display:flex; align-items:center; gap:1.25rem; position:relative;">
        <div style="font-size:3rem; line-height:1;
            filter: drop-shadow(0 0 12px rgba(229,57,53,0.6));">🚨</div>
        <div>
            <div style="font-size:1.75rem;font-weight:800;color:#e6edf3;
                letter-spacing:-0.01em;line-height:1.2;">
                Emergency Response
                <span style="background:linear-gradient(90deg,#e53935,#00b4d8);
                    -webkit-background-clip:text;-webkit-text-fill-color:transparent;
                    background-clip:text;"> Dashboard</span>
            </div>
            <div style="font-size:0.9rem;color:#8b949e;margin-top:4px;">
                Road Incident Detection &amp; Emergency Dispatch System
            </div>
        </div>
        <div style="margin-left:auto;display:flex;gap:0.5rem;flex-wrap:wrap;">
            <div style="background:rgba(0,200,83,0.12);border:1px solid rgba(0,200,83,0.3);
                border-radius:20px;padding:4px 12px;font-size:0.75rem;font-weight:600;color:#00c853;
                display:flex;align-items:center;gap:5px;">
                <div style="width:6px;height:6px;border-radius:50%;background:#00c853;
                    box-shadow:0 0 5px #00c853;"></div>LIVE
            </div>
            <div style="background:rgba(0,180,216,0.12);border:1px solid rgba(0,180,216,0.3);
                border-radius:20px;padding:4px 12px;font-size:0.75rem;font-weight:600;color:#00b4d8;">
                AI-Powered
            </div>
        </div>
    </div>
    <div style="display:flex;gap:1.5rem;margin-top:1.25rem;flex-wrap:wrap;
        padding-top:1.25rem;border-top:1px solid #2d3650;">
        <div style="display:flex;align-items:center;gap:6px;font-size:0.8rem;color:#8b949e;">
            <span style="color:#e53935;">●</span> Accident Detection
        </div>
        <div style="display:flex;align-items:center;gap:6px;font-size:0.8rem;color:#8b949e;">
            <span style="color:#ff9800;">●</span> Traffic Analysis
        </div>
        <div style="display:flex;align-items:center;gap:6px;font-size:0.8rem;color:#8b949e;">
            <span style="color:#00b4d8;">●</span> Auto Dispatch
        </div>
        <div style="display:flex;align-items:center;gap:6px;font-size:0.8rem;color:#8b949e;">
            <span style="color:#00c853;">●</span> Location Mapping
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    st.markdown("""
    <div style="background:#1c2230;border:1px solid #2d3650;border-radius:12px;
        padding:1rem 1.25rem 0.25rem 1.25rem;margin-bottom:0.75rem;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:0.5rem;">
            <span>📍</span>
            <span style="font-weight:700;font-size:0.95rem;color:#e6edf3;">Location Selection</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    selected_area = st.selectbox("Select Major Area", list(location_data["areas"].keys()))
    area_info = location_data["areas"][selected_area]

    sub_location_names = [loc["name"] for loc in area_info["sub_locations"]]
    selected_sub_name = st.selectbox("Select Sub-Location / CCTV Point", sub_location_names)
    selected_sub = next(loc for loc in area_info["sub_locations"] if loc["name"] == selected_sub_name)

    st.markdown(
        f"<div style='background:#161b22;border:1px solid #2d3650;border-radius:8px;"
        f"padding:0.5rem 0.75rem;margin:0.25rem 0 0.75rem 0;"
        f"font-size:0.79rem;color:#8b949e;display:flex;gap:1.5rem;flex-wrap:wrap;'>"
        f"<span><span style='color:#00b4d8;font-weight:600;'>LAT:</span> {selected_sub.get('lat', 'N/A')}</span>"
        f"<span><span style='color:#00b4d8;font-weight:600;'>LON:</span> {selected_sub.get('lon', 'N/A')}</span>"
        f"<span><span style='color:#00b4d8;font-weight:600;'>CCTV:</span> {selected_sub_name}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    st.markdown("""
    <div style="background:#1c2230;border:1px solid #2d3650;border-radius:12px;
        padding:1rem 1.25rem 0.25rem 1.25rem;margin-bottom:0.5rem;">
        <div style="display:flex;align-items:center;gap:8px;margin-bottom:0.4rem;">
            <span>📷</span>
            <span style="font-weight:700;font-size:0.95rem;color:#e6edf3;">Upload Road Image</span>
            <span style="margin-left:auto;font-size:0.7rem;color:#8b949e;background:#0d1117;
                border:1px solid #2d3650;border-radius:4px;padding:2px 6px;">JPG · JPEG · PNG</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "Upload a road image", type=["jpg", "jpeg", "png"], label_visibility="collapsed"
    )

with col_right:
    if uploaded_file:
        st.markdown(
            f"<div style='background:#1c2230;border:1px solid #2d3650;border-radius:12px;"
            f"padding:0.75rem 1rem 0.25rem 1rem;margin-bottom:0.5rem;'>"
            f"<div style='display:flex;align-items:center;gap:8px;'>"
            f"<span>🖼️</span>"
            f"<span style='font-weight:700;font-size:0.9rem;color:#e6edf3;'>Preview</span>"
            f"<span style='margin-left:auto;font-size:0.75rem;color:#8b949e;'>{uploaded_file.name}</span>"
            f"</div></div>",
            unsafe_allow_html=True,
        )
        st.image(uploaded_file, caption="", use_container_width=True)
    else:
        st.markdown("""
        <div style="background:#1c2230;border:2px dashed #2d3650;border-radius:12px;
            padding:3rem 2rem;text-align:center;
            display:flex;flex-direction:column;align-items:center;justify-content:center;
            min-height:260px;">
            <div style="font-size:2.5rem;margin-bottom:0.75rem;opacity:0.3;">📷</div>
            <div style="color:#8b949e;font-size:0.85rem;font-weight:500;">No image uploaded yet</div>
            <div style="color:#2d3650;font-size:0.75rem;margin-top:4px;">
                Upload a road/CCTV image to begin analysis
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Inference + dispatch
# ---------------------------------------------------------------------------
if uploaded_file:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

    st.markdown("""
    <div style="background:#1c2230;border:1px solid #2d3650;border-radius:12px;
        padding:1rem 1.25rem 0.5rem 1.25rem;margin-bottom:0.75rem;">
        <div style="display:flex;align-items:center;gap:8px;">
            <span>🧠</span>
            <span style="font-weight:700;font-size:0.95rem;color:#e6edf3;">AI Analysis</span>
            <span style="margin-left:auto;font-size:0.7rem;color:#00b4d8;
                background:rgba(0,180,216,0.1);border:1px solid rgba(0,180,216,0.3);
                border-radius:12px;padding:2px 8px;font-weight:600;">Deep Learning Model</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.spinner("🔍 Analysing image with AI model…"):
        try:
            img_array = preprocess_image(uploaded_file)
        except ValueError as exc:
            st.error(f"❌ Could not process image: {exc}")
            st.stop()
        label, probs = predict_image(img_array, model, threshold=confidence_threshold)

    LABEL_COLORS = {
        "Accident":           ("#e53935", "rgba(229,57,53,0.18)",   "🚨"),
        "HeavyTraffic":       ("#ff9800", "rgba(255,152,0,0.18)",   "🚦"),
        "NormalRoadActivity": ("#00c853", "rgba(0,200,83,0.18)",    "✅"),
        "Uncertain":          ("#8b949e", "rgba(139,148,158,0.18)", "❓"),
    }
    lc_col, lc_glow, lc_icon = LABEL_COLORS.get(label, ("#8b949e", "rgba(139,148,158,0.18)", "❓"))
    display_label = label.replace("HeavyTraffic", "Heavy Traffic").replace("NormalRoadActivity", "Normal Activity")
    top_conf = float(np.max(probs))

    st.markdown(
        f"<div style='background:linear-gradient(135deg,{lc_glow} 0%,rgba(13,17,23,0.8) 100%);"
        f"border:1px solid {lc_col}44;border-radius:12px;padding:1.25rem 1.5rem;margin-bottom:1rem;"
        f"display:flex;align-items:center;gap:1rem;box-shadow:0 0 24px {lc_glow};'>"
        f"<div style='font-size:2.5rem;filter:drop-shadow(0 0 10px {lc_col}80);'>{lc_icon}</div>"
        f"<div>"
        f"<div style='font-size:0.68rem;color:#8b949e;text-transform:uppercase;"
        f"letter-spacing:0.1em;font-weight:600;'>Detection Result</div>"
        f"<div style='font-size:1.5rem;font-weight:800;color:{lc_col};margin-top:2px;'>{display_label}</div>"
        f"</div>"
        f"<div style='margin-left:auto;background:{lc_col}18;border:1px solid {lc_col}44;"
        f"border-radius:10px;padding:0.6rem 1rem;text-align:center;min-width:80px;'>"
        f"<div style='font-size:1.5rem;font-weight:800;color:{lc_col};font-family:monospace;'>{top_conf:.0%}</div>"
        f"<div style='font-size:0.65rem;color:#8b949e;font-weight:600;text-transform:uppercase;"
        f"letter-spacing:0.05em;'>Confidence</div>"
        f"</div></div>",
        unsafe_allow_html=True,
    )

    st.markdown(build_confidence_html(LABELS, probs), unsafe_allow_html=True)
    st.markdown("<div style='height:0.75rem'></div>", unsafe_allow_html=True)

    # Dispatch
    dispatch = get_dispatch_info(label, area_info, location_data["general_emergency_hotlines"])

    if dispatch:
        st.markdown("""
        <div style="background:linear-gradient(135deg,rgba(229,57,53,0.14) 0%,rgba(229,57,53,0.04) 100%);
            border:1px solid rgba(229,57,53,0.4);border-radius:12px;
            padding:1rem 1.5rem;margin:0.5rem 0 1rem 0;
            display:flex;align-items:center;gap:12px;animation:glow-pulse 2s infinite;">
            <div style="font-size:2rem;">🚨</div>
            <div>
                <div style="font-weight:800;font-size:1rem;color:#ff4444;letter-spacing:0.03em;">ACCIDENT DETECTED</div>
                <div style="font-size:0.8rem;color:#e57373;margin-top:2px;">Dispatching emergency services to location</div>
            </div>
            <div style="margin-left:auto;font-size:0.75rem;color:#ff4444;font-weight:700;
                background:rgba(229,57,53,0.12);border:1px solid rgba(229,57,53,0.3);
                border-radius:6px;padding:4px 10px;white-space:nowrap;">
                📡 DISPATCHING…
            </div>
        </div>
        """, unsafe_allow_html=True)

        info_col, map_col = st.columns([1, 1], gap="large")
        with info_col:
            hotlines = dispatch["hotlines"]
            amb_lines = "".join(
                f"<div style='font-size:0.8rem;color:#e6edf3;padding:3px 0;'>🚑 {a}</div>"
                for a in hotlines["ambulance_services"]
            )
            st.markdown(
                f"<div style='display:flex;flex-direction:column;gap:0.65rem;'>"

                f"<div style='background:#1c2230;border:1px solid #2d3650;border-left:3px solid #00c853;"
                f"border-radius:10px;padding:0.9rem 1.1rem;'>"
                f"<div style='font-size:0.65rem;color:#8b949e;text-transform:uppercase;"
                f"letter-spacing:0.1em;font-weight:600;margin-bottom:4px;'>🏥 Nearest Hospital</div>"
                f"<div style='font-weight:700;font-size:0.92rem;color:#e6edf3;'>{dispatch['hospital']['name']}</div>"
                f"<div style='font-size:0.82rem;color:#00c853;margin-top:3px;'>📞 {dispatch['hospital']['phone']}</div>"
                f"</div>"

                f"<div style='background:#1c2230;border:1px solid #2d3650;border-left:3px solid #00b4d8;"
                f"border-radius:10px;padding:0.9rem 1.1rem;'>"
                f"<div style='font-size:0.65rem;color:#8b949e;text-transform:uppercase;"
                f"letter-spacing:0.1em;font-weight:600;margin-bottom:4px;'>👮 Nearest Police Station</div>"
                f"<div style='font-weight:700;font-size:0.92rem;color:#e6edf3;'>{dispatch['police']['name']}</div>"
                f"<div style='font-size:0.82rem;color:#00b4d8;margin-top:3px;'>📞 {dispatch['police']['phone']}</div>"
                f"</div>"

                f"<div style='background:#1c2230;border:1px solid #2d3650;border-left:3px solid #e53935;"
                f"border-radius:10px;padding:0.9rem 1.1rem;'>"
                f"<div style='font-size:0.65rem;color:#8b949e;text-transform:uppercase;"
                f"letter-spacing:0.1em;font-weight:600;margin-bottom:6px;'>📋 Emergency Hotlines</div>"
                f"<div style='font-size:0.8rem;color:#e6edf3;padding:3px 0;'>🚔 Police Control: "
                f"<span style='color:#00b4d8;font-weight:600;'>{hotlines['police_control_room']}</span></div>"
                f"{amb_lines}"
                f"</div>"

                f"</div>",
                unsafe_allow_html=True,
            )

        with map_col:
            st.markdown(
                "<div style='font-size:0.68rem;color:#8b949e;text-transform:uppercase;"
                "letter-spacing:0.1em;font-weight:600;margin-bottom:0.4rem;'>"
                "🗺️ Dispatch Map</div>",
                unsafe_allow_html=True,
            )
            map_center = [selected_sub["lat"], selected_sub["lon"]]
            m = folium.Map(location=map_center, zoom_start=14, tiles="CartoDB dark_matter")
            folium.Marker(
                map_center,
                popup=folium.Popup(f"<b>📷 CCTV:</b> {selected_sub_name}", max_width=200),
                icon=folium.Icon(color="red", icon="camera"),
            ).add_to(m)
            folium.Marker(
                [dispatch["hospital"]["lat"], dispatch["hospital"]["lon"]],
                popup=folium.Popup(f"<b>🏥 Hospital:</b> {dispatch['hospital']['name']}", max_width=200),
                icon=folium.Icon(color="green", icon="plus-sign"),
            ).add_to(m)
            folium.Marker(
                [dispatch["police"]["lat"], dispatch["police"]["lon"]],
                popup=folium.Popup(f"<b>👮 Police:</b> {dispatch['police']['name']}", max_width=200),
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(m)
            folium.PolyLine(
                [[dispatch["hospital"]["lat"], dispatch["hospital"]["lon"]], map_center],
                color="#00c853", weight=3, opacity=0.85, dash_array="8 4",
            ).add_to(m)
            folium.PolyLine(
                [[dispatch["police"]["lat"], dispatch["police"]["lon"]], map_center],
                color="#00b4d8", weight=3, opacity=0.85, dash_array="8 4",
            ).add_to(m)
            folium.Circle(
                map_center, radius=200,
                color="#e53935", fill=True, fill_color="#e53935", fill_opacity=0.15, weight=2,
            ).add_to(m)
            st_folium(m, width=None, height=400, returned_objects=[])

    elif label == "Uncertain":
        st.markdown(
            f"<div style='background:rgba(255,152,0,0.08);border:1px solid rgba(255,152,0,0.3);"
            f"border-radius:10px;padding:1rem 1.25rem;'>"
            f"<div style='font-weight:700;color:#ff9800;margin-bottom:4px;'>⚠️ Low Confidence</div>"
            f"<div style='font-size:0.85rem;color:#8b949e;'>Confidence below threshold ({confidence_threshold:.0%}). "
            f"Adjust the slider in the sidebar or upload a clearer image.</div>"
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown("""
        <div style="background:rgba(0,200,83,0.08);border:1px solid rgba(0,200,83,0.3);
            border-radius:10px;padding:1rem 1.25rem;">
            <div style="font-weight:700;color:#00c853;margin-bottom:4px;">✅ No Emergency Detected</div>
            <div style="font-size:0.85rem;color:#8b949e;">Road conditions are normal. No dispatch required.</div>
        </div>
        """, unsafe_allow_html=True)

    st.session_state.history.append({
        "File":        uploaded_file.name,
        "Area":        selected_area,
        "CCTV Point": selected_sub_name,
        "Prediction":  label,
        "Confidence":  f"{float(np.max(probs)):.1%}",
        "Time":        time.strftime("%H:%M:%S"),
    })

# ---------------------------------------------------------------------------
# Incident history
# ---------------------------------------------------------------------------
if st.session_state.history:
    st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)
    with st.expander("📋 Incident History (this session)", expanded=False):
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Download CSV",
            csv, "incident_history.csv", "text/csv",
            use_container_width=True,
        )
