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
# Sidebar — branding + settings
# ---------------------------------------------------------------------------
with st.sidebar:
    if os.path.exists(LOGO_PATH):
        with open(LOGO_PATH, "r", encoding="utf-8") as _f:
            st.markdown(_f.read(), unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Settings")
    confidence_threshold = st.slider(
        "Confidence threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence required to trigger an alert. "
             "Predictions below this are labelled 'Uncertain'.",
    )
    st.markdown("---")
    st.caption("Emergency Response: Road Incident Detection")

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
        _msg.success("Model loaded successfully!")
        time.sleep(1.5)
        _msg.empty()
    except Exception as exc:
        st.warning(f"Could not load model — running in demo mode.\n\nError: {exc}")

# ---------------------------------------------------------------------------
# Load location data
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner=False)
def _load_locations(path: str):
    return load_locations(path)


try:
    location_data = _load_locations(LOCATIONS_PATH)
except FileNotFoundError:
    st.error(f"locations.json not found at: {LOCATIONS_PATH}")
    st.stop()

# ---------------------------------------------------------------------------
# Session state — incident history
# ---------------------------------------------------------------------------
if "history" not in st.session_state:
    st.session_state.history: list[dict] = []

# ---------------------------------------------------------------------------
# Main UI
# ---------------------------------------------------------------------------
st.title("Emergency Response: Road Incident Detection")
st.markdown(
    "Upload a road camera image to classify traffic conditions and, when an "
    "accident is detected, surface nearest emergency services automatically."
)

col_left, col_right = st.columns([1, 1], gap="large")

with col_left:
    # Area + CCTV selector
    selected_area = st.selectbox("Select Major Area", list(location_data["areas"].keys()))
    area_info = location_data["areas"][selected_area]

    sub_location_names = [loc["name"] for loc in area_info["sub_locations"]]
    selected_sub_name = st.selectbox("Select Sub-Location / CCTV Point", sub_location_names)
    selected_sub = next(
        loc for loc in area_info["sub_locations"] if loc["name"] == selected_sub_name
    )

    # Image uploader
    uploaded_file = st.file_uploader(
        "Upload a road image", type=["jpg", "jpeg", "png"], label_visibility="visible"
    )

with col_right:
    if uploaded_file:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

# ---------------------------------------------------------------------------
# Inference + dispatch
# ---------------------------------------------------------------------------
if uploaded_file:
    with st.spinner("Analysing image…"):
        try:
            img_array = preprocess_image(uploaded_file)
        except ValueError as exc:
            st.error(f"Could not process image: {exc}")
            st.stop()

        label, probs = predict_image(img_array, model, threshold=confidence_threshold)

    st.subheader(f"Result: {label}")

    # Confidence bars
    st.markdown(build_confidence_html(LABELS, probs), unsafe_allow_html=True)
    st.markdown("")

    # Dispatch
    dispatch = get_dispatch_info(label, area_info, location_data["general_emergency_hotlines"])

    if dispatch:
        st.error("⚠️ Accident detected! Dispatching emergency services…")

        info_col, map_col = st.columns([1, 1], gap="large")
        with info_col:
            st.markdown(
                f"**Nearest Hospital:** {dispatch['hospital']['name']}  \n"
                f"📞 {dispatch['hospital']['phone']}"
            )
            st.markdown(
                f"**Nearest Police Station:** {dispatch['police']['name']}  \n"
                f"📞 {dispatch['police']['phone']}"
            )
            st.markdown("**General Emergency Hotlines**")
            hotlines = dispatch["hotlines"]
            st.markdown(f"- Police Control Room: {hotlines['police_control_room']}")
            for amb in hotlines["ambulance_services"]:
                st.markdown(f"- {amb}")

        with map_col:
            map_center = [selected_sub["lat"], selected_sub["lon"]]
            m = folium.Map(location=map_center, zoom_start=14)
            folium.Marker(
                map_center,
                popup=f"CCTV: {selected_sub_name}",
                icon=folium.Icon(color="red", icon="camera"),
            ).add_to(m)
            folium.Marker(
                [dispatch["hospital"]["lat"], dispatch["hospital"]["lon"]],
                popup=f"Hospital: {dispatch['hospital']['name']}",
                icon=folium.Icon(color="green", icon="plus-sign"),
            ).add_to(m)
            folium.Marker(
                [dispatch["police"]["lat"], dispatch["police"]["lon"]],
                popup=f"Police: {dispatch['police']['name']}",
                icon=folium.Icon(color="blue", icon="info-sign"),
            ).add_to(m)
            folium.PolyLine(
                [[dispatch["hospital"]["lat"], dispatch["hospital"]["lon"]], map_center],
                color="green", weight=3, opacity=0.8,
            ).add_to(m)
            folium.PolyLine(
                [[dispatch["police"]["lat"], dispatch["police"]["lon"]], map_center],
                color="blue", weight=3, opacity=0.8,
            ).add_to(m)
            st.subheader("Dispatch Map")
            st_folium(m, width=None, height=420, returned_objects=[])
    elif label == "Uncertain":
        st.warning(
            f"Confidence below threshold ({confidence_threshold:.0%}). "
            "Adjust the slider in the sidebar or upload a clearer image."
        )
    else:
        st.info("No emergency dispatch required.")

    # Record in incident history
    st.session_state.history.append(
        {
            "File": uploaded_file.name,
            "Area": selected_area,
            "CCTV Point": selected_sub_name,
            "Prediction": label,
            "Confidence": f"{float(np.max(probs)):.1%}",
        }
    )

# ---------------------------------------------------------------------------
# Incident history
# ---------------------------------------------------------------------------
if st.session_state.history:
    with st.expander("Incident History (this session)", expanded=False):
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, use_container_width=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Download CSV", csv, "incident_history.csv", "text/csv")
