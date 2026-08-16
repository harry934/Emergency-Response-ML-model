import os

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from core.dispatcher import get_dispatch_info
from core.location_loader import load_locations
from core.predictor import LABELS, load_model, predict_image
from core.preprocessor import preprocess_image
from core.render import build_confidence_html, build_header_html, format_label, get_app_styles

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model.keras")
LOCATIONS_PATH = os.path.join(BASE_DIR, "..", "locations.json")

st.set_page_config(
    page_title="Emergency Response",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown(get_app_styles(), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Load resources
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def _load_model(path: str):
    return load_model(path)


@st.cache_data(show_spinner=False)
def _load_locations(path: str):
    return load_locations(path)


model = None
model_ready = False
try:
    model = _load_model(MODEL_PATH)
    model_ready = True
except Exception:
    pass

try:
    location_data = _load_locations(LOCATIONS_PATH)
except FileNotFoundError:
    st.error(f"Configuration file not found: {LOCATIONS_PATH}")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history: list[dict] = []

# ---------------------------------------------------------------------------
# Header bar
# ---------------------------------------------------------------------------
st.markdown(build_header_html(model_ready), unsafe_allow_html=True)

# ---------------------------------------------------------------------------
# Settings (main area, not sidebar)
# ---------------------------------------------------------------------------
with st.expander("Settings", expanded=False):
    confidence_threshold = st.slider(
        "Alert threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Minimum confidence required before an alert is raised.",
    )
    st.caption("Predictions below this value are marked as uncertain.")

# ---------------------------------------------------------------------------
# Input
# ---------------------------------------------------------------------------
input_col, preview_col = st.columns([1, 1], gap="large")

with input_col:
    with st.container(border=True):
        st.subheader("Location")
        selected_area = st.selectbox(
            "Area",
            list(location_data["areas"].keys()),
        )
        area_info = location_data["areas"][selected_area]

        sub_location_names = [loc["name"] for loc in area_info["sub_locations"]]
        selected_sub_name = st.selectbox(
            "CCTV point",
            sub_location_names,
        )
        selected_sub = next(
            loc for loc in area_info["sub_locations"] if loc["name"] == selected_sub_name
        )

        st.subheader("Image upload")
        uploaded_file = st.file_uploader(
            "Choose a road camera image (JPG or PNG)",
            type=["jpg", "jpeg", "png"],
        )

with preview_col:
    with st.container(border=True):
        st.subheader("Preview")
        if uploaded_file:
            st.image(uploaded_file, width="stretch")
        else:
            st.caption("Upload an image to see a preview here.")

# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------
if uploaded_file:
    with st.spinner("Analysing image…"):
        try:
            img_array = preprocess_image(uploaded_file)
        except ValueError as exc:
            st.error(f"Could not process image: {exc}")
            st.stop()

        label, probs = predict_image(img_array, model, threshold=confidence_threshold)

    display_label = format_label(label)
    top_confidence = float(np.max(probs))

    st.divider()

    with st.container(border=True):
        st.subheader("Analysis result")

        metric_col1, metric_col2 = st.columns(2)
        with metric_col1:
            st.metric("Classification", display_label)
        with metric_col2:
            st.metric("Confidence", f"{top_confidence:.1%}")

        st.markdown("**Class probabilities**")
        st.markdown(build_confidence_html(LABELS, probs), unsafe_allow_html=True)

    dispatch = get_dispatch_info(label, area_info, location_data["general_emergency_hotlines"])

    if dispatch:
        st.markdown(
            '<div class="status-accident">Accident detected. Emergency dispatch recommended.</div>',
            unsafe_allow_html=True,
        )

        info_col, map_col = st.columns([1, 1], gap="large")
        with info_col:
            with st.container(border=True):
                st.subheader("Emergency contacts")

                st.markdown(
                    f"""
                    <div class="contact-block">
                      <h4>Hospital</h4>
                      <p>{dispatch['hospital']['name']}<br>{dispatch['hospital']['phone']}</p>
                    </div>
                    <div class="contact-block">
                      <h4>Police</h4>
                      <p>{dispatch['police']['name']}<br>{dispatch['police']['phone']}</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

                hotlines = dispatch["hotlines"]
                st.markdown("**General hotlines**")
                st.markdown(f"Police control: {hotlines['police_control_room']}")
                for amb in hotlines["ambulance_services"]:
                    st.markdown(f"- {amb}")

        with map_col:
            with st.container(border=True):
                st.subheader("Response map")
                map_center = [selected_sub["lat"], selected_sub["lon"]]
                m = folium.Map(location=map_center, zoom_start=14, tiles="CartoDB positron")
                folium.Marker(
                    map_center,
                    popup=f"CCTV: {selected_sub_name}",
                    tooltip="CCTV location",
                ).add_to(m)
                folium.Marker(
                    [dispatch["hospital"]["lat"], dispatch["hospital"]["lon"]],
                    popup=f"Hospital: {dispatch['hospital']['name']}",
                    tooltip="Hospital",
                    icon=folium.Icon(color="green", icon="plus"),
                ).add_to(m)
                folium.Marker(
                    [dispatch["police"]["lat"], dispatch["police"]["lon"]],
                    popup=f"Police: {dispatch['police']['name']}",
                    tooltip="Police",
                    icon=folium.Icon(color="blue", icon="info-sign"),
                ).add_to(m)
                folium.PolyLine(
                    [[dispatch["hospital"]["lat"], dispatch["hospital"]["lon"]], map_center],
                    color="#2e7d32",
                    weight=2,
                    opacity=0.7,
                ).add_to(m)
                folium.PolyLine(
                    [[dispatch["police"]["lat"], dispatch["police"]["lon"]], map_center],
                    color="#1565c0",
                    weight=2,
                    opacity=0.7,
                ).add_to(m)
                st_folium(m, width=None, height=400, returned_objects=[])

    elif label == "Uncertain":
        st.markdown(
            f'<div class="status-uncertain">'
            f"Confidence is below the alert threshold ({confidence_threshold:.0%}). "
            f"Lower the threshold or upload a clearer image."
            f"</div>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            '<div class="status-normal">No emergency dispatch required.</div>',
            unsafe_allow_html=True,
        )

    st.session_state.history.append(
        {
            "File": uploaded_file.name,
            "Area": selected_area,
            "CCTV Point": selected_sub_name,
            "Prediction": display_label,
            "Confidence": f"{top_confidence:.1%}",
        }
    )

# ---------------------------------------------------------------------------
# History
# ---------------------------------------------------------------------------
if st.session_state.history:
    st.divider()
    with st.expander("Session history", expanded=False):
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, width="stretch", hide_index=True)
        csv = df.to_csv(index=False).encode("utf-8")
        st.download_button("Export CSV", csv, "incident_history.csv", "text/csv")
