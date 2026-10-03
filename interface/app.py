import os
import time

import folium
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium

from core.dispatcher import get_dispatch_info
from core.icons import icon
from core.location_loader import load_locations
from core.predictor import LABELS, load_model, predict_image
from core.preprocessor import preprocess_image
from core.render import ICON_CSS, format_label, section_heading

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "..", "model.keras")
LOCATIONS_PATH = os.path.join(BASE_DIR, "..", "locations.json")
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.svg")
CONFIDENCE_THRESHOLD = 0.5

# Esri basemaps need no API key (CARTO tiles now do).
ESRI_TILE_ROOT = "https://server.arcgisonline.com/ArcGIS/rest/services"
ESRI_ATTRIBUTION = "Tiles &copy; Esri"
ESRI_LIGHT_LAYERS = ["World_Street_Map"]
ESRI_DARK_LAYERS = ["Canvas/World_Dark_Gray_Base", "Canvas/World_Dark_Gray_Reference"]

st.set_page_config(
    page_title="Emergency Response",
    page_icon=":material/shield:",
    layout="centered",
)
st.markdown(ICON_CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def _load_model(path: str):
    return load_model(path)


@st.cache_data(show_spinner=False)
def _load_locations(path: str):
    return load_locations(path)


def _heading(icon_name: str, title: str) -> None:
    st.markdown(section_heading(icon_name, title), unsafe_allow_html=True)


def _is_dark_theme() -> bool:
    theme = getattr(st.context, "theme", None)
    return getattr(theme, "type", None) == "dark"


model = None
try:
    model = _load_model(MODEL_PATH)
except Exception:
    st.warning("Model could not be loaded. Results are simulated.")

try:
    location_data = _load_locations(LOCATIONS_PATH)
except FileNotFoundError:
    st.error(f"locations.json not found at: {LOCATIONS_PATH}")
    st.stop()

if "history" not in st.session_state:
    st.session_state.history = []

# Header
with open(LOGO_PATH, encoding="utf-8-sig") as _f:
    logo_svg = _f.read().strip()
logo_col, title_col = st.columns([1, 12], vertical_alignment="center")
with logo_col:
    st.markdown(f'<div class="app-logo">{logo_svg}</div>', unsafe_allow_html=True)
with title_col:
    st.title("Road Incident Detection")
st.caption(
    "Classify a road camera image as an accident, heavy traffic or normal activity, "
    "and get the nearest emergency contacts when an accident is found."
)

with st.container(border=True):
    _heading("map_pin", "Location")
    area_col, camera_col = st.columns(2)
    with area_col:
        selected_area = st.selectbox("Area", list(location_data["areas"].keys()))
    area_info = location_data["areas"][selected_area]
    with camera_col:
        sub_names = [loc["name"] for loc in area_info["sub_locations"]]
        selected_sub_name = st.selectbox("Camera", sub_names)
    selected_sub = next(loc for loc in area_info["sub_locations"] if loc["name"] == selected_sub_name)

with st.container(border=True):
    _heading("upload", "Upload image")
    uploaded_file = st.file_uploader("Road camera image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, width="stretch")

if not uploaded_file:
    st.caption("Upload an image to run the analysis.")
else:
    with st.spinner("Analysing image..."):
        try:
            img_array = preprocess_image(uploaded_file)
        except ValueError as exc:
            st.error(f"Could not process image: {exc}")
            st.stop()
        label, probs = predict_image(img_array, model, threshold=CONFIDENCE_THRESHOLD)

    top_conf = float(np.max(probs))

    with st.container(border=True):
        _heading("activity", "Result")
        if label == "Accident":
            st.error("Accident detected. Contact emergency services below.")
        elif label == "Uncertain":
            st.warning(
                f"Confidence is below the alert threshold ({CONFIDENCE_THRESHOLD:.0%}). "
                "Check the image manually."
            )
        else:
            st.success("No emergency response required.")

        class_col, conf_col = st.columns(2)
        class_col.metric("Classification", format_label(label))
        conf_col.metric("Confidence", f"{top_conf:.1%}")

        for name, p in zip(LABELS, probs):
            st.caption(f"{format_label(name)}: {float(p):.0%}")
            st.progress(float(p))

    dispatch = get_dispatch_info(label, area_info, location_data["general_emergency_hotlines"])
    if dispatch:
        with st.container(border=True):
            _heading("alert_triangle", "Emergency response")
            contacts_col, map_col = st.columns([2, 3])

            with contacts_col:
                phone = icon("phone")
                st.markdown(section_heading("hospital", "Nearest hospital"), unsafe_allow_html=True)
                st.markdown(
                    f"{dispatch['hospital']['name']}<br><span class='ico'>{phone}</span>"
                    f"{dispatch['hospital']['phone']}",
                    unsafe_allow_html=True,
                )
                st.markdown(section_heading("shield", "Nearest police station"), unsafe_allow_html=True)
                st.markdown(
                    f"{dispatch['police']['name']}<br><span class='ico'>{phone}</span>"
                    f"{dispatch['police']['phone']}",
                    unsafe_allow_html=True,
                )
                hotlines = dispatch["hotlines"]
                st.markdown("**Hotlines**")
                lines = [f"- Police control room: {hotlines['police_control_room']}"]
                lines += [f"- {a}" for a in hotlines["ambulance_services"]]
                st.markdown("\n".join(lines))

            with map_col:
                center = [selected_sub["lat"], selected_sub["lon"]]
                hospital = [dispatch["hospital"]["lat"], dispatch["hospital"]["lon"]]
                police = [dispatch["police"]["lat"], dispatch["police"]["lon"]]
                m = folium.Map(location=center, zoom_start=14, tiles=None)
                layers = ESRI_DARK_LAYERS if _is_dark_theme() else ESRI_LIGHT_LAYERS
                for layer in layers:
                    folium.TileLayer(
                        tiles=f"{ESRI_TILE_ROOT}/{layer}/MapServer/tile/{{z}}/{{y}}/{{x}}",
                        attr=ESRI_ATTRIBUTION,
                        max_zoom=16,
                    ).add_to(m)
                folium.Marker(center, tooltip=f"Camera: {selected_sub_name}",
                              icon=folium.Icon(color="red", icon="camera")).add_to(m)
                folium.Marker(hospital, tooltip=f"Hospital: {dispatch['hospital']['name']}",
                              icon=folium.Icon(color="green", icon="plus")).add_to(m)
                folium.Marker(police, tooltip=f"Police: {dispatch['police']['name']}",
                              icon=folium.Icon(color="blue", icon="info-sign")).add_to(m)
                folium.PolyLine([hospital, center], color="#2e7d32", weight=2).add_to(m)
                folium.PolyLine([police, center], color="#1565c0", weight=2).add_to(m)
                st_folium(m, width=None, height=320, returned_objects=[])

    st.session_state.history.append(
        {
            "Time": time.strftime("%H:%M:%S"),
            "File": uploaded_file.name,
            "Area": selected_area,
            "Camera": selected_sub_name,
            "Result": format_label(label),
            "Confidence": f"{top_conf:.1%}",
        }
    )

# History
if st.session_state.history:
    with st.expander("History"):
        df = pd.DataFrame(st.session_state.history)
        st.dataframe(df, width="stretch", hide_index=True)
        st.download_button(
            "Export CSV", df.to_csv(index=False).encode("utf-8"), "incident_history.csv", "text/csv"
        )
