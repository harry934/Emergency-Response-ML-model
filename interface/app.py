import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pandas as pd
import json
import folium
import os
from streamlit_folium import st_folium
import time


# Get folder of current script
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build path to model in parent folder
MODEL_PATH = os.path.join(BASE_DIR, "..", "model.keras")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    _mdl_msg = st.empty()
    _mdl_msg.success("Model loaded successfully!")
    time.sleep(2)
    _mdl_msg.empty()
except Exception as e:
    st.warning(f"Could not load model. Using dummy predictions.\nError: {e}")
    model = None  # fallback

labels = ["Accident", "HeavyTraffic", "NormalRoadActivity"]

# ----------------- Load location data (fixed path) -----------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))  
file_path = os.path.join(BASE_DIR, "..", "locations.json")  

try:
    with open(file_path, "r") as f:
        location_data = json.load(f)
    _loc_msg = st.empty()
    _loc_msg.success("Location data loaded successfully!")
    time.sleep(2)
    _loc_msg.empty()
except FileNotFoundError:
    st.error(f" locations.json not found.\nTried path: {file_path}")
    st.stop()

st.title("Emergency Response: Road Incident Detection")

# ----------------- Select area and sub-location -----------------
selected_area = st.selectbox("Select Major Area", list(location_data["areas"].keys()))
area_info = location_data["areas"][selected_area]

sub_locations = [loc["name"] for loc in area_info["sub_locations"]]
selected_sub_location_name = st.selectbox("Select Sub-Location / CCTV Point", sub_locations)
selected_sub_location = next(loc for loc in area_info["sub_locations"] if loc["name"] == selected_sub_location_name)

# ----------------- Upload road image -----------------
uploaded_file = st.file_uploader("Upload a road image...", type=["jpg", "jpeg", "png"])
if uploaded_file:
    st.image(uploaded_file, caption="Uploaded Image", use_container_width=True)

    # Preprocess image
    img = image.load_img(uploaded_file, target_size=(224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, 0)

    # ----------------- Predict -----------------
    if model:
        pred = model.predict(img_array)[0]
    else:
        pred = np.random.rand(3)
        pred = pred / pred.sum()

    final = labels[np.argmax(pred)]
    st.subheader(f"Result: {final}")

    # ----------------- Confidence chart (styled progress bars) -----------------
    def _render_confidence_bars(labels, preds):
        # preds: array-like floats 0..1 — renders animated, styled bars with risk badges
        css = """
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
        .bar-inner-text{position:absolute;left:8px;top:0;bottom:0;display:flex;align-items:center;color:#fff;font-weight:600;font-size:12px;padding-left:6px}
        </style>
        """

        # base colors per label
        base_colors = {
            "Accident": "#e53935",
            "HeavyTraffic": "#fb8c00",
            "NormalRoadActivity": "#43a047",
        }

        def risk_level_color(pct):
            if pct >= 70:
                return "#b71c1c", "High"
            if pct >= 40:
                return "#f57c00", "Medium"
            return "#2e7d32", "Low"

        rows = [css, "<div>"]
        for lab, p in zip(labels, preds):
            pct = int(round(float(p) * 100))
            base = base_colors.get(lab, "#2196f3")
            # compute risk color and label
            rcolor, rlabel = risk_level_color(pct)
            # use a subtle gradient for fill
            fill_style = f"background: linear-gradient(90deg, {base}, {rcolor}); width:{pct}%;"
            # show percentage inside the bar when enough space (pct > 10)
            inner_text = f"{pct}%" if pct > 10 else ""
            # accessible tooltip via title
            title = f"{lab}: {pct}% — Risk: {rlabel}"
            rows.append(
                f'<div class="pred-row" title="{title}">'
                f'<div class="pred-label">{lab.replace("NormalRoadActivity","Normal Activity")}</div>'
                f'<div class="pred-bar"><div class="pred-fill" style="{fill_style}"><div class="bar-inner-text">{inner_text}</div></div></div>'
                f'<div class="pred-pct">{pct}%</div>'
                f'<div class="risk-badge"><div class="risk-dot" style="background:{rcolor}"></div><div style="color:#444;font-size:13px">{rlabel}</div></div>'
                f'</div>'
            )

        # legend
        rows.append('<div class="legend"><div class="item"><div class="risk-dot" style="background:#2e7d32"></div>Low</div><div class="item"><div class="risk-dot" style="background:#f57c00"></div>Medium</div><div class="item"><div class="risk-dot" style="background:#b71c1c"></div>High</div></div>')

        rows.append("</div>")
        html = "\n".join(rows)
        st.markdown(html, unsafe_allow_html=True)

    _render_confidence_bars(labels, pred)

    # ----------------- Dispatch info + map -----------------
    if final == "Accident":
        st.error("⚠️ Accident detected! Dispatching emergency services...")

        st.write(f"Nearest Hospital: **{area_info['hospital']['name']}** — Call: {area_info['hospital']['phone']}")
        st.write(f"Nearest Police Station: **{area_info['police']['name']}** — Call: {area_info['police']['phone']}")
        st.write("General emergency hotlines:")
        st.write(f"- Police Control Room: {location_data['general_emergency_hotlines']['police_control_room']}")
        st.write(f"- Ambulance Services: {', '.join(location_data['general_emergency_hotlines']['ambulance_services'])}")

        # ----------------- Display map -----------------
        map_center = [selected_sub_location["lat"], selected_sub_location["lon"]]
        m = folium.Map(location=map_center, zoom_start=14)

        # CCTV marker
        folium.Marker(
            location=map_center,
            popup=f"CCTV: {selected_sub_location_name}",
            icon=folium.Icon(color="red", icon="camera")
        ).add_to(m)

        # Hospital marker
        folium.Marker(
            location=[area_info["hospital"]["lat"], area_info["hospital"]["lon"]],
            popup=f"Hospital: {area_info['hospital']['name']}",
            icon=folium.Icon(color="green", icon="plus-sign")
        ).add_to(m)

        # Police marker
        folium.Marker(
            location=[area_info["police"]["lat"], area_info["police"]["lon"]],
            popup=f"Police: {area_info['police']['name']}",
            icon=folium.Icon(color="blue", icon="info-sign")
        ).add_to(m)

        # Lines to CCTV
        folium.PolyLine(
            locations=[[area_info["hospital"]["lat"], area_info["hospital"]["lon"]], map_center],
            color="green", weight=3, opacity=0.8
        ).add_to(m)

        folium.PolyLine(
            locations=[[area_info["police"]["lat"], area_info["police"]["lon"]], map_center],
            color="blue", weight=3, opacity=0.8
        ).add_to(m)

        st.subheader("Map of CCTV and Nearest Emergency Units")
        st_folium(m, width=700, height=500)

    else:
        st.info("No emergency dispatch required.")
