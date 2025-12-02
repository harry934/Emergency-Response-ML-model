import streamlit as st
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import pandas as pd
import json
import folium
import os
from streamlit_folium import st_folium


BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(BASE_DIR, "emergency_cnn_model.keras")
LOCATION_PATH = os.path.join(BASE_DIR, "locations.json")

try:
    model = tf.keras.models.load_model(MODEL_PATH)
    st.success("Model loaded successfully!")
except Exception as e:
    st.warning(f"Could not load model. Using dummy predictions.\nError: {e}")
    model = None

# Load locations
import json
try:
    with open(LOCATION_PATH, "r") as f:
        location_data = json.load(f)
    st.success("Locations loaded successfully!")
except Exception as e:
    st.error(f"Could not load locations.json\nError: {e}")
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

    # ----------------- Confidence chart -----------------
    df = pd.DataFrame({"Label": labels, "Confidence": pred})
    st.bar_chart(df, x="Label", y="Confidence")

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
