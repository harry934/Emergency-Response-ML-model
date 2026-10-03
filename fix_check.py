APP_CODE = r"""import os
import time
import math

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

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
LOGO_PATH = os.path.join(BASE_DIR, "assets", "logo.svg")
MODEL_PATH = os.path.join(BASE_DIR, "..", "model.keras")
LOCATIONS_PATH = os.path.join(BASE_DIR, "..", "locations.json")

st.set_page_config(
    page_title="Emergency Response | Road Incident Detection",
    page_icon="\U0001f6a8",
    layout="wide",
    initial_sidebar_state="expanded",
)
"""

with open(r"C:\Users\kibag\Desktop\Github Projects\Emergency-Responce\interface\app_test.py", "w", encoding="utf-8") as f:
    f.write(APP_CODE)
print("OK")
