# Emergency Response: Road Incident Detection System

A machine learning–powered web application that analyzes road camera images to classify traffic conditions and automatically dispatch emergency services when an accident is detected. Built to shorten emergency response times and improve road safety on Kenyan roads.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [System Architecture](#system-architecture)
- [ML Model](#ml-model)
- [Tech Stack](#tech-stack)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the App](#running-the-app)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Location Data](#location-data)
- [Testing](#testing)
  - [Test Strategy](#test-strategy)
  - [Running Tests](#running-tests)
  - [Test Coverage](#test-coverage)
- [Known Limitations & Roadmap](#known-limitations--roadmap)
- [Authors](#authors)

---

## Project Overview

Emergency response times on Kenyan roads are often delayed due to slow accident detection and manual reporting chains. This project addresses that gap by providing an automated image classification system that:

1. Accepts a road camera image (CCTV snapshot or uploaded photo)
2. Classifies the scene as **Accident**, **Heavy Traffic**, or **Normal Road Activity**
3. On accident detection, instantly surfaces the nearest hospital and police station with contact details and an interactive dispatch map

The system is scoped to the Nairobi metropolitan area and uses real geographic coordinates for CCTV points, hospitals, and police stations.

**Academic context:** Final year project — School of Computing and Informatics, course unit SWE 2020A. Supervisor: Edward Ombui. Authors: Harry Mokaya, Elijah Lempoko.

---

## Features

- **3-class image classifier** — Accident / Heavy Traffic / Normal Road Activity
- **Confidence visualization** — animated progress bars with colour-coded risk badges (Low / Medium / High) for each class
- **Automatic emergency dispatch** — nearest hospital and police station details surface only when an accident is detected
- **Interactive Folium map** — shows the CCTV point, hospital, and police station with connecting route lines
- **General emergency hotlines** — Kenya Police Control Room and ambulance services always available
- **Area + CCTV selector** — users pick their major area and specific CCTV sub-location before uploading
- **Graceful fallback** — if the model file cannot be loaded, the app continues with random predictions for UI demonstration purposes

---

## System Architecture

```
User uploads road image
        │
        ▼
┌───────────────────┐
│   Streamlit UI    │  ← interface/app.py
│  (area selector,  │
│  file uploader,   │
│  result display)  │
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Preprocessing    │  resize → 224×224, normalize to [0,1], expand dims
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  MobileNetV2      │  model.keras — 3-class softmax output
│  Classifier       │
└────────┬──────────┘
         │
    ┌────┴──────────┐
    │               │
 Accident      HeavyTraffic /
    │           NormalRoadActivity
    ▼               ▼
Dispatch info   "No emergency
+ Folium map    dispatch required"
```

---

## ML Model

| Property | Value |
|---|---|
| Architecture | MobileNetV2 (ImageNet pre-trained) |
| Head | GlobalAveragePooling2D → Dense(128, ReLU) → Dropout → Dense(3, Softmax) |
| Input shape | (224, 224, 3) |
| Output classes | Accident, HeavyTraffic, NormalRoadActivity |
| Parameters | ~7.2 million |
| Loss | Categorical Cross-Entropy |
| Optimizer | Adam |
| Training strategy | Freeze base (10 epochs) → fine-tune last layers (5 epochs @ lr=1e-5) |
| Model file | `model.keras` (~29.5 MB) |

**Training pipeline (not in repo):** Dataset.zip → data augmentation → MobileNetV2 transfer learning → saved as `emergency_cnn_model.keras` (Google Colab notebook). The model file is committed directly to the repository via Git LFS.

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI Framework | [Streamlit](https://streamlit.io/) |
| ML Framework | [TensorFlow / Keras](https://www.tensorflow.org/) |
| Image Processing | `tensorflow.keras.preprocessing.image`, NumPy |
| Map Rendering | [Folium](https://python-visualization.github.io/folium/) + [streamlit-folium](https://folium.streamlit.app/) |
| Data | JSON (locations.json), Pandas (incident history) |
| Testing | pytest, pytest-cov, Streamlit AppTest |
| Dev Container | GitHub Codespaces (Python 3.11) |

---

## Getting Started

### Prerequisites

- Python 3.9 or higher
- pip

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/harry934/Emergency-Response-ML-model.git
cd Emergency-Response-ML-model

# 2. Create and activate a virtual environment (recommended)
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Running the App

```bash
streamlit run interface/app.py
```

The app will open at `http://localhost:8501` in your browser.

**GitHub Codespaces:** The dev container in `.devcontainer/devcontainer.json` automatically installs dependencies and starts the Streamlit server on port 8501 when you open the repo in Codespaces.

---

## Usage

1. **Select Major Area** — choose a Nairobi area from the dropdown (e.g. "Kasarani / Roysambu / Zimmerman")
2. **Select Sub-Location / CCTV Point** — pick the specific CCTV camera location
3. **Upload a road image** — JPEG or PNG, any resolution (resized internally to 224×224)
4. The model classifies the image and displays:
   - The predicted class label
   - Confidence bars for all three classes with risk level badges
5. **If Accident is detected:**
   - Contact details for the nearest hospital and police station are shown
   - General emergency hotlines (Kenya Police, Red Cross, St. John Ambulance) are listed
   - An interactive map shows the CCTV point, hospital, and police station with route lines

---

## Project Structure

```
Emergency-Response-ML-model/
├── .devcontainer/
│   └── devcontainer.json        # GitHub Codespaces configuration
├── interface/
│   ├── app.py                   # Main Streamlit application
│   └── assets/
│       ├── logo.svg             # Project logo / branding
│       ├── safety1.svg          # "Drive Safely" safety graphic
│       └── safety2.svg          # "Fast Response" safety graphic
├── locations.json               # Nairobi CCTV, hospital, police coordinates
├── model.keras                  # Trained MobileNetV2 model (~29.5 MB)
├── requirements.txt             # Python dependencies
└── README.md
```

---

## Location Data

`locations.json` contains geographic data organized by major Nairobi areas. Each area has:

- **sub_locations** — CCTV camera points with latitude/longitude
- **hospital** — nearest hospital name, phone, and coordinates
- **police** — nearest police station name, phone, and coordinates

Currently covered areas:

| Area | CCTV Points | Hospital | Police |
|---|---|---|---|
| Kasarani / Roysambu / Zimmerman | Kasarani, Roysambu, Zimmerman | Megalife Hospital | Kasarani Police Station |
| Westlands / Parklands | Westlands, Parklands | MP Shah Hospital | Kilimani Police Station |

**General hotlines** (always shown on accident):
- Kenya Police Control Room: `020 2724154 / 0721 233999`
- Kenya Red Cross Ambulance: `020 3950000`
- St. John Ambulance: `020 2210000`

To add more areas, extend the `areas` object in `locations.json` following the existing schema.

---

## Testing

### Test Strategy

The project uses a three-tier testing approach covering unit, integration, and UI layers.

```
tests/
├── conftest.py                   # Shared fixtures (mock model, mock locations, sample images)
├── unit/
│   ├── test_preprocessor.py      # Image resize, normalize, channel handling
│   ├── test_predictor.py         # Inference output shape, label mapping, fallback mode
│   ├── test_dispatcher.py        # Dispatch dict contents per predicted class
│   ├── test_location_loader.py   # JSON loading, missing file, malformed JSON
│   └── test_confidence_bars.py   # HTML output correctness for known inputs
├── integration/
│   ├── test_model_loading.py     # Real model.keras load, input/output shape checks
│   └── test_end_to_end.py        # Full pipeline: image → preprocess → predict → dispatch
└── ui/
    └── test_app_ui.py            # Streamlit AppTest: widgets, render flow, result display
```

#### Unit Tests

| Test file | What it covers |
|---|---|
| `test_preprocessor.py` | Input image → `(1, 224, 224, 3)` shape; pixel values in `[0, 1]`; RGBA/grayscale → RGB conversion; corrupt input raises exception |
| `test_predictor.py` | Returns `(label, probs)` tuple; label is one of the 3 valid classes; probabilities sum to ~1.0; fallback (no model) returns normalized random probs |
| `test_dispatcher.py` | Accident class → dict with `hospital`, `police`, `hotlines` keys; non-accident → `None`; missing keys in area_info → `KeyError` |
| `test_location_loader.py` | Valid JSON loads correctly; missing file → `FileNotFoundError`; malformed JSON → `json.JSONDecodeError`; schema validation for required keys |
| `test_confidence_bars.py` | HTML contains all 3 labels; risk badge is "High" when confidence ≥ 70%; "Low" when < 40%; percentages sum to ~100 |

#### Integration Tests

| Test file | What it covers |
|---|---|
| `test_model_loading.py` | `model.keras` loads without error; input shape is `(None, 224, 224, 3)`; output shape is `(None, 3)`; inference on valid input returns array of length 3 |
| `test_end_to_end.py` | Real image through full pipeline produces valid class label; accident prediction triggers non-empty dispatch; non-accident produces no dispatch |

#### UI Tests (Streamlit AppTest)

| Scenario | Expected behaviour |
|---|---|
| App startup | No crash on `at.run()` |
| Area dropdown | Options match keys in `locations.json` |
| Sub-location dropdown | Updates when major area changes |
| File uploader | Present in widget tree |
| Non-accident image uploaded | `st.info("No emergency dispatch required.")` rendered |
| Accident image uploaded | `st.error` with "⚠️ Accident detected!" rendered; hospital/police info shown |
| Confidence bars | HTML markdown component rendered after image upload |

### Running Tests

```bash
# Install test dependencies
pip install pytest pytest-cov

# Run all tests
pytest

# Run with coverage report
pytest --cov=interface --cov-report=term-missing

# Run only unit tests
pytest tests/unit/

# Run only integration tests
pytest tests/integration/

# Run only UI tests
pytest tests/ui/

# Run a specific test file
pytest tests/unit/test_preprocessor.py -v
```

### Test Coverage

Target coverage for core logic modules:

| Module | Target Coverage |
|---|---|
| `interface/core/preprocessor.py` | ≥ 95% |
| `interface/core/predictor.py` | ≥ 90% |
| `interface/core/dispatcher.py` | ≥ 95% |
| `interface/core/location_loader.py` | ≥ 95% |
| `interface/app.py` (UI layer) | ≥ 70% (via AppTest) |

> **Note:** The `tests/` directory and the `interface/core/` refactored modules are part of the active development roadmap. The current version has all logic in `interface/app.py`. Tests and the module split are planned for the next sprint.

---

## Known Limitations & Roadmap

### Current Limitations

- **Image-only input** — no support for live video streams or RTSP camera feeds
- **Static dispatch** — nearest unit is based on location JSON only; no real-time unit availability
- **No incident history** — predictions are not stored between sessions
- **Two areas only** — location data covers Kasarani and Westlands only
- **No REST API** — the system is a self-contained Streamlit app; no external integrations
- **Unpinned dependencies** — `requirements.txt` has no version pins

### Roadmap

- [ ] Refactor `app.py` into testable `interface/core/` modules
- [ ] Full pytest test suite (unit + integration + UI)
- [ ] Pin dependency versions in `requirements.txt`
- [ ] Expand `locations.json` to cover all major Nairobi divisions
- [ ] Add incident history table (session-based CSV export)
- [ ] Confidence threshold slider to reduce false-positive dispatches
- [ ] Video / frame-by-frame analysis for live CCTV feeds
- [ ] REST API layer (FastAPI) for programmatic integration
- [ ] Database logging of all incidents with timestamps
- [ ] SMS/email alert integration on accident detection

---

## Authors

| Name | Role |
|---|---|
| Harry Mokaya | Developer |
| Elijah Lempoko | Developer |

**Supervisor:** Edward Ombui  
**Course:** SWE 2020A — School of Computing and Informatics

---

*Built to make Kenyan roads safer — one prediction at a time.*
