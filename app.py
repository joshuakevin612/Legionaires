"""
CardioRisk AI — Multimodal Cardiovascular Risk Dashboard
A dark-themed Streamlit application that fuses tabular vitals, ECG waveform
data, and CT scan imagery into a single composite risk score.

Run with:
    streamlit run app.py

Requires an external `ml_engine.py` module exposing:
    predict_tabular_risk(data_dict: dict) -> float
    predict_ecg_risk(file_path: str) -> tuple[float, list]
    predict_ct_risk(file_path: str) -> float
    calculate_composite_risk(p_tab, p_ecg, p_ct) -> tuple[float, str]
"""

import os
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

from ml_engine import (
    predict_tabular_risk,
    predict_ecg_risk,
    predict_ct_risk,
    calculate_composite_risk,
)

# --------------------------------------------------------------------------
# Page configuration & dark theme styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioRisk AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

DARK_CSS = """
<style>
    .stApp {
        background-color: #0e1117;
        color: #e6e6e6;
    }
    section[data-testid="stSidebar"] {
        background-color: #14181f;
        border-right: 1px solid #262b36;
    }
    div[data-testid="stMetric"] {
        background-color: #171b24;
        border: 1px solid #2a2f3a;
        border-radius: 12px;
        padding: 18px 20px;
    }
    div[data-testid="stMetricValue"] {
        color: #ffffff;
    }
    .status-badge {
        display: inline-block;
        padding: 10px 22px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.05rem;
        letter-spacing: 0.5px;
        text-align: center;
    }
    .badge-green {
        background-color: rgba(34, 197, 94, 0.15);
        color: #22c55e;
        border: 1px solid #22c55e;
    }
    .badge-yellow {
        background-color: rgba(234, 179, 8, 0.15);
        color: #eab308;
        border: 1px solid #eab308;
    }
    .badge-red {
        background-color: rgba(239, 68, 68, 0.15);
        color: #ef4444;
        border: 1px solid #ef4444;
    }
    .section-card {
        background-color: #12151c;
        border: 1px solid #232733;
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 14px;
    }
    .subtle {
        color: #9aa3b2;
        font-size: 0.88rem;
    }
    h1, h2, h3 {
        color: #f5f5f5;
    }
    div.stButton > button {
        width: 100%;
        border-radius: 10px;
        border: 1px solid #3b4252;
        background-color: #1c2029;
        color: #e6e6e6;
        font-weight: 600;
    }
    div.stButton > button:hover {
        border-color: #ef4444;
        color: #ef4444;
    }
</style>
"""
st.markdown(DARK_CSS, unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Session state defaults
# --------------------------------------------------------------------------
DEFAULTS = {
    "age": 65,
    "hypertension": 1,
    "heart_disease": 0,
    "avg_glucose_level": 140.5,
    "bmi": 28.4,
    "smoking_status": "formerly smoked",
}

DEMO_HIGH_RISK = {
    "age": 82,
    "hypertension": 1,
    "heart_disease": 1,
    "avg_glucose_level": 231.7,
    "bmi": 38.9,
    "smoking_status": "smokes",
}

for key, value in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = value


def load_demo_patient():
    """Populate session state with a canned high-risk patient profile."""
    for key, value in DEMO_HIGH_RISK.items():
        st.session_state[key] = value


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def save_upload_to_temp(uploaded_file) -> str:
    """Persist an in-memory UploadedFile to a temp path and return its path."""
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def load_waveform(file_path: str, original_name: str) -> np.ndarray:
    """Load an ECG waveform from a .csv or .npy file into a 1-D array."""
    if original_name.lower().endswith(".npy"):
        arr = np.load(file_path)
    else:
        df = pd.read_csv(file_path, header=None)
        arr = df.select_dtypes(include=[np.number]).to_numpy()
    return np.asarray(arr).flatten()


def plot_waveform(waveform: np.ndarray):
    """Render an ECG waveform as a dark-themed matplotlib line plot."""
    fig, ax = plt.subplots(figsize=(5, 2.6))
    fig.patch.set_facecolor("#12151c")
    ax.set_facecolor("#12151c")
    ax.plot(waveform, color="#22c55e", linewidth=1.1)
    ax.set_xlabel("Sample", color="#9aa3b2", fontsize=8)
    ax.set_ylabel("Amplitude", color="#9aa3b2", fontsize=8)
    ax.tick_params(colors="#9aa3b2", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#2a2f3a")
    ax.grid(alpha=0.15)
    fig.tight_layout()
    return fig


def badge_for_status(status: str) -> str:
    """Map a risk status string to a CSS badge class."""
    status_lower = status.lower()
    if "low" in status_lower:
        return "badge-green"
    if "moderate" in status_lower or "medium" in status_lower:
        return "badge-yellow"
    return "badge-red"


# --------------------------------------------------------------------------
# Sidebar — Patient Vitals
# --------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## 🩺 Patient Vitals")
    st.markdown('<p class="subtle">Baseline demographic & metabolic inputs</p>', unsafe_allow_html=True)

    st.button("⚡ Load Demo High-Risk Patient", on_click=load_demo_patient)

    st.divider()

    st.slider("Age", min_value=18, max_value=100, key="age")

    col_a, col_b = st.columns(2)
    with col_a:
        hypertension = st.toggle(
            "Hypertension",
            value=bool(st.session_state["hypertension"]),
            key="hypertension_toggle",
        )
    with col_b:
        heart_disease = st.toggle(
            "Heart Disease",
            value=bool(st.session_state["heart_disease"]),
            key="heart_disease_toggle",
        )
    st.session_state["hypertension"] = int(hypertension)
    st.session_state["heart_disease"] = int(heart_disease)

    st.number_input(
        "Avg Glucose Level (mg/dL)",
        min_value=0.0,
        max_value=500.0,
        step=0.1,
        key="avg_glucose_level",
    )
    st.number_input(
        "BMI",
        min_value=0.0,
        max_value=80.0,
        step=0.1,
        key="bmi",
    )
    st.selectbox(
        "Smoking Status",
        options=["formerly smoked", "never smoked", "smokes"],
        key="smoking_status",
    )

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.title("🫀 CardioRisk AI")
st.markdown(
    '<p class="subtle">Multimodal cardiovascular risk stratification — '
    "tabular vitals, ECG waveform, and CT imaging fused into one composite score.</p>",
    unsafe_allow_html=True,
)
st.write("")

# --------------------------------------------------------------------------
# Tabular risk (computed silently from sidebar inputs)
# --------------------------------------------------------------------------
tabular_payload = {
    "age": st.session_state["age"],
    "hypertension": st.session_state["hypertension"],
    "heart_disease": st.session_state["heart_disease"],
    "avg_glucose_level": st.session_state["avg_glucose_level"],
    "bmi": st.session_state["bmi"],
    "smoking_status": st.session_state["smoking_status"],
}

try:
    p_tab = predict_tabular_risk(tabular_payload)
except Exception as exc:  # noqa: BLE001
    st.error(f"Tabular model error: {exc}")
    p_tab = 0.0

# --------------------------------------------------------------------------
# Main body — ECG & CT columns
# --------------------------------------------------------------------------
col1, col2 = st.columns(2, gap="large")

p_ecg = 0.0
p_ct = 0.0

with col1:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("📈 ECG Waveform")
    ecg_file = st.file_uploader("Upload ECG (.csv or .npy)", type=["csv", "npy"], key="ecg_uploader")

    if ecg_file is not None:
        ecg_path = save_upload_to_temp(ecg_file)
        try:
            p_ecg, ecg_extra = predict_ecg_risk(ecg_path)
            waveform = load_waveform(ecg_path, ecg_file.name)
            st.pyplot(plot_waveform(waveform), use_container_width=True)
            st.metric("ECG Risk Score", f"{p_ecg * 100:.1f}%")
            if ecg_extra:
                with st.expander("Model details"):
                    st.write(ecg_extra)
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not process ECG file: {exc}")
        finally:
            try:
                os.remove(ecg_path)
            except OSError:
                pass
    else:
        st.info("Awaiting ECG upload — waveform and risk score will appear here.")
    st.markdown("</div>", unsafe_allow_html=True)

with col2:
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("🖼️ CT Scan")
    ct_file = st.file_uploader("Upload CT Scan (.png or .jpg)", type=["png", "jpg", "jpeg"], key="ct_uploader")

    if ct_file is not None:
        ct_path = save_upload_to_temp(ct_file)
        try:
            p_ct = predict_ct_risk(ct_path)
            image = Image.open(ct_path)
            st.image(image, caption="Uploaded CT Scan", use_container_width=True)
            st.metric("CT Risk Score", f"{p_ct * 100:.1f}%")
        except Exception as exc:  # noqa: BLE001
            st.error(f"Could not process CT scan: {exc}")
        finally:
            try:
                os.remove(ct_path)
            except OSError:
                pass
    else:
        st.info("Awaiting CT scan upload — image preview and risk score will appear here.")
    st.markdown("</div>", unsafe_allow_html=True)

# --------------------------------------------------------------------------
# Bottom dashboard — Composite risk
# --------------------------------------------------------------------------
st.divider()
st.subheader("📊 Composite Risk Dashboard")

try:
    composite_score, status = calculate_composite_risk(p_tab, p_ecg, p_ct)
except Exception as exc:  # noqa: BLE001
    st.error(f"Composite risk calculation error: {exc}")
    composite_score, status = 0.0, "Unknown"

dash_col1, dash_col2, dash_col3, dash_col4 = st.columns(4)

with dash_col1:
    st.metric("Tabular Risk", f"{p_tab * 100:.1f}%")
with dash_col2:
    st.metric("ECG Risk", f"{p_ecg * 100:.1f}%")
with dash_col3:
    st.metric("CT Risk", f"{p_ct * 100:.1f}%")
with dash_col4:
    st.metric("Composite Risk", f"{composite_score * 100:.1f}%")

st.write("")
badge_class = badge_for_status(status)
st.markdown(
    f'<div class="status-badge {badge_class}">Status: {status.upper()}</div>',
    unsafe_allow_html=True,
)

st.write("")
st.progress(min(max(composite_score, 0.0), 1.0))
