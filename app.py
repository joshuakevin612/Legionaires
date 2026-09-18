"""
CardioRisk AI — Multimodal Silent Stroke Risk Dashboard
=========================================================
A clean, dependency-light Streamlit app that fuses tabular vitals, an ECG
waveform, and a CT scan into one composite stroke-risk score.

Run with:
    streamlit run app.py

Requires ml_engine.py (same folder) exposing:
    predict_tabular_risk(data_dict: dict) -> float
    predict_ecg_risk(file_path: str) -> tuple[float, list]
    predict_ct_risk(file_path: str) -> float
    calculate_composite_risk(p_tab, p_ecg, p_ct) -> tuple[float, str]

Design notes (why this version is written the way it is)
----------------------------------------------------------
- No CSS :has() selectors, no @keyframes, no google-font @import blocking
  render. Those are the most common cause of a Streamlit page that loads
  but renders blank: unsupported/slow CSS can stall first paint in some
  browsers, and there's nothing here to fall back on if it fails.
- Every external call (ml_engine functions, file loads) is wrapped in
  try/except with a visible st.error(), so a bug surfaces as a red banner
  in the page instead of a silent blank screen.
- No st.set_page_config() call is skipped or duplicated, and it is always
  the first Streamlit command, which avoids Streamlit's own startup error
  for that specific ordering rule.
"""

import os
import tempfile

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from PIL import Image

# --------------------------------------------------------------------------
# Import ml_engine defensively. If it's missing or broken, the app still
# renders (with a clear warning) instead of showing a blank page.
# --------------------------------------------------------------------------
ML_ENGINE_OK = True
ML_ENGINE_ERROR = ""
try:
    from ml_engine import (
        predict_tabular_risk,
        predict_ecg_risk,
        predict_ct_risk,
        calculate_composite_risk,
    )
except Exception as exc:  # noqa: BLE001
    ML_ENGINE_OK = False
    ML_ENGINE_ERROR = str(exc)

# --------------------------------------------------------------------------
# Page configuration — MUST be the first Streamlit command.
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioRisk AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --------------------------------------------------------------------------
# Minimal, safe theme CSS — plain rules only, nothing that can stall the
# browser's first paint.
# --------------------------------------------------------------------------
st.markdown(
    """
    <style>
    :root {
        --bg: #f7e8a8;
        --surface: #fdf6d8;
        --border: #14110c;
        --yellow: #f2a90c;
        --text: #14110c;
        --text-muted: #6b6350;
        --red: #c62828;
        --red-soft: #f6d6d6;
        --green: #2f7d3c;
        --green-soft: #d9ecdb;
        --yellow-soft: #fbe7b8;
    }
    .stApp { background-color: var(--bg); }
    .app-title {
        font-size: 2.2rem;
        font-weight: 800;
        color: var(--text);
        margin-bottom: 0;
    }
    .app-title .accent { color: var(--yellow); }
    .subtle { color: var(--text-muted); font-size: 0.95rem; }
    div[data-testid="stMetric"] {
        background-color: var(--surface);
        border: 2px solid var(--border);
        border-radius: 12px;
        padding: 14px 16px;
    }
    .status-badge {
        display: inline-block;
        padding: 10px 22px;
        border-radius: 999px;
        font-weight: 700;
        font-size: 1.05rem;
        border: 1px solid var(--border);
    }
    .badge-green { background-color: var(--green-soft); color: var(--green); border-color: var(--green); }
    .badge-yellow { background-color: var(--yellow-soft); color: #8a6300; border-color: var(--yellow); }
    .badge-red { background-color: var(--red-soft); color: var(--red); border-color: var(--red); }
    .section-card {
        background-color: var(--surface);
        border: 2px solid var(--border);
        border-radius: 14px;
        padding: 16px 18px;
        margin-bottom: 10px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --------------------------------------------------------------------------
# Session-state defaults
# --------------------------------------------------------------------------
DEFAULTS = {
    "age": 55,
    "hypertension": 0,
    "heart_disease": 0,
    "avg_glucose_level": 100.0,
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
    for key, value in DEMO_HIGH_RISK.items():
        st.session_state[key] = value


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def save_upload_to_temp(uploaded_file) -> str:
    suffix = os.path.splitext(uploaded_file.name)[1]
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getbuffer())
        return tmp.name


def load_waveform(file_path: str, original_name: str) -> np.ndarray:
    if original_name.lower().endswith(".npy"):
        arr = np.load(file_path)
    else:
        df = pd.read_csv(file_path, header=None)
        arr = df.select_dtypes(include=[np.number]).to_numpy()
    return np.asarray(arr).flatten()


def plot_waveform(waveform: np.ndarray):
    fig, ax = plt.subplots(figsize=(5, 2.4))
    fig.patch.set_facecolor("#fdf6d8")
    ax.set_facecolor("#fdf6d8")
    ax.plot(waveform, color="#d99a0a", linewidth=1.2)
    ax.set_xlabel("Sample", color="#6b6350", fontsize=8)
    ax.set_ylabel("Amplitude", color="#6b6350", fontsize=8)
    ax.tick_params(colors="#6b6350", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#14110c")
    ax.grid(alpha=0.15, color="#14110c")
    fig.tight_layout()
    return fig


def badge_for_status(status: str) -> str:
    status_lower = status.lower()
    if "low" in status_lower:
        return "badge-green"
    if "moderate" in status_lower or "medium" in status_lower:
        return "badge-yellow"
    return "badge-red"


# --------------------------------------------------------------------------
# If ml_engine failed to import, stop here with a clear message instead of
# letting every downstream call raise and risk a broken page.
# --------------------------------------------------------------------------
if not ML_ENGINE_OK:
    st.markdown('<div class="app-title">🫀 Cardio<span class="accent">Risk</span> AI</div>', unsafe_allow_html=True)
    st.error(
        "Could not import ml_engine.py — the app cannot compute risk scores "
        "until this is fixed.\n\n"
        f"Import error: {ML_ENGINE_ERROR}\n\n"
        "Make sure ml_engine.py is in the same folder as app.py, and that "
        "its dependencies (numpy at minimum; joblib/torch/torchvision are "
        "optional) are installed in this environment."
    )
    st.stop()

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
            "Hypertension", value=bool(st.session_state["hypertension"]), key="hypertension_toggle"
        )
    with col_b:
        heart_disease = st.toggle(
            "Heart Disease", value=bool(st.session_state["heart_disease"]), key="heart_disease_toggle"
        )
    st.session_state["hypertension"] = int(hypertension)
    st.session_state["heart_disease"] = int(heart_disease)

    st.number_input(
        "Avg Glucose Level (mg/dL)", min_value=0.0, max_value=500.0, step=0.1, key="avg_glucose_level"
    )
    st.number_input("BMI", min_value=0.0, max_value=80.0, step=0.1, key="bmi")
    st.selectbox(
        "Smoking Status",
        options=["formerly smoked", "never smoked", "smokes"],
        key="smoking_status",
    )

# --------------------------------------------------------------------------
# Header
# --------------------------------------------------------------------------
st.markdown('<div class="app-title">🫀 Cardio<span class="accent">Risk</span> AI</div>', unsafe_allow_html=True)
st.markdown(
    '<p class="subtle">Multimodal silent-stroke risk stratification — '
    "tabular vitals, ECG waveform, and CT imaging fused into one composite score.</p>",
    unsafe_allow_html=True,
)
st.write("")

# --------------------------------------------------------------------------
# Tabular risk (computed live from sidebar inputs)
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
                    st.write(f"Signal length: {len(ecg_extra)} samples")
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
    # composite_score already comes back from calculate_composite_risk() on
    # a 0-100 scale, so it's displayed as-is (no second *100 multiply).
    st.metric("Composite Risk", f"{composite_score:.1f}%")

st.write("")
badge_class = badge_for_status(status)
st.markdown(
    f'<div class="status-badge {badge_class}">Status: {status.upper()}</div>',
    unsafe_allow_html=True,
)

st.write("")
# st.progress() expects a 0-1 fraction, but composite_score is 0-100, so it
# is normalized here before clamping.
st.progress(min(max(composite_score / 100.0, 0.0), 1.0))

st.write("")
st.caption(
    "This tool is a research prototype for decision support only. "
    "It is not a validated diagnostic device and does not replace clinical judgment."
)
