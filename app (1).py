"""
CardioRisk AI — Multimodal Cardiovascular Risk Dashboard
A black / amber-yellow themed Streamlit application that fuses tabular
vitals, ECG waveform data, and CT scan imagery into a single composite
risk score.

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
# Page configuration & theme styling
# --------------------------------------------------------------------------
st.set_page_config(
    page_title="CardioRisk AI",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Design tokens: near-black base, one amber-yellow accent, warm mid-tone
# greys for structure. Risk badges keep semantic red/amber/green — the one
# deliberate exception to the palette, because a screening tool that mutes
# "high risk" to on-brand yellow is a usability problem, not a style win.
DARK_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Archivo:wght@600;700;800;900&family=Inter:wght@400;500;600;700&display=swap');

:root {
    --bg: #0a0906;
    --bg-soft: #100e0a;
    --surface: #16130e;
    --surface-2: #1e1a12;
    --border: #33291a;
    --border-strong: #574726;
    --yellow: #f2b807;
    --yellow-soft: rgba(242, 184, 7, 0.12);
    --yellow-dim: #8c6a16;
    --text: #f4efe3;
    --text-muted: #9c9280;
    --red: #e5484d;
    --red-soft: rgba(229, 72, 77, 0.14);
    --green: #5fb77e;
    --green-soft: rgba(95, 183, 126, 0.14);
}

@keyframes riseIn {
    from { opacity: 0; transform: translateY(10px); }
    to   { opacity: 1; transform: translateY(0); }
}
@keyframes underlineDraw {
    from { transform: scaleX(0); }
    to   { transform: scaleX(1); }
}
@keyframes glowPulse {
    0%, 100% { box-shadow: 0 0 0 0 rgba(229, 72, 77, 0.35); }
    50%      { box-shadow: 0 0 0 8px rgba(229, 72, 77, 0); }
}
@keyframes shimmer {
    0%   { background-position: -120px 0; }
    100% { background-position: 220px 0; }
}
@keyframes dotPulse {
    0%, 100% { opacity: 1; }
    50%      { opacity: 0.35; }
}

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: var(--bg);
    color: var(--text);
    position: relative;
}

/* faint hazard-stripe watermark, top-right corner only — a wink at the
   subject matter (risk / caution), never in reading areas */
.stApp::before {
    content: "";
    position: fixed;
    top: -120px;
    right: -160px;
    width: 480px;
    height: 480px;
    background: repeating-linear-gradient(
        45deg,
        var(--yellow) 0px, var(--yellow) 14px,
        transparent 14px, transparent 28px
    );
    opacity: 0.045;
    pointer-events: none;
    z-index: 0;
    border-radius: 50%;
}

section[data-testid="stSidebar"] {
    background-color: var(--bg-soft);
    border-right: 1px solid var(--border);
}
section[data-testid="stSidebar"] * { color: var(--text); }

/* ---- typography ---- */
h1, h2, h3 {
    font-family: 'Archivo', sans-serif;
    color: var(--text);
    letter-spacing: -0.01em;
}
h1 { font-weight: 900; }
h2, h3 { font-weight: 700; }

.app-title {
    font-family: 'Archivo', sans-serif;
    font-weight: 900;
    font-size: 2.4rem;
    color: var(--text);
    margin-bottom: 0;
    animation: riseIn 0.5s ease both;
}
.app-title .accent { color: var(--yellow); position: relative; }
.app-title .accent::after {
    content: "";
    position: absolute;
    left: 0; right: 0; bottom: 2px;
    height: 4px;
    background: var(--yellow);
    border-radius: 2px;
    transform-origin: left;
    animation: underlineDraw 0.6s 0.35s cubic-bezier(0.65, 0, 0.35, 1) both;
}
.subtle {
    color: var(--text-muted);
    font-size: 0.9rem;
    animation: riseIn 0.5s 0.12s ease both;
}

/* ---- metrics ---- */
div[data-testid="stMetric"] {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 18px 20px;
    transition: border-color 0.25s ease, transform 0.25s ease;
    animation: riseIn 0.5s ease both;
}
div[data-testid="stMetric"]:hover {
    border-color: var(--yellow-dim);
    transform: translateY(-2px);
}
div[data-testid="stMetricValue"] { color: var(--text); font-family: 'Archivo', sans-serif; }
div[data-testid="stMetricLabel"] { color: var(--text-muted); }

/* ---- status badge (capsule, mirrors a pill-link aesthetic) ---- */
.status-badge {
    display: inline-flex;
    align-items: center;
    gap: 10px;
    padding: 10px 22px;
    border-radius: 999px;
    font-family: 'Archivo', sans-serif;
    font-weight: 700;
    font-size: 1.02rem;
    letter-spacing: 0.3px;
    animation: riseIn 0.5s ease both;
}
.status-badge .dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    background: currentColor;
}
.badge-green {
    background-color: var(--green-soft);
    color: var(--green);
    border: 1px solid var(--green);
}
.badge-yellow {
    background-color: var(--yellow-soft);
    color: var(--yellow);
    border: 1px solid var(--yellow);
}
.badge-red {
    background-color: var(--red-soft);
    color: var(--red);
    border: 1px solid var(--red);
    animation: riseIn 0.5s ease both, glowPulse 2.4s ease-in-out infinite;
}
.badge-red .dot { animation: dotPulse 1.4s ease-in-out infinite; }

/* ---- section cards (fallback for the manual div wrapper) ---- */
.section-card {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: 16px;
    padding: 16px 18px;
    margin-bottom: 14px;
}

/* ---- real card containers: the two-column ECG / CT row ---- */
div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-child(2):last-child)
    > div[data-testid="column"] > div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-child(2):last-child)
    > div[data-testid="column"] > div[data-testid="stVerticalBlock"] {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: 18px;
    padding: 20px 22px;
    box-shadow: 6px 6px 0px 0px rgba(242, 184, 7, 0.05);
    transition: border-color 0.25s ease, box-shadow 0.25s ease, transform 0.25s ease;
    animation: riseIn 0.55s ease both;
}
div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-child(2):last-child)
    > div[data-testid="column"]:nth-child(2) > div[data-testid="stVerticalBlockBorderWrapper"],
div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-child(2):last-child)
    > div[data-testid="column"]:nth-child(2) > div[data-testid="stVerticalBlock"] {
    animation-delay: 0.1s;
}
div[data-testid="stHorizontalBlock"]:has(> div[data-testid="column"]:nth-child(2):last-child)
    > div[data-testid="column"] > div[data-testid="stVerticalBlock"]:hover {
    border-color: var(--border-strong);
    box-shadow: 8px 8px 0px 0px rgba(242, 184, 7, 0.08);
}

/* ---- buttons (pill, mirrors an "OPEN LINK" capsule button) ---- */
div.stButton > button {
    width: 100%;
    border-radius: 999px;
    border: 1.5px solid var(--yellow-dim);
    background-color: var(--surface-2);
    color: var(--text);
    font-family: 'Archivo', sans-serif;
    font-weight: 700;
    letter-spacing: 0.2px;
    padding: 0.55rem 1rem;
    transition: background-color 0.2s ease, border-color 0.2s ease,
                color 0.2s ease, transform 0.15s ease;
}
div.stButton > button:hover {
    background-color: var(--yellow);
    border-color: var(--yellow);
    color: #0a0906;
    transform: translateY(-1px);
}
div.stButton > button:active { transform: translateY(0); }

/* ---- inputs: sliders, number inputs, selects, toggles ---- */
div[data-testid="stSlider"] [data-baseweb="slider"] > div > div { background: var(--border); }
div[data-testid="stSlider"] [role="slider"] {
    background-color: var(--yellow) !important;
    border-color: var(--yellow) !important;
    box-shadow: 0 0 0 4px var(--yellow-soft);
}
div[data-testid="stNumberInput"] input,
div[data-baseweb="select"] > div,
div[data-testid="stFileUploaderDropzone"] {
    background-color: var(--surface) !important;
    border: 1px solid var(--border) !important;
    color: var(--text) !important;
    border-radius: 10px !important;
}
div[data-testid="stFileUploaderDropzone"] {
    transition: border-color 0.2s ease, background-color 0.2s ease;
}
div[data-testid="stFileUploaderDropzone"]:hover {
    border-color: var(--yellow-dim) !important;
    background-color: var(--surface-2) !important;
}
div[data-baseweb="checkbox"] { color: var(--text); }
label[data-testid="stWidgetLabel"] p { color: var(--text-muted); font-weight: 500; }

/* toggle switch accent when on */
div[data-testid="stCheckbox"] div[aria-checked="true"],
div[role="switch"][aria-checked="true"] {
    background-color: var(--yellow) !important;
    border-color: var(--yellow) !important;
}

/* ---- expander ---- */
div[data-testid="stExpander"] {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
}

/* ---- alerts (info / error) ---- */
div[data-testid="stAlert"] {
    background-color: var(--surface);
    border: 1px solid var(--border);
    border-radius: 12px;
}

/* ---- progress bar: gradient fill with a slow shimmer sweep ---- */
div[data-testid="stProgress"] > div > div {
    background-color: var(--border) !important;
    border-radius: 999px;
}
div[data-testid="stProgress"] > div > div > div {
    background: linear-gradient(90deg, var(--yellow-dim), var(--yellow) 60%, var(--yellow));
    background-size: 200px 100%;
    animation: shimmer 2.6s linear infinite;
    border-radius: 999px;
    transition: width 0.8s cubic-bezier(0.22, 1, 0.36, 1);
}

hr, div[data-testid="stDivider"] { border-color: var(--border) !important; }
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
    """Render an ECG waveform as a dark, amber-lined matplotlib plot."""
    fig, ax = plt.subplots(figsize=(5, 2.6))
    fig.patch.set_facecolor("#16130e")
    ax.set_facecolor("#16130e")
    ax.plot(waveform, color="#f2b807", linewidth=1.2)
    ax.set_xlabel("Sample", color="#9c9280", fontsize=8)
    ax.set_ylabel("Amplitude", color="#9c9280", fontsize=8)
    ax.tick_params(colors="#9c9280", labelsize=7)
    for spine in ax.spines.values():
        spine.set_color("#33291a")
    ax.grid(alpha=0.15, color="#574726")
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
st.markdown('<div class="app-title">🫀 Cardio<span class="accent">Risk</span> AI</div>', unsafe_allow_html=True)
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
    f'<div class="status-badge {badge_class}"><span class="dot"></span>Status: {status.upper()}</div>',
    unsafe_allow_html=True,
)

st.write("")
st.progress(min(max(composite_score, 0.0), 1.0))
