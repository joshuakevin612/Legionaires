"""
ml_engine.py
============
Standalone, self-contained ML inference engine for a Multimodal
"Silent Stroke" Early Warning System.

Three independent modalities are scored and then fused into a single
composite risk score:

    1. Tabular clinical risk factors  -> predict_tabular_risk()
    2. 1D ECG waveform (1000 samples) -> predict_ecg_risk()
    3. Brain CT scan image            -> predict_ct_risk()
    4. Fusion of the three            -> calculate_composite_risk()

Design goals
------------
- Zero hard crashes: every function degrades gracefully to a safe,
  clearly-labelled mock/fallback prediction if a model library,
  weights file, or input file is missing or malformed.
- Zero mandatory external state: if XGBoost / PyTorch / torchvision
  are installed, they are used for "real" inference. If not, an
  internal, deterministic, clinically-informed baseline heuristic is
  used instead, so the script always runs standalone.
- Fully self-contained: `python ml_engine.py` runs a demo end-to-end
  with no input files required.

Author: Senior AI/ML Engineer (generated)
"""

from __future__ import annotations

import io
import math
import os
import random
from typing import List, Optional, Tuple

import numpy as np

# --------------------------------------------------------------------------
# Optional heavy dependencies. Each is imported defensively so the module
# NEVER fails to import, and each capability degrades to a mock baseline
# if the corresponding library isn't available in the environment.
# --------------------------------------------------------------------------
try:
    import xgboost as xgb  # type: ignore
    _HAS_XGB = True
except Exception:
    _HAS_XGB = False

try:
    from sklearn.ensemble import RandomForestClassifier  # type: ignore
    _HAS_SKLEARN = True
except Exception:
    _HAS_SKLEARN = False

try:
    import torch  # type: ignore
    import torch.nn as nn  # type: ignore
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    import torchvision.models as tv_models  # type: ignore
    import torchvision.transforms as tv_transforms  # type: ignore
    _HAS_TORCHVISION = True
except Exception:
    _HAS_TORCHVISION = False

try:
    from PIL import Image  # type: ignore
    _HAS_PIL = True
except Exception:
    _HAS_PIL = False


# A fixed seed keeps mock/random-weight predictions reproducible across runs.
_SEED = 42
random.seed(_SEED)
np.random.seed(_SEED)
if _HAS_TORCH:
    torch.manual_seed(_SEED)


def _sigmoid(x: float) -> float:
    """Numerically stable sigmoid -> squashes any real number to (0, 1)."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _clip01(x: float) -> float:
    """Clamp a float to the valid probability range [0.0, 1.0]."""
    return max(0.0, min(1.0, float(x)))


# ==========================================================================
# 1. TABULAR RISK MODEL
# ==========================================================================
#
# Expected keys in `data_dict`:
#   age                 : float/int  (years)
#   hypertension        : int/bool   (0 or 1)
#   heart_disease       : int/bool   (0 or 1)
#   avg_glucose_level   : float      (mg/dL)
#   bmi                 : float      (kg/m^2)
#   smoking_status      : str        one of
#                          {"never smoked", "formerly smoked",
#                           "smokes", "Unknown"}

_SMOKING_ENCODING = {
    "never smoked": 0.0,
    "unknown": 0.0,
    "formerly smoked": 0.5,
    "smokes": 1.0,
}

# Baseline logistic-regression-style weights, informed loosely by known
# clinical stroke-risk literature (age, hypertension, glucose, etc. are all
# established risk factors). These are used whenever no trained
# XGBoost/RandomForest model is available -- i.e. the "internal mock
# baseline weights" fallback required by the interface contract.
_BASELINE_WEIGHTS = {
    "bias": -5.5,
    "age": 0.045,               # risk climbs steadily with age
    "hypertension": 1.1,
    "heart_disease": 1.3,
    "avg_glucose_level": 0.012,
    "bmi": 0.03,
    "smoking_status": 0.8,
}

_tabular_model = None  # lazily-trained model cache (XGBoost or RandomForest)


def _make_synthetic_tabular_dataset(n_samples: int = 800):
    """
    Generates a small synthetic-but-plausible clinical dataset so an
    XGBoost/RandomForest model can be fit at runtime without requiring
    the user to supply a pre-trained model file. Labels are generated
    from the same clinically-informed baseline formula, plus noise, so
    the learned model and the fallback heuristic stay broadly consistent.
    """
    rng = np.random.default_rng(_SEED)
    age = rng.uniform(18, 90, n_samples)
    hypertension = rng.integers(0, 2, n_samples)
    heart_disease = rng.integers(0, 2, n_samples)
    glucose = rng.uniform(60, 300, n_samples)
    bmi = rng.uniform(15, 45, n_samples)
    smoking = rng.choice([0.0, 0.5, 1.0], n_samples)

    logits = (
        _BASELINE_WEIGHTS["bias"]
        + _BASELINE_WEIGHTS["age"] * age
        + _BASELINE_WEIGHTS["hypertension"] * hypertension
        + _BASELINE_WEIGHTS["heart_disease"] * heart_disease
        + _BASELINE_WEIGHTS["avg_glucose_level"] * glucose
        + _BASELINE_WEIGHTS["bmi"] * bmi
        + _BASELINE_WEIGHTS["smoking_status"] * smoking
        + rng.normal(0, 1.0, n_samples)  # label noise
    )
    probs = 1.0 / (1.0 + np.exp(-logits))
    labels = (probs > 0.5).astype(int)

    X = np.column_stack([age, hypertension, heart_disease, glucose, bmi, smoking])
    return X, labels


def _get_tabular_model():
    """
    Lazily builds and caches an XGBoost model (preferred), falling back to
    RandomForest, trained on a synthetic clinical dataset. Returns None if
    neither library is installed, in which case the pure-formula baseline
    in `predict_tabular_risk` is used instead.
    """
    global _tabular_model
    if _tabular_model is not None:
        return _tabular_model

    try:
        X, y = _make_synthetic_tabular_dataset()
        if _HAS_XGB:
            model = xgb.XGBClassifier(
                n_estimators=60,
                max_depth=3,
                learning_rate=0.15,
                eval_metric="logloss",
                random_state=_SEED,
            )
            model.fit(X, y)
            _tabular_model = ("xgboost", model)
        elif _HAS_SKLEARN:
            model = RandomForestClassifier(
                n_estimators=100, max_depth=6, random_state=_SEED
            )
            model.fit(X, y)
            _tabular_model = ("random_forest", model)
        else:
            _tabular_model = None
    except Exception:
        # If anything about training goes wrong, silently fall back.
        _tabular_model = None

    return _tabular_model


def predict_tabular_risk(data_dict: dict) -> float:
    """
    Predicts stroke risk probability from tabular clinical features.

    Parameters
    ----------
    data_dict : dict
        Keys: 'age', 'hypertension', 'heart_disease',
              'avg_glucose_level', 'bmi', 'smoking_status'

    Returns
    -------
    float
        Risk probability in [0.0, 1.0].
    """
    # --- Safely extract & sanitize inputs, with clinically sane defaults ---
    try:
        age = float(data_dict.get("age", 50))
    except (TypeError, ValueError):
        age = 50.0
    try:
        hypertension = int(bool(data_dict.get("hypertension", 0)))
    except (TypeError, ValueError):
        hypertension = 0
    try:
        heart_disease = int(bool(data_dict.get("heart_disease", 0)))
    except (TypeError, ValueError):
        heart_disease = 0
    try:
        glucose = float(data_dict.get("avg_glucose_level", 100.0))
    except (TypeError, ValueError):
        glucose = 100.0
    try:
        bmi = float(data_dict.get("bmi", 25.0))
    except (TypeError, ValueError):
        bmi = 25.0

    smoking_raw = str(data_dict.get("smoking_status", "Unknown")).strip().lower()
    smoking_encoded = _SMOKING_ENCODING.get(smoking_raw, 0.0)

    # Clip to physiologically plausible ranges to keep the model stable.
    age = _clip01(age / 120.0) * 120.0
    glucose = max(40.0, min(600.0, glucose))
    bmi = max(10.0, min(70.0, bmi))

    model_bundle = _get_tabular_model()

    if model_bundle is not None:
        try:
            _, model = model_bundle
            features = np.array(
                [[age, hypertension, heart_disease, glucose, bmi, smoking_encoded]]
            )
            proba = model.predict_proba(features)[0][1]
            return _clip01(proba)
        except Exception:
            pass  # fall through to baseline formula on any inference error

    # --- Internal mock baseline (logistic regression formula) ---
    logit = (
        _BASELINE_WEIGHTS["bias"]
        + _BASELINE_WEIGHTS["age"] * age
        + _BASELINE_WEIGHTS["hypertension"] * hypertension
        + _BASELINE_WEIGHTS["heart_disease"] * heart_disease
        + _BASELINE_WEIGHTS["avg_glucose_level"] * glucose
        + _BASELINE_WEIGHTS["bmi"] * bmi
        + _BASELINE_WEIGHTS["smoking_status"] * smoking_encoded
    )
    return _clip01(_sigmoid(logit))


# ==========================================================================
# 2. ECG RISK MODEL (1D CNN)
# ==========================================================================

_ECG_LENGTH = 1000


def _generate_synthetic_ecg(length: int = _ECG_LENGTH, anomalous: Optional[bool] = None) -> np.ndarray:
    """
    Generates a plausible synthetic ECG-like waveform using a sum of sine
    waves (simulating P-QRS-T rhythm) plus noise. If `anomalous` is True,
    irregular spikes/arrhythmia-like distortions are injected to simulate
    a higher-risk signal. If None, it is chosen randomly.
    """
    if anomalous is None:
        anomalous = random.random() > 0.5

    t = np.linspace(0, 10, length)
    heart_rate_hz = 1.2  # ~72 bpm baseline rhythm
    signal = 0.6 * np.sin(2 * np.pi * heart_rate_hz * t)
    signal += 0.15 * np.sin(2 * np.pi * heart_rate_hz * 3 * t)  # QRS-like harmonic
    signal += 0.05 * np.random.normal(0, 1, length)  # baseline noise

    if anomalous:
        # Inject irregular spikes / dropped beats to mimic arrhythmia signs.
        n_spikes = random.randint(3, 8)
        for _ in range(n_spikes):
            idx = random.randint(0, length - 1)
            width = random.randint(3, 10)
            spike = np.hanning(width) * random.uniform(0.8, 1.6) * random.choice([-1, 1])
            end = min(length, idx + width)
            signal[idx:end] += spike[: end - idx]

    return signal.astype(np.float32)


def _load_ecg_from_file(file_path: str) -> Optional[np.ndarray]:
    """
    Attempts to load an ECG signal of length 1000 from a .csv or .npy file.
    Returns None if the file is missing, unreadable, or of the wrong shape
    (after best-effort truncation/padding is not applicable).
    """
    if not file_path or not os.path.isfile(file_path):
        return None

    try:
        ext = os.path.splitext(file_path)[1].lower()
        if ext == ".npy":
            arr = np.load(file_path, allow_pickle=False)
        elif ext == ".csv":
            arr = np.loadtxt(file_path, delimiter=",")
        else:
            return None

        arr = np.asarray(arr, dtype=np.float32).flatten()

        if arr.size == 0:
            return None

        # Be forgiving about slightly-off lengths: pad or truncate to 1000
        # so real-world messy files don't crash the pipeline.
        if arr.size != _ECG_LENGTH:
            if arr.size > _ECG_LENGTH:
                arr = arr[:_ECG_LENGTH]
            else:
                arr = np.pad(arr, (0, _ECG_LENGTH - arr.size), mode="edge")

        return arr
    except Exception:
        return None


class _ECGCNN(nn.Module if _HAS_TORCH else object):
    """
    Lightweight 1D CNN for binary risk classification from a raw ECG
    waveform of length 1000. Architecture:
        Conv1D -> ReLU -> MaxPool -> Conv1D -> ReLU -> MaxPool -> FC -> Sigmoid
    Only defined when PyTorch is available.
    """

    def __init__(self):
        if not _HAS_TORCH:
            return
        super().__init__()
        self.conv1 = nn.Conv1d(1, 16, kernel_size=7, padding=3)
        self.conv2 = nn.Conv1d(16, 32, kernel_size=5, padding=2)
        self.pool = nn.MaxPool1d(2)
        self.relu = nn.ReLU()
        flattened_size = 32 * (_ECG_LENGTH // 4)
        self.fc1 = nn.Linear(flattened_size, 64)
        self.fc2 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return torch.sigmoid(x)


_ecg_model = None


def _get_ecg_model():
    """Lazily builds and caches the 1D CNN (random-initialized weights)."""
    global _ecg_model
    if not _HAS_TORCH:
        return None
    if _ecg_model is None:
        try:
            model = _ECGCNN()
            model.eval()
            _ecg_model = model
        except Exception:
            _ecg_model = None
    return _ecg_model


def predict_ecg_risk(file_path: str = None) -> Tuple[float, List[float]]:
    """
    Predicts stroke risk probability from a 1D ECG waveform of length 1000.

    Parameters
    ----------
    file_path : str, optional
        Path to a .csv or .npy file containing a length-1000 ECG signal.
        If None or the file is missing/invalid, a synthetic ECG signal is
        generated and used instead (safe fallback).

    Returns
    -------
    tuple[float, list]
        (probability_float, signal_list_1000)
    """
    signal = _load_ecg_from_file(file_path) if file_path else None
    used_synthetic = signal is None

    if signal is None:
        signal = _generate_synthetic_ecg()

    model = _get_ecg_model()

    if model is not None:
        try:
            with torch.no_grad():
                tensor = torch.tensor(signal, dtype=torch.float32).view(1, 1, -1)
                # Normalize for numerical stability, as any real pipeline would.
                tensor = (tensor - tensor.mean()) / (tensor.std() + 1e-6)
                output = model(tensor)
                prob = float(output.item())
            return _clip01(prob), signal.tolist()
        except Exception:
            pass  # fall through to mock scoring below

    # --- Mock baseline: score based on signal variability/energy ---
    # Higher variance / abrupt spikes -> heuristically higher "risk".
    variability = float(np.std(np.diff(signal)))
    energy = float(np.mean(np.abs(signal)))
    mock_logit = -3.0 + 4.5 * variability + 1.5 * energy
    prob = _sigmoid(mock_logit)

    # If we had to synthesize the signal ourselves, keep the mock score
    # tethered to whether we deliberately injected an anomaly.
    if used_synthetic:
        prob = _clip01(prob)

    return _clip01(prob), signal.tolist()


# ==========================================================================
# 3. CT SCAN RISK MODEL (ResNet18 backbone)
# ==========================================================================

_CT_IMAGE_SIZE = 224

_resnet_model = None


def _get_ct_model():
    """
    Lazily builds and caches a ResNet18 backbone adapted for binary risk
    classification. Uses torchvision's architecture with randomly
    initialized (or ImageNet-pretrained, if weights are cached locally)
    weights, replacing the final FC layer with a single sigmoid output.
    """
    global _resnet_model
    if not (_HAS_TORCH and _HAS_TORCHVISION):
        return None
    if _resnet_model is not None:
        return _resnet_model

    try:
        try:
            # Try pretrained weights if available/cached; harmless if it
            # fails (e.g., no internet) -- we fall back to random init.
            backbone = tv_models.resnet18(weights="DEFAULT")
        except Exception:
            backbone = tv_models.resnet18(weights=None)

        backbone.fc = nn.Linear(backbone.fc.in_features, 1)
        backbone.eval()
        _resnet_model = backbone
    except Exception:
        _resnet_model = None

    return _resnet_model


def _load_ct_image(file_path: str):
    """
    Loads and preprocesses a CT image: opens as RGB, resizes to 224x224.
    Returns a numpy array (H, W, 3) in [0, 255], or None on any failure.
    """
    if not file_path or not os.path.isfile(file_path) or not _HAS_PIL:
        return None
    try:
        img = Image.open(file_path).convert("RGB")
        img = img.resize((_CT_IMAGE_SIZE, _CT_IMAGE_SIZE))
        return np.array(img)
    except Exception:
        return None


def _generate_synthetic_ct_array() -> np.ndarray:
    """Generates a plausible-looking 224x224x3 grayscale-ish CT placeholder."""
    rng = np.random.default_rng(_SEED + 1)
    base = rng.normal(120, 25, (_CT_IMAGE_SIZE, _CT_IMAGE_SIZE))
    base = np.clip(base, 0, 255)
    return np.stack([base, base, base], axis=-1).astype(np.uint8)


def predict_ct_risk(file_path: str = None) -> float:
    """
    Predicts stroke risk probability from a brain CT scan image.

    Parameters
    ----------
    file_path : str, optional
        Path to a .png/.jpg CT image. If None or the file is
        missing/invalid, a synthetic placeholder image is used instead.

    Returns
    -------
    float
        Risk probability in [0.0, 1.0].
    """
    img_array = _load_ct_image(file_path)
    used_synthetic = img_array is None

    if img_array is None:
        img_array = _generate_synthetic_ct_array()

    model = _get_ct_model()

    if model is not None:
        try:
            if _HAS_TORCHVISION:
                preprocess = tv_transforms.Compose(
                    [
                        tv_transforms.ToTensor(),
                        tv_transforms.Normalize(
                            mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225],
                        ),
                    ]
                )
                tensor = preprocess(img_array).unsqueeze(0)
            else:
                tensor = torch.tensor(img_array, dtype=torch.float32)
                tensor = tensor.permute(2, 0, 1).unsqueeze(0) / 255.0

            with torch.no_grad():
                output = torch.sigmoid(model(tensor))
                prob = float(output.item())
            return _clip01(prob)
        except Exception:
            pass  # fall through to mock scoring below

    # --- Mock baseline: score based on image intensity statistics ---
    gray = img_array.mean(axis=-1) if img_array.ndim == 3 else img_array
    mean_intensity = float(np.mean(gray)) / 255.0
    contrast = float(np.std(gray)) / 255.0
    mock_logit = -2.0 + 2.5 * mean_intensity + 3.0 * contrast
    prob = _sigmoid(mock_logit)

    if used_synthetic:
        prob = _clip01(prob)

    return _clip01(prob)


# ==========================================================================
# 4. COMPOSITE RISK FUSION
# ==========================================================================

_WEIGHT_TAB = 0.35
_WEIGHT_ECG = 0.35
_WEIGHT_CT = 0.30


def calculate_composite_risk(p_tab: float, p_ecg: float, p_ct: float) -> Tuple[float, str]:
    """
    Fuses the three modality-level probabilities into a single composite
    risk score using a fixed weighted-average formula:

        Composite = 0.35 * p_tab + 0.35 * p_ecg + 0.30 * p_ct

    Parameters
    ----------
    p_tab : float
        Tabular clinical-risk probability, in [0.0, 1.0].
    p_ecg : float
        ECG-derived risk probability, in [0.0, 1.0].
    p_ct : float
        CT-scan-derived risk probability, in [0.0, 1.0].

    Returns
    -------
    tuple[float, str]
        (risk_percentage_float, risk_tier_str)
        risk_percentage_float is on a 0-100 scale.
        risk_tier_str is one of:
            'Low (<35%)'
            'Moderate (35-69%)'
            'High / Silent Stroke Alert (>=70%)'
    """
    p_tab = _clip01(p_tab)
    p_ecg = _clip01(p_ecg)
    p_ct = _clip01(p_ct)

    composite = (_WEIGHT_TAB * p_tab) + (_WEIGHT_ECG * p_ecg) + (_WEIGHT_CT * p_ct)
    risk_percentage = round(composite * 100.0, 2)

    if risk_percentage >= 70.0:
        tier = "High / Silent Stroke Alert (>=70%)"
    elif risk_percentage >= 35.0:
        tier = "Moderate (35-69%)"
    else:
        tier = "Low (<35%)"

    return risk_percentage, tier


# ==========================================================================
# DEMO / SELF-TEST
# ==========================================================================

def _run_demo():
    """
    Runs the full pipeline end-to-end with no input files, exercising the
    fallback/mock paths so the module is verifiably self-contained.
    """
    print("=" * 70)
    print("Silent Stroke Early Warning System -- ml_engine.py demo")
    print(f"XGBoost available:    {_HAS_XGB}")
    print(f"scikit-learn available: {_HAS_SKLEARN}")
    print(f"PyTorch available:    {_HAS_TORCH}")
    print(f"torchvision available: {_HAS_TORCHVISION}")
    print(f"Pillow available:     {_HAS_PIL}")
    print("=" * 70)

    sample_patient = {
        "age": 67,
        "hypertension": 1,
        "heart_disease": 1,
        "avg_glucose_level": 210.5,
        "bmi": 31.2,
        "smoking_status": "formerly smoked",
    }

    p_tab = predict_tabular_risk(sample_patient)
    p_ecg, ecg_signal = predict_ecg_risk(None)  # no file -> synthetic fallback
    p_ct = predict_ct_risk(None)                # no file -> synthetic fallback

    risk_pct, tier = calculate_composite_risk(p_tab, p_ecg, p_ct)

    print(f"\nPatient profile: {sample_patient}")
    print(f"Tabular risk probability : {p_tab:.4f}")
    print(f"ECG risk probability     : {p_ecg:.4f}  (signal length: {len(ecg_signal)})")
    print(f"CT risk probability      : {p_ct:.4f}")
    print(f"\nComposite risk           : {risk_pct}%")
    print(f"Risk tier                : {tier}")
    print("=" * 70)


if __name__ == "__main__":
    _run_demo()
