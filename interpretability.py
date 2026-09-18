"""
interpretability.py
====================
Explains individual predictions from ml_engine.py's three models, so the
dashboard can show clinicians/users WHY a risk score came out the way it
did -- not just the number.

- explain_tabular_risk(data_dict)  -> per-feature contribution to risk
      Uses SHAP TreeExplainer when the `shap` package is installed
      (exact, fast for tree models). Falls back to a transparent
      "baseline attribution" method otherwise: how much would this
      patient's risk drop if THIS ONE feature were swapped for a
      healthy reference value, holding everything else fixed? This is
      NOT a true Shapley value (contributions aren't guaranteed to sum
      exactly to the total) but it's an honest, easy-to-explain,
      zero-dependency stand-in.

- explain_ct_risk(file_path)   -> Grad-CAM heatmap over the CT image
- explain_ecg_risk(file_path)  -> Grad-CAM heatmap over the ECG spectrogram

Grad-CAM requires torch + torchvision + matplotlib (same as ml_engine.py's
CT/ECG paths). If unavailable, both functions return None and the caller
should just show the raw prediction without the heatmap.
"""

from typing import Optional

import numpy as np

import ml_engine as engine

try:
    import shap  # type: ignore
    _HAS_SHAP = True
except Exception:
    _HAS_SHAP = False

try:
    import torch
    import torch.nn.functional as F
    _HAS_TORCH = True
except Exception:
    _HAS_TORCH = False

try:
    import matplotlib.cm as cm
    _HAS_MATPLOTLIB = True
except Exception:
    _HAS_MATPLOTLIB = False


# ==========================================================================
# TABULAR: SHAP (preferred) or baseline-attribution (fallback)
# ==========================================================================

# A clinically reasonable "healthy adult" reference point -- NOT the
# dataset average, so contributions read as "vs. a healthy baseline"
# rather than "vs. the average patient in our training data."
_HEALTHY_BASELINE = {
    "age": 45,
    "hypertension": 0,
    "heart_disease": 0,
    "avg_glucose_level": 90.0,
    "bmi": 22.0,
    "smoking_status": "never smoked",
}


def explain_tabular_risk(data_dict: dict) -> dict:
    """
    Returns a dict:
        {
          "method": "shap" | "baseline_attribution",
          "baseline_risk": float,      # risk for the healthy reference patient
          "predicted_risk": float,     # risk for this actual patient
          "contributions": {feature_name: float, ...}   # signed, in probability points
        }
    Positive contribution = pushed risk UP relative to baseline;
    negative = pushed risk DOWN.
    """
    predicted_risk = engine.predict_tabular_risk(data_dict)
    baseline_risk = engine.predict_tabular_risk(_HEALTHY_BASELINE)

    model_bundle = engine._get_tabular_model()  # noqa: SLF001 -- intentional internal reuse

    if _HAS_SHAP and model_bundle is not None:
        try:
            return _explain_tabular_shap(data_dict, model_bundle, predicted_risk)
        except Exception:
            pass  # fall through to the manual method below

    return _explain_tabular_baseline(data_dict, predicted_risk, baseline_risk)


def _featurize(data_dict: dict) -> np.ndarray:
    age = float(data_dict.get("age", 50))
    hyp = int(bool(data_dict.get("hypertension", 0)))
    hd = int(bool(data_dict.get("heart_disease", 0)))
    glucose = float(data_dict.get("avg_glucose_level", 100.0))
    bmi = float(data_dict.get("bmi", 25.0))
    smoking_raw = str(data_dict.get("smoking_status", "Unknown")).strip().lower()
    smoking = engine._SMOKING_ENCODING.get(smoking_raw, 0.0)  # noqa: SLF001
    return np.array([[age, hyp, hd, glucose, bmi, smoking]])


def _explain_tabular_shap(data_dict, model_bundle, predicted_risk) -> dict:
    _, model = model_bundle
    features = _featurize(data_dict)
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(features)

    # Different sklearn/xgboost/shap version combos return slightly
    # different shapes for binary classification -- normalize to a flat
    # per-feature array for the positive class.
    values = np.asarray(shap_values)
    if values.ndim == 3:       # (n_classes, n_samples, n_features)
        values = values[1][0]
    elif values.ndim == 2:     # (n_samples, n_features)
        values = values[0]

    names = ["age", "hypertension", "heart_disease", "avg_glucose_level", "bmi", "smoking_status"]
    contributions = {name: float(val) for name, val in zip(names, values)}

    return {
        "method": "shap",
        "baseline_risk": None,  # SHAP's baseline is the model's expected value, not our healthy reference
        "predicted_risk": predicted_risk,
        "contributions": contributions,
    }


def _explain_tabular_baseline(data_dict, predicted_risk, baseline_risk) -> dict:
    contributions = {}
    for feature in _HEALTHY_BASELINE:
        # Swap ONLY this feature to the patient's actual value, everything
        # else stays at the healthy reference -- isolates this feature's
        # individual effect on risk.
        probe = dict(_HEALTHY_BASELINE)
        probe[feature] = data_dict.get(feature, _HEALTHY_BASELINE[feature])
        probe_risk = engine.predict_tabular_risk(probe)
        contributions[feature] = probe_risk - baseline_risk

    return {
        "method": "baseline_attribution",
        "baseline_risk": baseline_risk,
        "predicted_risk": predicted_risk,
        "contributions": contributions,
    }


# ==========================================================================
# CT / ECG: Grad-CAM
# ==========================================================================
#
# Grad-CAM highlights which pixels/spectrogram regions the CNN weighted
# most heavily when producing its prediction, by looking at the gradient
# of the output with respect to the last convolutional layer's activations.
# Works the same way for the CT ResNet18 and the ECG spectrogram-ResNet18,
# since both share that architecture.

def _grad_cam(model, input_tensor, target_layer):
    """Core Grad-CAM: returns a (H, W) heatmap in [0, 1], same spatial size
    as target_layer's output, upsampled to match input_tensor's H/W."""
    activations = {}
    gradients = {}

    def fwd_hook(_module, _inp, output):
        activations["value"] = output

    def bwd_hook(_module, _grad_in, grad_out):
        gradients["value"] = grad_out[0]

    handle_fwd = target_layer.register_forward_hook(fwd_hook)
    handle_bwd = target_layer.register_full_backward_hook(bwd_hook)

    try:
        model.zero_grad()
        output = model(input_tensor)  # raw logit, shape (1,1)
        output.backward()

        acts = activations["value"][0]      # (C, H, W)
        grads = gradients["value"][0]        # (C, H, W)
        weights = grads.mean(dim=(1, 2))     # (C,) -- global-average-pooled gradient per channel

        cam = torch.zeros(acts.shape[1:], dtype=torch.float32)
        for c, w in enumerate(weights):
            cam += w * acts[c]
        cam = F.relu(cam)
        cam = cam / (cam.max() + 1e-8)

        target_h, target_w = input_tensor.shape[-2], input_tensor.shape[-1]
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0), size=(target_h, target_w), mode="bilinear", align_corners=False
        )
        return cam.squeeze().detach().numpy()
    finally:
        handle_fwd.remove()
        handle_bwd.remove()


def _overlay_heatmap_on_image(base_image_uint8: np.ndarray, heatmap01: np.ndarray) -> np.ndarray:
    """Blends a Grad-CAM heatmap (values in [0,1]) over a base RGB image."""
    if _HAS_MATPLOTLIB:
        colored = (cm.jet(heatmap01)[:, :, :3] * 255).astype(np.uint8)
    else:
        # Fallback: simple red-channel-only heatmap if matplotlib is missing.
        colored = np.zeros_like(base_image_uint8)
        colored[..., 0] = (heatmap01 * 255).astype(np.uint8)

    overlay = (0.55 * base_image_uint8 + 0.45 * colored).astype(np.uint8)
    return overlay


def explain_ct_risk(file_path: str = None) -> Optional[np.ndarray]:
    """
    Returns a (224, 224, 3) uint8 image: the CT scan with a Grad-CAM
    heatmap overlaid showing which regions drove the risk prediction.
    Returns None if torch/torchvision aren't installed or no trained
    ct_resnet.pt checkpoint exists (a Grad-CAM over random-init weights
    is meaningless, so we don't produce one).
    """
    if not _HAS_TORCH or not engine._HAS_TORCHVISION:  # noqa: SLF001
        return None

    import os
    if not os.path.isfile(engine._CT_WEIGHTS_PATH):  # noqa: SLF001
        return None

    model = engine._get_ct_model()  # noqa: SLF001
    if model is None:
        return None

    img_array = engine._load_ct_image(file_path) or engine._generate_synthetic_ct_array()  # noqa: SLF001

    import torchvision.transforms as tv_transforms
    preprocess = tv_transforms.Compose(
        [
            tv_transforms.ToTensor(),
            tv_transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    tensor = preprocess(img_array).unsqueeze(0)
    tensor.requires_grad_(True)

    heatmap = _grad_cam(model, tensor, target_layer=model.layer4[-1])
    return _overlay_heatmap_on_image(img_array, heatmap)


def explain_ecg_risk(file_path: str = None) -> Optional[np.ndarray]:
    """
    Returns a (224, 224, 3) uint8 image: the ECG spectrogram with a
    Grad-CAM heatmap overlaid. Returns None if torch/torchvision aren't
    installed, the ECG model isn't the 2D spectrogram-CNN, or no trained
    ecg_cnn.pt checkpoint exists.
    """
    if not _HAS_TORCH or not engine._HAS_TORCHVISION:  # noqa: SLF001
        return None

    import os
    if not os.path.isfile(engine._ECG_WEIGHTS_PATH):  # noqa: SLF001
        return None

    model, kind = engine._get_ecg_model()  # noqa: SLF001
    if model is None or kind != "2d_resnet":
        return None

    signal = engine._load_ecg_from_file(file_path) if file_path else None  # noqa: SLF001
    if signal is None:
        signal = engine._generate_synthetic_ecg()  # noqa: SLF001

    spec_img = engine.ecg_signal_to_spectrogram_image(signal)

    tensor = torch.tensor(spec_img, dtype=torch.float32).permute(2, 0, 1).unsqueeze(0) / 255.0
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    tensor = (tensor - mean) / std
    tensor.requires_grad_(True)

    heatmap = _grad_cam(model, tensor, target_layer=model.layer4[-1])
    return _overlay_heatmap_on_image(spec_img, heatmap)


if __name__ == "__main__":
    # Quick smoke test of the tabular explainer (works without torch).
    sample = {
        "age": 67,
        "hypertension": 1,
        "heart_disease": 1,
        "avg_glucose_level": 210.5,
        "bmi": 31.2,
        "smoking_status": "formerly smoked",
    }
    result = explain_tabular_risk(sample)
    print(f"Method: {result['method']}")
    print(f"Baseline risk (healthy reference): {result['baseline_risk']}")
    print(f"Predicted risk (this patient):     {result['predicted_risk']:.4f}")
    print("\nPer-feature contribution (probability points, + = raises risk):")
    for feature, contribution in sorted(result["contributions"].items(), key=lambda kv: -abs(kv[1])):
        print(f"  {feature:20s} {contribution:+.4f}")
